from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .combinatorics import batched, masks_by_cardinality
from .config import DEFAULT_CONFIG, freeze_config, load_config
from .io import append_jsonl, atomic_json, read_jsonl
from .replay import QwenReplayScorer, ReplayLayout
from .runtime import GPUHourBudget
from .synthetic import build_examples


def _layout(trace: dict, cfg: dict, stage: str) -> ReplayLayout:
    prompt = len(trace["prompt_ids"])
    if stage in ("main", "synthetic"):
        p = cfg["main"]
        return ReplayLayout.main(prompt, p["checkpoint_reasoning_token"], p["target_tokens"],
                                 p["recent_tokens"], p["n_blocks"], p["block_tokens"])
    p = cfg["late_stress"]
    checkpoint = int(len(trace["reasoning_ids"]) * p["checkpoint_fraction"])
    checkpoint = min(checkpoint, len(trace["reasoning_ids"]) - p["target_tokens"])
    return ReplayLayout.late(prompt, checkpoint, p["target_tokens"], p["recent_tokens"], p["n_blocks"])


def _trace_dir(run_dir: Path, stage: str, trace_id: str) -> Path:
    return run_dir / "utilities" / stage / trace_id


def _retrieval_gate(scorer: QwenReplayScorer, examples: list[dict], run_dir: Path) -> None:
    out = run_dir / "synthetic_retrieval.jsonl"
    existing = {r["trace_id"] for r in read_jsonl(out)}
    for ex in examples:
        if ex["trace_id"] in existing:
            continue
        full = ex["prompt_ids"] + ex["reasoning_ids"][:896] + ex["query_ids"]
        x = torch.tensor([full], device=scorer.device)
        with torch.inference_mode():
            generated = scorer.model.generate(input_ids=x, max_new_tokens=12, do_sample=False, use_cache=True)
        text = scorer.tokenizer.decode(generated[0, len(full):], skip_special_tokens=True)
        append_jsonl(out, [{"trace_id": ex["trace_id"], "needed_value": ex["needed_value"],
                            "generated": text, "retrieved": ex["needed_value"].lower() in text.lower()}])


def score_trace(scorer: QwenReplayScorer, trace: dict, cfg: dict, stage: str,
                budget: GPUHourBudget) -> None:
    layout = _layout(trace, cfg, stage)
    trace_dir = _trace_dir(Path(cfg["paths"]["run_dir"]), stage, trace["trace_id"])
    trace_dir.mkdir(parents=True, exist_ok=True)
    rows_path = trace_dir / "subsets.jsonl"
    seen = {int(r["mask"]) for r in read_jsonl(rows_path)}
    prefix = scorer.prefill(trace["prompt_ids"], trace["reasoning_ids"], layout)

    attention_path = trace_dir / "attention.json"
    if not attention_path.exists():
        masses = scorer.future_attention_mass(prefix, trace["reasoning_ids"], layout)
        atomic_json(attention_path, {"block_mass": masses})

    grouped = masks_by_cardinality(layout.n_blocks)
    batch_size = cfg["main" if stage != "late" else "late_stress"]["subset_batch_size"]
    for k in range(layout.n_blocks + 1):
        missing = [mask for mask in grouped[k] if mask not in seen]
        for mask_batch in batched(missing, batch_size):
            budget.check()
            scored = scorer.score_masks(prefix, trace["reasoning_ids"], layout, mask_batch)
            for row in scored:
                row.update({"trace_id": trace["trace_id"], "stage": stage})
            append_jsonl(rows_path, scored)
            seen.update(mask_batch)
            budget.flush(f"{stage}_{trace['trace_id']}_k{k}")

    control_path = trace_dir / "control.json"
    if not control_path.exists():
        untouched_cache = scorer.prefill(trace["prompt_ids"], trace["reasoning_ids"], layout)
        untouched = scorer.score_untouched(untouched_cache, trace["reasoning_ids"], layout)
        all_mask = (1 << layout.n_blocks) - 1
        subset = next(r for r in read_jsonl(rows_path) if int(r["mask"]) == all_mask)
        atomic_json(control_path, {
            "untouched_full_nll": untouched["nll"],
            "all_subset_nll": subset["nll"],
            "absolute_nll_error": abs(untouched["nll"] - subset["nll"]),
            "tolerance": 1e-3,
            "passed": abs(untouched["nll"] - subset["nll"]) <= 1e-3,
            "layout": {
                "prompt_tokens": layout.prompt_tokens, "checkpoint": layout.checkpoint,
                "target_tokens": layout.target_tokens, "recent_tokens": layout.recent_tokens,
                "block_sizes": [len(b) for b in layout.blocks],
            },
        })


def run(stage: str, config_path: str) -> None:
    cfg = load_config(config_path)
    run_dir = Path(cfg["paths"]["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    freeze_config(cfg)
    rt = cfg["runtime"]
    budget = GPUHourBudget(run_dir, rt["max_h100_equivalent_hours"],
                           rt["gpu_h100_equivalent_factor"], rt["stop_margin_hours"])
    scorer = QwenReplayScorer.load(
        cfg["model"], cfg["dtype"], cfg["attention_backend"], cfg["model_revision"]
    )
    import transformers
    provenance = {
        "upstream_random_attention_commit": "64db9688a12f0926db45fec039370f6ceb1ab4fe",
        "model": cfg["model"], "requested_model_revision": cfg["model_revision"],
        "resolved_model_commit": getattr(scorer.model.config, "_commit_hash", None),
        "torch": torch.__version__, "transformers": transformers.__version__,
        "cuda_device": torch.cuda.get_device_name(0),
    }
    provenance_path = run_dir / "provenance.json"
    if provenance_path.exists():
        previous = json.loads(provenance_path.read_text())
        if previous.get("resolved_model_commit") != provenance["resolved_model_commit"]:
            raise RuntimeError("resolved model revision changed across resume")
    else:
        atomic_json(provenance_path, provenance)
    if stage == "synthetic":
        traces = build_examples(scorer.tokenizer, cfg["synthetic"]["n_examples"])
        atomic_json(run_dir / "synthetic_examples.json", traces)
        _retrieval_gate(scorer, traces, run_dir)
    else:
        traces = read_jsonl(run_dir / "traces" / "selected.jsonl")
        if stage == "main" and len(traces) != 48:
            raise RuntimeError(f"main stage requires exactly 48 selected traces, found {len(traces)}")
        if stage == "late":
            marker = run_dir / "analysis" / "run_late_stress.json"
            if not marker.exists() or not json.loads(marker.read_text()).get("run"):
                print("late-context stress test not triggered; skipping")
                return
            p = cfg["late_stress"]
            eligible = [t for t in traces if len(t["reasoning_ids"]) >= p["target_tokens"] + p["recent_tokens"] + 8]
            traces = sorted(eligible, key=lambda t: len(t["reasoning_ids"]), reverse=True)[:p["n_traces"]]
            if len(traces) != p["n_traces"]:
                raise RuntimeError("not enough long traces for late-context stress test")
    with budget.active(f"exhaustive_{stage}"):
        for trace in traces:
            score_trace(scorer, trace, cfg, stage, budget)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("synthetic", "main", "late"))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()
    run(args.stage, args.config)


if __name__ == "__main__":
    main()
