from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from .analyze import analyze
from .combinatorics import masks_by_cardinality
from .config import DEFAULT_CONFIG, load_config
from .exhaustive import _layout
from .io import append_jsonl, read_jsonl
from .replay import QwenReplayScorer, attention_topk_mask
from .runtime import GPUHourBudget
from .traces import _correct


def run(config_path: str) -> None:
    cfg = load_config(config_path)
    run_dir = Path(cfg["paths"]["run_dir"])
    marker = run_dir / "analysis" / "run_free_generation.json"
    if not marker.exists():
        analyze(config_path)
    if not json.loads(marker.read_text()).get("run"):
        print("free generation not triggered; skipping")
        return
    # CSV is intentionally read without pandas.
    import csv
    with (run_dir / "analysis" / "per_trace_budget.csv").open() as handle:
        k2 = [r for r in csv.DictReader(handle) if r["stage"] == "main" and int(r["k"]) == cfg["main"]["primary_k"]]
    k2.sort(key=lambda r: float(r["headroom"]), reverse=True)
    selected_ids = {r["trace_id"] for r in k2[:cfg["free_generation"]["n_traces"]]}
    traces = {t["trace_id"]: t for t in read_jsonl(run_dir / "traces" / "selected.jsonl") if t["trace_id"] in selected_ids}
    scorer = QwenReplayScorer.load(
        cfg["model"], cfg["dtype"], cfg["attention_backend"], cfg["model_revision"]
    )
    rt = cfg["runtime"]
    budget = GPUHourBudget(run_dir, rt["max_h100_equivalent_hours"], rt["gpu_h100_equivalent_factor"], rt["stop_margin_hours"])
    out = run_dir / "free_generation.jsonl"
    done = {(r["trace_id"], r["condition"], int(r["replicate"])) for r in read_jsonl(out)}
    rng = random.Random(cfg["seed"] + 71)
    with budget.active("free_generation"):
        for metric in k2[:cfg["free_generation"]["n_traces"]]:
            trace = traces[metric["trace_id"]]
            layout = _layout(trace, cfg, "main")
            attention_path = run_dir / "utilities" / "main" / trace["trace_id"] / "attention.json"
            attention = json.loads(attention_path.read_text())["block_mass"]
            oracle = int(metric["oracle_mask"])
            attn = attention_topk_mask(attention, cfg["main"]["primary_k"])
            random_masks = rng.sample(masks_by_cardinality(layout.n_blocks)[cfg["main"]["primary_k"]],
                                      cfg["free_generation"]["random_replicates"])
            conditions = [("oracle", 0, oracle), ("future_attention", 0, attn)]
            conditions += [("random", rep, mask) for rep, mask in enumerate(random_masks)]
            prefix = scorer.prefill(trace["prompt_ids"], trace["reasoning_ids"], layout)
            for condition, rep, subset in conditions:
                if (trace["trace_id"], condition, rep) in done:
                    continue
                budget.check()
                generated = scorer.generate_from_mask(
                    prefix, trace["reasoning_ids"], layout, subset,
                    cfg["free_generation"]["max_new_tokens"],
                    cfg["generation"]["temperature"], cfg["generation"]["top_p"],
                    cfg["seed"] + 1009 * rep + int(trace["example_index"]),
                )
                full_ids = trace["reasoning_ids"][:layout.checkpoint] + generated
                text = scorer.tokenizer.decode(full_ids, skip_special_tokens=True)
                row = {"trace_id": trace["trace_id"], "dataset": trace["dataset"],
                       "condition": condition, "replicate": rep, "mask": subset,
                       "n_generated": len(generated), "correct": _correct(text, trace["gold"], trace["dataset"]),
                       "completion_text": text}
                append_jsonl(out, [row]); done.add((trace["trace_id"], condition, rep))
                budget.flush(f"free_{trace['trace_id']}_{condition}_{rep}")
    analyze(config_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
