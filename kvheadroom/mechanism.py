from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

from .combinatorics import batched, masks_by_cardinality
from .exhaustive import _layout
from .io import atomic_json, read_jsonl
from .replay import QwenReplayScorer
from .runtime import GPUHourBudget
from .synthetic import build_examples


def _load(path: str) -> dict:
    return json.loads(Path(path).read_text())


def _stable_seed(base: int, trace_id: str) -> int:
    digest = hashlib.sha256(trace_id.encode()).digest()
    return base + int.from_bytes(digest[:4], "big")


def _select_natural(cfg: dict, source: Path) -> list[dict]:
    with (source / "analysis" / "per_trace_budget.csv").open() as handle:
        rows = [r for r in csv.DictReader(handle) if r["stage"] == "main" and int(r["k"]) == cfg["k"]]
    rows.sort(key=lambda r: (-float(r["headroom"]), r["trace_id"]))
    traces = {r["trace_id"]: r for r in read_jsonl(source / "traces" / "selected.jsonl")}
    need = max(cfg["depths"]) + cfg["target_tokens"]
    eligible = [r for r in rows if len(traces[r["trace_id"]]["reasoning_ids"]) >= 896 + need]
    if len(eligible) < cfg["n_natural"]:
        raise RuntimeError(f"only {len(eligible)} top-H2 traces have the required original horizon")
    return [dict(traces[r["trace_id"]], fixed_h2=float(r["headroom"]))
            for r in eligible[:cfg["n_natural"]]]


def _select_synthetic(cfg: dict, source: Path, tokenizer) -> list[dict]:
    successful = {r["trace_id"] for r in read_jsonl(source / "synthetic_retrieval.jsonl") if r["retrieved"]}
    examples = [x for x in build_examples(tokenizer, 16) if x["trace_id"] in successful]
    if len(examples) < cfg["n_synthetic"]:
        raise RuntimeError(f"only {len(examples)} successful synthetic controls")
    return examples[:cfg["n_synthetic"]]


def _summarize(rows: list[dict]) -> dict:
    empty = next(r["utility"] for r in rows if r["mask"] == 0)
    k2 = [r for r in rows if r["k"] == 2]
    mean = float(np.mean([r["utility"] for r in k2]))
    best = max(k2, key=lambda r: (r["utility"], -r["mask"]))
    denom = best["utility"] - empty
    return {"u_empty": empty, "u_rand": mean, "u_oracle": best["utility"],
            "oracle_mask": best["mask"], "h2": best["utility"] - mean,
            "rr2": (mean - empty) / denom if abs(denom) > 1e-12 else None}


def _score_landscape(scorer, prefix, trace, layout, continuation, d, masks, batch_size):
    rows = []
    for group in ([0], masks):
        for chunk in batched(group, batch_size):
            rows.extend(scorer.score_continuation_masks(
                prefix, trace["reasoning_ids"], layout, chunk, continuation, d))
    return rows


def run(config_path: str) -> dict:
    cfg = _load(config_path)
    source, run_dir = Path(cfg["source_run_dir"]), Path(cfg["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(run_dir / "frozen_config.json", cfg)
    scorer = QwenReplayScorer.load(cfg["model"], cfg["dtype"], cfg["attention_backend"], cfg["model_revision"])
    natural = _select_natural(cfg, source)
    synthetic = _select_synthetic(cfg, source, scorer.tokenizer)
    cohort = natural + synthetic
    atomic_json(run_dir / "cohort.json", [{
        "trace_id": t["trace_id"], "dataset": t["dataset"], "fixed_h2": t.get("fixed_h2"),
        "reasoning_tokens": len(t["reasoning_ids"]), "needed_block": t.get("needed_block"),
    } for t in cohort])

    rt = cfg["runtime"]
    budget = GPUHourBudget(run_dir, rt["max_h100_equivalent_hours"],
                           rt["gpu_h100_equivalent_factor"], rt["stop_margin_hours"])
    k2_masks = masks_by_cardinality(12)[cfg["k"]]
    max_needed = max(cfg["depths"]) + cfg["target_tokens"]
    trajectory_dir = run_dir / "trajectories"
    utility_dir = run_dir / "utilities"
    with budget.active("adaptation_mechanism"):
        for trace in cohort:
            layout = _layout(trace, {"main": {
                "checkpoint_reasoning_token": 896, "target_tokens": cfg["target_tokens"],
                "recent_tokens": 128, "n_blocks": 12, "block_tokens": 64}}, "main")
            prefix = scorer.prefill(trace["prompt_ids"], trace["reasoning_ids"], layout)
            rng = random.Random(_stable_seed(cfg["seed"], trace["trace_id"]))
            intervention_mask = rng.choice(k2_masks)
            trajectory_path = trajectory_dir / f"{trace['trace_id']}.json"
            if trajectory_path.exists():
                trajectory = json.loads(trajectory_path.read_text())
                adapted = trajectory["adapted_ids"]
                if int(trajectory["intervention_mask"]) != intervention_mask:
                    raise RuntimeError("resume intervention mask mismatch")
            else:
                adapted = scorer.generate_from_mask(
                    prefix, trace["reasoning_ids"], layout, intervention_mask, max_needed,
                    temperature=0.0, top_p=1.0, seed=_stable_seed(cfg["seed"], trace["trace_id"]),
                    stop_at_eos=False)
                if len(adapted) != max_needed:
                    raise RuntimeError(f"short deterministic trajectory for {trace['trace_id']}")
                original = list(trace["reasoning_ids"][896:896 + max_needed])
                extension = None
                if len(original) < max_needed:
                    if trace["dataset"] != "synthetic":
                        raise RuntimeError(f"short original trajectory for {trace['trace_id']}")
                    base = list(trace["reasoning_ids"][896:])
                    original = (base * ((max_needed + len(base) - 1) // len(base)))[:max_needed]
                    extension = "cycled predeclared query-answer target"
                trajectory = {
                    "trace_id": trace["trace_id"], "dataset": trace["dataset"],
                    "intervention_mask": intervention_mask, "adapted_ids": adapted,
                    "original_ids": original, "synthetic_original_extension": extension,
                    "adapted_text": scorer.tokenizer.decode(adapted, skip_special_tokens=True),
                    "original_text": scorer.tokenizer.decode(original, skip_special_tokens=True),
                    "needed_value": trace.get("needed_value"),
                    "depth_text": {str(d): {
                        "adapted_prefix": scorer.tokenizer.decode(adapted[:d], skip_special_tokens=True),
                        "adapted_target": scorer.tokenizer.decode(adapted[d:d + cfg["target_tokens"]], skip_special_tokens=True),
                        "original_prefix": scorer.tokenizer.decode(original[:d], skip_special_tokens=True),
                        "original_target": scorer.tokenizer.decode(original[d:d + cfg["target_tokens"]], skip_special_tokens=True),
                    } for d in cfg["depths"]},
                    "deleted_block_text": [
                        scorer.tokenizer.decode(trace["reasoning_ids"][64 * b:64 * (b + 1)], skip_special_tokens=True)
                        for b in range(12) if not (intervention_mask & (1 << b))
                    ],
                }
                atomic_json(trajectory_path, trajectory)
            for condition, continuation in (("adapted", trajectory["adapted_ids"]),
                                            ("original", trajectory["original_ids"])):
                for d in cfg["depths"]:
                    path = utility_dir / trace["dataset"] / trace["trace_id"] / condition / f"d{d}.json"
                    if path.exists():
                        continue
                    budget.check()
                    rows = _score_landscape(scorer, prefix, trace, layout, continuation, d,
                                            k2_masks, cfg["subset_batch_size"])
                    atomic_json(path, {"trace_id": trace["trace_id"], "dataset": trace["dataset"],
                                       "condition": condition, "d": d, "rows": rows,
                                       "summary": _summarize(rows)})
                    budget.flush(f"{trace['trace_id']}_{condition}_d{d}")
    return analyze(config_path)


def _bootstrap(values: list[float], samples: int, seed: int) -> list[float] | None:
    if not values:
        return None
    rng = np.random.default_rng(seed)
    x = np.asarray(values)
    means = x[rng.integers(0, len(x), size=(samples, len(x)))].mean(1)
    return [float(np.quantile(means, .025)), float(np.quantile(means, .975))]


def analyze(config_path: str) -> dict:
    cfg = _load(config_path); run_dir = Path(cfg["run_dir"])
    rows = []
    for path in sorted((run_dir / "utilities").glob("*/*/*/d*.json")):
        item = json.loads(path.read_text()); s = item["summary"]
        rows.append({"trace_id": item["trace_id"], "dataset": item["dataset"],
                     "condition": item["condition"], "d": item["d"], **s})
    analysis_dir = run_dir / "analysis"; analysis_dir.mkdir(parents=True, exist_ok=True)
    fields = ["trace_id", "dataset", "condition", "d", "u_empty", "u_rand", "u_oracle", "oracle_mask", "h2", "rr2"]
    with (analysis_dir / "per_trace.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fields); writer.writeheader(); writer.writerows(rows)

    aggregate = {"complete": len(rows) == (cfg["n_natural"] + cfg["n_synthetic"]) * 2 * len(cfg["depths"]),
                 "expected_landscapes": (cfg["n_natural"] + cfg["n_synthetic"]) * 2 * len(cfg["depths"]),
                 "n_landscapes": len(rows), "curves": {}}
    for dataset_group, datasets in (("natural", {"math500", "gpqa_diamond"}), ("synthetic", {"synthetic"})):
        aggregate["curves"][dataset_group] = {}
        for condition in ("adapted", "original"):
            aggregate["curves"][dataset_group][condition] = {}
            for d in cfg["depths"]:
                vals = [r for r in rows if r["dataset"] in datasets and r["condition"] == condition and r["d"] == d]
                aggregate["curves"][dataset_group][condition][str(d)] = {
                    "n": len(vals), "median_h2": float(np.median([r["h2"] for r in vals])) if vals else None,
                    "mean_h2": float(np.mean([r["h2"] for r in vals])) if vals else None,
                    "mean_h2_bootstrap_95_ci": _bootstrap([r["h2"] for r in vals], cfg["bootstrap_samples"], cfg["seed"] + d),
                    "median_rr2": float(np.median([r["rr2"] for r in vals if r["rr2"] is not None])) if vals else None,
                }
    by_trace = defaultdict(dict)
    for r in rows:
        if r["dataset"] != "synthetic" and r["condition"] == "adapted": by_trace[r["trace_id"]][r["d"]] = r["h2"]
    sequences = [v for v in by_trace.values() if len(v) == len(cfg["depths"])]
    tol = cfg["near_monotonic_tolerance_nats"]
    aggregate["natural_decay_fraction"] = {
        "monotonic": float(np.mean([all(v[b] <= v[a] for a, b in zip(cfg["depths"], cfg["depths"][1:])) for v in sequences])) if sequences else None,
        "near_monotonic": float(np.mean([all(v[b] <= v[a] + tol for a, b in zip(cfg["depths"], cfg["depths"][1:])) for v in sequences])) if sequences else None,
    }
    def med(group, condition, d): return aggregate["curves"][group][condition][str(d)]["median_h2"]
    d0, d1 = cfg["depths"][0], cfg["depths"][-1]; gates = cfg["gates"]
    nat_decay = med("natural", "adapted", d0) - med("natural", "adapted", d1) if rows else None
    syn_decay = med("synthetic", "adapted", d0) - med("synthetic", "adapted", d1) if rows else None
    matched_gap = med("natural", "original", d1) - med("natural", "adapted", d1) if rows else None
    aggregate["interaction"] = {"natural_adapted_decay_16_to_128": nat_decay,
                                "synthetic_adapted_decay_16_to_128": syn_decay,
                                "natural_minus_synthetic_decay": nat_decay - syn_decay if rows else None,
                                "matched_original_minus_adapted_at_128": matched_gap}
    verdict = "INCOMPLETE"
    if aggregate["complete"]:
        if syn_decay > gates["synthetic_max_collapse_h2"]:
            verdict = "CONFOUNDED: SYNTHETIC COLLAPSE"
        elif (nat_decay >= gates["substantial_decay_h2"] and med("natural", "adapted", d1) <= gates["adapted_near_zero_h2"]
              and matched_gap >= gates["matched_original_gap_h2"]):
            verdict = "STRONG ADAPTATION MECHANISM EVIDENCE"
        elif abs(matched_gap) < gates["matched_original_gap_h2"]:
            verdict = "HORIZON/POSITION EFFECT"
        else:
            verdict = "MIXED MECHANISM EVIDENCE"
    aggregate["verdict"] = verdict
    atomic_json(analysis_dir / "aggregate.json", aggregate)
    qualitative = []
    for trace_id, values in sorted(by_trace.items(), key=lambda x: x[1].get(d0, 0) - x[1].get(d1, 0), reverse=True)[:6]:
        trajectory = json.loads((run_dir / "trajectories" / f"{trace_id}.json").read_text())
        qualitative.append({"trace_id": trace_id,
                            "h2_adapted_by_depth": {str(k): values[k] for k in sorted(values)},
                            "intervention_mask": trajectory["intervention_mask"],
                            "depth_text": trajectory["depth_text"],
                            "deleted_block_text": trajectory["deleted_block_text"]})
    atomic_json(analysis_dir / "qualitative_examples.json", qualitative)
    _plots(rows, run_dir, cfg)
    report = [f"# Adaptation mechanism verdict: {verdict}", "", json.dumps(aggregate["interaction"], indent=2), "",
              f"Natural monotonic/near-monotonic decay fractions: {aggregate['natural_decay_fraction']}", "",
              "Qualitative trajectory and deleted-block excerpts are saved in analysis/qualitative_examples.json.", "",
              "This assay is restricted to Qwen3-4B and the frozen top-H2 natural and successful synthetic cohorts."]
    (run_dir / "VERDICT.md").write_text("\n".join(report) + "\n")
    return aggregate


def _plots(rows, run_dir, cfg):
    if not rows: return
    import matplotlib.pyplot as plt
    plot_dir = run_dir / "plots"; plot_dir.mkdir(parents=True, exist_ok=True)
    for metric in ("h2", "rr2"):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True)
        for ax, (label, datasets) in zip(axes, (("natural", {"math500", "gpqa_diamond"}), ("synthetic", {"synthetic"}))):
            for condition in ("adapted", "original"):
                ys = [np.median([r[metric] for r in rows if r["dataset"] in datasets and r["condition"] == condition and r["d"] == d and r[metric] is not None]) for d in cfg["depths"]]
                ax.plot(cfg["depths"], ys, marker="o", label=condition)
            ax.set(title=label, xlabel="adaptation depth d", ylabel=metric.upper()); ax.legend(frameon=False)
        fig.tight_layout(); fig.savefig(plot_dir / f"{metric}_curves.png", dpi=180); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--config", default="configs/mechanism_adaptation.json")
    args = parser.parse_args(); print(json.dumps(run(args.config), indent=2))


if __name__ == "__main__": main()
