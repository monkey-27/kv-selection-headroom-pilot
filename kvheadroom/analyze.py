from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .combinatorics import summarize_utilities
from .config import DEFAULT_CONFIG, load_config
from .io import atomic_json, read_jsonl
from .replay import attention_topk_mask


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    columns = list(rows[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _bootstrap_mean_ci(values: list[float], samples: int, seed: int) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    x = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), size=(samples, len(x)))].mean(1)
    return tuple(float(v) for v in np.quantile(means, [0.025, 0.975]))


def _stage(run_dir: Path, stage: str, n_blocks: int) -> tuple[list[dict], list[dict]]:
    budget_rows, trace_rows = [], []
    base = run_dir / "utilities" / stage
    if not base.exists():
        return budget_rows, trace_rows
    for trace_dir in sorted(p for p in base.iterdir() if p.is_dir()):
        rows = read_jsonl(trace_dir / "subsets.jsonl")
        expected = 1 << n_blocks
        if len({int(r["mask"]) for r in rows}) != expected:
            continue
        control = json.loads((trace_dir / "control.json").read_text())
        attention = json.loads((trace_dir / "attention.json").read_text())["block_mass"]
        summaries, nested = summarize_utilities(rows, n_blocks)
        by_mask = {int(r["mask"]): float(r["utility"]) for r in rows}
        u_full = -float(control["untouched_full_nll"])
        for item in summaries:
            attn_mask = attention_topk_mask(attention, item["k"])
            item.update({
                "stage": stage, "trace_id": trace_dir.name,
                "u_full": u_full,
                "memory_delta": u_full - item["u_empty"],
                "attention_mask": attn_mask,
                "u_attention": by_mask[attn_mask],
                "attention_regret": item["u_oracle"] - by_mask[attn_mask],
                "control_error": float(control["absolute_nll_error"]),
                "control_passed": bool(control["passed"]),
            })
            denom = item["headroom"]
            item["attention_headroom_fraction"] = (
                (item["u_attention"] - item["u_rand"]) / denom if denom > 1e-12 else None
            )
            item["nested_headroom_fraction"] = (
                (item["u_nested"] - item["u_rand"]) / denom if denom > 1e-12 else None
            )
            budget_rows.append(item)
        trace_rows.append({
            "stage": stage, "trace_id": trace_dir.name, "u_full": u_full,
            "u_empty": summaries[0]["u_empty"],
            "memory_delta": u_full - summaries[0]["u_empty"],
            "control_error": float(control["absolute_nll_error"]),
            "control_passed": bool(control["passed"]),
            "nested_order": json.dumps(nested["block_order"]),
        })
    return budget_rows, trace_rows


def _plot(rows: list[dict], run_dir: Path) -> None:
    import matplotlib.pyplot as plt

    natural = [r for r in rows if r["stage"] == "main"]
    if not natural:
        return
    figure_dir = run_dir / "plots"
    figure_dir.mkdir(parents=True, exist_ok=True)
    grouped = defaultdict(list)
    for row in natural:
        grouped[int(row["k"])].append(row)
    ks = sorted(grouped)
    med = lambda field: [np.nanmedian([r[field] for r in grouped[k] if r[field] is not None]) for k in ks]

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for field, label in (("u_rand", "exact random"), ("u_oracle", "oracle"),
                         ("u_attention", "future attention"), ("u_nested", "nested DP")):
        ax.plot(ks, med(field), marker="o", label=label)
    ax.set(xlabel="retained blocks k", ylabel="utility (mean token log-prob)")
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(figure_dir / "utility_vs_budget.png", dpi=180); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(ks, med("headroom"), marker="o")
    axes[0].axhline(0.08, color="grey", linestyle="--", linewidth=1)
    axes[0].set(xlabel="retained blocks k", ylabel="oracle headroom (nats/token)")
    axes[1].plot(ks, med("rr"), marker="o")
    axes[1].axhline(0.75, color="grey", linestyle="--", linewidth=1)
    axes[1].set(xlabel="retained blocks k", ylabel="random recovery")
    fig.tight_layout(); fig.savefig(figure_dir / "headroom_rr.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(ks, med("attention_regret"), marker="o", label="future attention")
    ax.plot(ks, med("nested_regret"), marker="o", label="nested DP")
    ax.set(xlabel="retained blocks k", ylabel="regret vs independent oracle (nats/token)")
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(figure_dir / "ranking_regret.png", dpi=180); plt.close(fig)


def _free_summary(run_dir: Path, expected_rows: int) -> dict | None:
    rows = read_jsonl(run_dir / "free_generation.jsonl")
    if not rows:
        return None
    by = defaultdict(list)
    for row in rows:
        by[row["condition"]].append(float(row["correct"]))
    accuracy = {key: float(np.mean(values)) for key, values in by.items()}
    comparator = max(accuracy.get("future_attention", float("nan")), accuracy.get("random", float("nan")))
    return {"n_rows": len(rows), "expected_rows": expected_rows, "complete": len(rows) == expected_rows,
            "accuracy": accuracy,
            "oracle_advantage_vs_best_comparator": accuracy.get("oracle", float("nan")) - comparator}


def analyze(config_path: str) -> dict:
    cfg = load_config(config_path)
    run_dir = Path(cfg["paths"]["run_dir"])
    analysis_dir = run_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    all_budget, all_traces = [], []
    for stage, blocks in (("synthetic", 12), ("main", 12), ("late", 8)):
        budget, traces = _stage(run_dir, stage, blocks)
        all_budget.extend(budget); all_traces.extend(traces)
    _write_csv(analysis_dir / "per_trace_budget.csv", all_budget)
    _write_csv(analysis_dir / "per_trace.csv", all_traces)
    _plot(all_budget, run_dir)

    v = cfg["verdict"]
    primary = cfg["main"]["primary_k"]
    natural = [r for r in all_budget if r["stage"] == "main" and r["k"] == primary]
    synthetic = [r for r in all_budget if r["stage"] == "synthetic" and r["k"] == primary]
    active = [r for r in natural if r["memory_delta"] >= v["memory_active_delta_nats"]]
    retrieval = read_jsonl(run_dir / "synthetic_retrieval.jsonl")
    retrieval_rate = float(np.mean([r["retrieved"] for r in retrieval])) if retrieval else None
    controls_ok = bool(all_traces) and all(r["control_passed"] for r in all_traces)

    median_h2 = float(np.median([r["headroom"] for r in active])) if active else None
    median_rr2 = float(np.median([r["rr"] for r in active if r["rr"] is not None])) if active else None
    ci = _bootstrap_mean_ci([r["headroom"] for r in active], v["bootstrap_samples"], cfg["seed"])
    substantial = bool(active) and (
        median_h2 >= cfg["free_generation"]["substantial_median_h2"] or
        median_rr2 < cfg["free_generation"]["substantial_rr2"]
    )
    fractions = []
    for row in active:
        vals = [x for x in (row["attention_headroom_fraction"], row["nested_headroom_fraction"]) if x is not None]
        if vals:
            fractions.append(max(vals))
    heuristic_fraction = float(np.median(fractions)) if fractions else None
    synthetic_large = bool(synthetic) and float(np.median([r["headroom"] for r in synthetic])) >= cfg["synthetic"]["large_headroom_nats"]
    synthetic_valid = retrieval_rate is not None and retrieval_rate >= cfg["synthetic"]["retrieval_threshold"] and synthetic_large

    atomic_json(analysis_dir / "run_free_generation.json", {
        "run": substantial, "reason": "natural fixed-replay headroom substantial" if substantial else "trigger not met"
    })
    atomic_json(analysis_dir / "run_late_stress.json", {
        "run": bool(natural) and not substantial,
        "reason": "natural oracle approximately random" if natural and not substantial else "trigger not met",
    })
    expected_free = cfg["free_generation"]["n_traces"] * (2 + cfg["free_generation"]["random_replicates"])
    free = _free_summary(run_dir, expected_free)
    core_complete = (len(natural) == 48 and len(synthetic) == cfg["synthetic"]["n_examples"]
                     and len(retrieval) == cfg["synthetic"]["n_examples"])
    late_primary = [r for r in all_budget if r["stage"] == "late" and r["k"] == primary]
    conditional_complete = ((substantial and free is not None and free["complete"])
                            or (not substantial and len(late_primary) == cfg["late_stress"]["n_traces"]))
    verdict, rationale = "INCONCLUSIVE", "required stages or quantitative gates are incomplete"
    if not core_complete or not conditional_complete:
        verdict, rationale = "INCONCLUSIVE", "required core or conditional stages are incomplete"
    elif not controls_ok:
        verdict, rationale = "INVALID", "one or more all-12 replay controls exceed 1e-3 nats/token"
    elif not synthetic_valid:
        verdict, rationale = "INVALID", "synthetic retrieval/headroom positive control did not pass"
    elif len(active) < v["minimum_memory_active_traces"]:
        verdict, rationale = "INCONCLUSIVE", "too few memory-active natural traces"
    elif (median_h2 <= v["strong_go_a_median_h2"] and median_rr2 >= v["strong_go_a_rr2"]
          and ci[1] < v["strong_go_a_upper_ci_mean_h2"]):
        verdict, rationale = "STRONG GO A", "natural headroom is tightly negligible while the synthetic control is positive"
    elif (substantial and free is not None and free["complete"] and heuristic_fraction is not None
          and heuristic_fraction <= v["heuristic_little_fraction"]):
        verdict, rationale = "STRONG GO B", "oracle headroom is large but attention and the best nested ranking recover little"
    elif (substantial and free is not None and free["complete"]
          and free["oracle_advantage_vs_best_comparator"] <= cfg["free_generation"]["disappears_accuracy_delta"]):
        verdict, rationale = "STRONG GO C", "fixed-replay oracle advantage disappears under trajectory adaptation"
    elif (substantial and heuristic_fraction is not None and heuristic_fraction >= v["heuristic_most_fraction"]
          and free is not None and free["complete"]
          and free["oracle_advantage_vs_best_comparator"] > cfg["free_generation"]["disappears_accuracy_delta"]):
        verdict, rationale = "KILL", "simple ranking recovers most headroom and the advantage survives free generation"

    result = {
        "verdict": verdict, "rationale": rationale,
        "n_natural": len(natural), "n_memory_active": len(active),
        "median_h2_memory_active": median_h2, "median_rr2_memory_active": median_rr2,
        "bootstrap_95_ci_mean_h2": ci, "median_best_heuristic_headroom_fraction": heuristic_fraction,
        "synthetic_retrieval_rate": retrieval_rate, "synthetic_large_headroom": synthetic_large,
        "all_cache_controls_passed": controls_ok, "substantial_fixed_replay_headroom": substantial,
        "free_generation": free,
    }
    atomic_json(analysis_dir / "aggregate.json", result)
    report = [
        f"# KV-selection-headroom verdict: {verdict}", "", rationale + ".", "",
        f"- Natural traces scored at k=2: {len(natural)}",
        f"- Memory-active traces: {len(active)}",
        f"- Median H2 (memory-active): {median_h2}",
        f"- Median RR2 (memory-active): {median_rr2}",
        f"- Bootstrap 95% CI, mean H2: {ci}",
        f"- Synthetic full-cache retrieval: {retrieval_rate}",
        f"- All-cache equivalence controls passed: {controls_ok}",
        f"- Free-generation result: {free}", "",
        "This verdict is protocol-scoped to Qwen3-4B, the collected MATH-500/GPQA-D traces, and the frozen replay checkpoints.",
    ]
    (run_dir / "VERDICT.md").write_text("\n".join(report) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()
    print(json.dumps(analyze(args.config), indent=2))


if __name__ == "__main__":
    main()
