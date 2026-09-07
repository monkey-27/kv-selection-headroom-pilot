# KV-selection-headroom pilot

This repository is based on Random Attention commit
`64db9688a12f0926db45fec039370f6ceb1ab4fe`. The pilot asks whether natural
reasoning traces contain causal token-selection headroom beyond exact random
block retention.

## Frozen cohort and intervention

- Model: `Qwen/Qwen3-4B`, BF16.
- Cohort: 24 correct MATH-500 traces and 24 correct GPQA-Diamond traces, each
  with at least 960 generated tokens. Candidate order and sampling are seeded.
- Main checkpoint: immediately before processing generated reasoning token 896
  (zero-based token 895). The query token completes the protected recent-128
  window; the cache contains the other 127 recent tokens.
- Selectable region: generated reasoning tokens 0--767, in twelve contiguous
  64-token blocks. The prompt is always retained.
- Target: teacher-forced original reasoning tokens 896--959. Utility is mean
  token log-probability, so `utility = -NLL`.
- Every one of the 4096 subsets is physically gathered from the full prefix K/V
  cache. Cached keys keep the RoPE rotation from their original absolute
  position. Target `position_ids` and `cache_position` also remain absolute.
- Subsets are batched only with equal cardinality. Therefore all rows in a GPU
  batch have identical compacted cache length.

For each trace and budget, analysis reports exact random mean, independent
oracle, oracle headroom, random recovery, future-attention top-k utility, and
the regret of the globally best nested chain. The nested chain maximizes the
sum of utility over all budgets by dynamic programming; it is not an additive
block-score approximation.

## Controls and conditional stages

The all-block gathered-cache NLL must match a fresh untouched full-cache replay
within `1e-3` nats/token. Sixteen synthetic once-stated-register examples use
the identical 12-by-64 layout. The needed value occurs in one old block only;
full-cache greedy retrieval must be at least 90%, and median synthetic H2 must
be at least 0.08 nats/token.

Natural headroom is called substantial when memory-active traces have median
H2 at least 0.08 or median RR2 below 0.75. Only then are the 24 highest-H2
traces freely regenerated from oracle, future-attention, and seeded random
`k=2` caches. Otherwise, the 16 longest eligible traces receive the registered
75%-trajectory, eight-block exhaustive stress test.

The qualitative verdict language in the request is operationalized before
execution: “little” heuristic recovery is at most 25% of oracle headroom,
“most” is at least 75%, and free-generation disappearance is an oracle accuracy
advantage of at most 5 percentage points over the better comparator. Values
between registered gates produce `INCONCLUSIVE`, not a post-hoc threshold.

## Runtime and resume

`runtime_ledger.json` records active GPU seconds across resumed invocations.
The process refuses new work at 9.75 H100-equivalent hours, leaving a 15-minute
safety margin below the hard 10-hour cap. Every subset batch is fsynced before
the next batch. Relaunching skips completed masks, traces, retrieval controls,
and free-generation conditions.

Prepare the environment and data, then run:

```bash
pip install -r requirements-headroom.txt
python3 scripts/download_data.py  # GPQA-Diamond requires accepted Hub terms
bash scripts/run_headroom.sh
```

Artifacts land in `runs/kv_headroom_v1/`: frozen config, selected traces,
per-subset JSONL, per-trace/budget CSV, aggregate JSON, conditional-generation
rows, three plots, runtime ledger, and `VERDICT.md`.

For Modal, `modal_app.py` provides matched A100-40GB and H100 throughput probes
plus full-run entrypoints. Remote artifacts are committed batch-by-batch to
`kv-headroom-output`; model/data downloads are cached in
`kv-headroom-hf-cache`.
