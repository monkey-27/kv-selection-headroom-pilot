# PageDrop vs official TriAttention headline runbook

This directory prepares the full headline comparison without changing the
frozen `pagedrop/` implementation or any existing result. No cluster job is
launched by the local setup.

## Frozen method boundary

PageDrop uses `pagedrop.cache.PilotCache` exactly as validated in the completed
pilot: page-16 physical storage, 576-to-512 buffered eviction, 64 generated
tokens protected, prompt KV separate and permanent, independent choices per
layer, original positions, deterministic decoding. A model architecture must
pass an eight-item smoke test with this exact behavior or its cell is marked
blocked.

TriAttention always comes from the official repository at
`325297218a0d85cc83bc9ca1ecfa1a33a178831f`. Hardware audit selects TensorRT-LLM
only on SM100/SM103; that checkout is pinned to
`8a26dd8f9d6fd09781d7e6f5f1162674f9f8fd05`. Other GPUs use the official vLLM
integration and label results `triattention_official_vllm`. Missing model
support or calibration fails closed.

## Dartmouth order

1. Copy this repository to persistent scratch and set `PDVT_ROOT`,
   `PDVT_REPO`, and `PDVT_PYTHON`.
2. Run `scripts/setup_pd_vs_tri_dartmouth.sh` on a login node.
3. Submit `scripts/audit_pd_vs_tri_dartmouth.sbatch`. Inspect the resulting
   `configs/hardware.json` before installing or using TensorRT-LLM.
4. Fetch canonical benchmark sources into `raw/`, retaining source metadata,
   and build immutable manifests with `python -m pd_vs_tri.manifests`. The
   builders reject wrong fixed-set sizes, duplicate IDs, missing code dates,
   and LiveCodeBench items published on or before 2025-04-30.
5. Resolve exact Hugging Face model revisions into
   `configs/model_revisions.lock.json`; download those revisions without
   substitution.
6. Generate model-specific official calibration via the four-task array
   `scripts/calibrate_triattention_dartmouth.sbatch --array=0-3`. Never reuse
   calibration between models.
7. Run eight-item FullKV/PageDrop/TriAttention smoke tests for every intended
   model-domain cell. Store explicit pass/block artifacts below `smoke/`.
8. Run FullKV traces, then freeze all 16 K values once with
   `python -m pd_vs_tri.budgets`. The command refuses to overwrite an existing
   freeze.
9. Run the fail-closed preflight. Exit status 2 blocks submission:
   `python -m pd_vs_tri.preflight --root "$PDVT_ROOT" --config configs/pd_vs_tri_headline.json`.
10. Build `configs/execution_plan.json` with `python -m pd_vs_tri.planner`.
   Submit PageDrop and TriAttention jobs interleaved by cell and shard.
11. Run `scripts/pd_vs_tri_progress_loop.sh` as a lightweight CPU job. Workers
    also refresh progress after each shard. Reports use only paired completed
    problems. The existing frozen PageDrop seed rule is reused, so no additional
    PageDrop rollout seed is introduced; aggregation remains seed-aware if a
    future preregistered plan contains repeats.
12. After all supported cells terminate, generate
    `reports/pd_vs_tri_final.md` with `python -m pd_vs_tri.final_report`.

The backend-specific generator is supplied through `PDVT_GENERATE_COMMAND`.
It receives one request JSON and must atomically write the requested response
JSON with the raw completion, objective grade, generated-token count, achieved
KV compression, exact backend/config/calibration hashes, and runtime counters.
The generic runner never substitutes an unofficial TriAttention path.
It retries generator/process failures three times and saves every raw completion
under `raw_completions/` before objective grading. Terminal failures and OOMs
remain durable rows, so the paired denominator cannot shrink silently.
