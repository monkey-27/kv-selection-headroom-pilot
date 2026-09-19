# Decisive PageDrop-16 pilot

Implemented in the existing Random Attention checkout. GPU validation and scoring
are pending: both `pranav-gupta-4-1-07` and `collaborator_abhiram` rejected the
initial launch due to workspace spend limits. No GPU started on either attempt.

## Execution

From the checkout root, on the original profile (existing dataset/model volumes):

```sh
MODAL_PROFILE=pranav-gupta-4-1-07 python3 -m modal run pagedrop_modal.py
```

For the active profile, the exact original MATH500 JSONL and 16 synthetic fixtures
have already been copied to `pagedrop16-output`:

```sh
PD_OUTPUT_VOLUME=pagedrop16-output PD_CACHE_VOLUME=pagedrop16-hf-cache MODAL_PROFILE=collaborator_abhiram python3 -m modal run pagedrop_modal.py
```

Another profile needs these same two frozen files in its output volume. Never
rebuild or substitute the once-stated fixtures. Model revision is pinned to the
existing pilot's resolved Qwen3-4B commit. No access tokens are created or stored.

One H100; platform timeout 14,100 seconds; no automatic retries. The active ledger
allows 3.82 hours including a conservative 60-second charge per wakeup;
generation heartbeats persist consumed time every 30 seconds, reserving
startup/synchronization overhead and lost heartbeat time on abrupt failure.
Before any resume, account for elapsed GPU container time across prior attempts,
and reduce platform timeout/ledger allowance so cumulative GPU allocation stays
under four hours. A missing/partial cell cannot produce a scientific verdict.

## Protocol

Exactly random_pp, shared random token, and shared PageDrop-16 on all 500 canonical
examples. Original Qwen instruct question format, no few-shot, original chat
wrapper and boxed-answer instruction, original symbolic grader, original 32,768
**total-token** generation cap. User-requested greedy decoding overrides original
sampling. BF16 throughout. Batch 32 heterogeneous prompts; right-padded causal
prefill; valid prompt lengths in decode; finished requests leave the active batch.
Eviction RNG is fixed independently per example/layer and is stable under resume.

Unlike the paper's original budget accounting, the prompt does not consume the
512 generated-token budget. All prompt KV remains permanently protected. As in
the source buffered engine, eviction reduces retained generated KV to 512 after
64 new tokens have accumulated: physical retained generated occupancy ranges
512–575 between rounds. The latest 64 generated tokens are always protected;
between rounds the protected tail may be longer. Rounds occur at generated
positions 576, 640, 704, …, making both the old/recent split and generated-origin
16-token page boundaries exact. No extra partial-page protection or token fills.

random_pp retains 448 uniformly selected old individual tokens independently per
KV head/layer, plus latest 64. The shared-token policy changes only head sharing.
PageDrop retains 28 uniformly selected old complete 16-token pages plus 4 recent
pages, shared across heads, independently per layer. Original post-RoPE keys are
never renumbered or rerotated. Native page-table and free-list updates release
physical block IDs to a reusable pool; KV tensors stay in place. Token policies
perform per-head gather/compaction as in the source. Permanent prompt and generated
attention softmax partitions are exactly combined using FlashInfer merge_state.
All policies use the same runtime so attention implementation is matched.

FlashInfer 0.6.18 FA2 is pinned. A lengths-only decode plan is reused across layers;
its pinned implementation's physical page-index buffer is updated per layer.
GPU validation compares this attention with direct SDPA, including differing
prompt lengths, independent layer tables, first eviction, page reuse, and shrinking.
No scored smoke examples or extra generations are used.

Cached `kv_headroom_v1/synthetic_examples.json` supplies 16 once-stated immutable
register values (the existing boundary fixtures). Teacher-force their exact first
896 reasoning tokens and original query through the decode/eviction path, then
continue greedily for 32 tokens. Exact leading-value recall is recorded. The
sanity results do not tune policies or determine the accuracy gate.

## Frozen operational gates and reports

Paired bootstrap: 10,000 resamples of canonical MATH example IDs, analysis RNG
20260915. Primary difference is PageDrop minus shared token. STRONG GO requires
absolute difference ≤2 pp, primary CI lower bound ≥−3 pp, absolute PageDrop versus
random_pp difference ≤3 pp, and longest common-reference quartile degradation
≤5 pp versus shared token. Longest groups use random_pp trace lengths and stable
canonical-ID tie ordering, avoiding policy-specific unpaired bins. Shared-token
accuracy more than 3 pp below random_pp kills the formulation. PageDrop differences
below −3 pp with paired CI excluding zero in the harmful direction produce KILL;
otherwise unresolved gates produce INCONCLUSIVE. No tuning after inspection.

Atomic per-example JSON includes original example ID, output, generated IDs,
correctness, trace length, termination and prompt length. Frozen request, data/code
hashes, model revision, numerical validation and cumulative ledger are persisted.
Aggregate report includes all paired differences/CIs, three pairwise correctness
matrices, eight joint correctness patterns, length-quartile results and sanity.

Only STRONG GO triggers the nine requested eviction microbenchmark cells:
8K/16K/32K × batch 1/8/32. Source Random Attention eviction versus the very same
native PageDrop physical-page release utility used during accuracy evaluation.
CUDA events; 20 warmups; 200–500 measured repetitions; block-median stability
check. Includes random selection, compaction or retained-table/free-list updates;
excludes allocation, reset, attention and model work. Times are per KV layer,
not full serving throughput. A released page means reusable capacity inside a
preallocated pool, not torch allocator memory reclamation.
