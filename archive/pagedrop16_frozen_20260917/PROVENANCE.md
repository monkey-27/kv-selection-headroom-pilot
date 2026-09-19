# Frozen PageDrop-16 source provenance

This directory is an untouched archival extraction of the source bundle used
to stage the completed Qwen3-4B PageDrop-16 pilot.

- Original local workspace: `/Users/Arjun/Documents/ChatGPT/iclr/kv_selection_headroom_pilot`
- Original Dartmouth workspace: `/dartfs-hpc/scratch/f0082p5/pagedrop16`
- Preserved bundle: `/private/tmp/pagedrop_dartmouth_bundle_20260917.tar.gz`
- Frozen `pagedrop/*.py` package SHA-256: `64bef46972d5378a6b59e8f47759078fdd671e76afba29f8dbee760b83f6b19c`
- Primary `pagedrop/cache.py` SHA-256: `15c64483d3092969437d800d68fb8dc30a3ece2a2c6657b2821ff29722fbfd77`
- Frozen request SHA-256: `f73ce3ca8a7403aef62ef6da2baf2eeb04a7cb66ea2ef20eb603948683a61a55`
- Completed local artifacts: `pagedrop_dartmouth_results/combined` in the original workspace

The completed pilot's `combined/execution.json` records the frozen request
hash above. That request records the package hash above. The merge program
refused to merge a shard unless its request, package, model revision, policies,
and dataset hashes matched the frozen request.

The source was recovered from the original untracked pilot workspace. It was
not present in the repository's earlier Git history. Files under this archive
must not be edited. Model support or execution changes belong outside this
directory and outside the frozen top-level `pagedrop/` package.
