# PageDrop source recovery

The canonical PageDrop-16 implementation is `pagedrop/`. It is copied exactly
from `archive/pagedrop16_frozen_20260917/pagedrop/`; both directory hashes are
`64bef46972d5378a6b59e8f47759078fdd671e76afba29f8dbee760b83f6b19c`.

The source was originally created as untracked files in the pilot workspace,
then staged at `/dartfs-hpc/scratch/f0082p5/pagedrop16`. It generated the
completed Qwen3-4B MATH500 and passcode results. The preserved request and
artifact linkage are documented in the archive provenance file.

Do not modify `pagedrop/` for later models. Architecture adapters must import
the frozen implementation and remain in a separate module.
