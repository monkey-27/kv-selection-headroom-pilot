from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable


_MODAL_VOLUME = None


def commit_remote_volume() -> None:
    """Commit the mounted Modal output volume when running remotely."""
    global _MODAL_VOLUME
    name = os.environ.get("KVH_MODAL_VOLUME_NAME")
    if not name:
        return
    if _MODAL_VOLUME is None:
        import modal
        _MODAL_VOLUME = modal.Volume.from_name(name)
    _MODAL_VOLUME.commit()


def read_jsonl(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    # JSONL records are delimited by LF, not every Unicode character Python's
    # splitlines() recognizes. GPQA contains U+2028 inside a valid JSON string.
    return [json.loads(line) for line in path.read_text().split("\n") if line.strip()]


def append_jsonl(path: str | Path, rows: Iterable[dict], *, commit: bool = True) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    if commit:
        commit_remote_volume()


def atomic_json(path: str | Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)
    commit_remote_volume()
