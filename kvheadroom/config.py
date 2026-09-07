from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "kv_headroom.json"


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    path = Path(path).resolve()
    cfg = json.loads(path.read_text())
    cfg["_config_path"] = str(path)
    cfg["_root"] = str(ROOT)
    run_dir = Path(cfg["paths"]["run_dir"])
    if not run_dir.is_absolute():
        run_dir = ROOT / run_dir
    cfg["paths"]["run_dir"] = str(run_dir)
    return cfg


def freeze_config(cfg: dict[str, Any]) -> Path:
    """Write once and reject protocol drift on resume."""
    source = Path(cfg["_config_path"])
    destination = Path(cfg["paths"]["run_dir"]) / "frozen_config.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = source.read_bytes()
    if destination.exists() and destination.read_bytes() != payload:
        raise RuntimeError(f"config differs from frozen run config: {destination}")
    if not destination.exists():
        destination.write_bytes(payload)
    return destination
