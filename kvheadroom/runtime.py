from __future__ import annotations

import json
import time
from contextlib import contextmanager
from pathlib import Path

from .io import commit_remote_volume


class GPUHourBudget:
    """Persistent active-compute ledger; idle time between resumes is excluded."""

    def __init__(self, run_dir: str | Path, cap_hours: float, factor: float, margin_hours: float):
        self.path = Path(run_dir) / "runtime_ledger.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.cap = float(cap_hours)
        self.factor = float(factor)
        self.margin = float(margin_hours)
        self.active_start: float | None = None
        self.data = {"active_seconds": 0.0, "segments": []}
        if self.path.exists():
            self.data = json.loads(self.path.read_text())

    @property
    def used_hours(self) -> float:
        active = float(self.data["active_seconds"])
        if self.active_start is not None:
            active += time.monotonic() - self.active_start
        return active * self.factor / 3600.0

    def check(self) -> None:
        if self.used_hours >= self.cap - self.margin:
            raise RuntimeError(
                f"H100-equivalent safety stop: {self.used_hours:.3f} h used; "
                f"cap={self.cap:.3f} h, margin={self.margin:.3f} h"
            )

    def flush(self, label: str) -> None:
        if self.active_start is not None:
            elapsed = time.monotonic() - self.active_start
            self.data["active_seconds"] = float(self.data["active_seconds"]) + elapsed
            self.data["segments"].append({"label": label, "seconds": elapsed, "ended_unix": time.time()})
            self.active_start = time.monotonic()
        self.data["h100_equivalent_hours"] = self.used_hours
        self.data["cap_hours"] = self.cap
        self.path.write_text(json.dumps(self.data, indent=2) + "\n")
        commit_remote_volume()

    @contextmanager
    def active(self, label: str):
        self.check()
        self.active_start = time.monotonic()
        try:
            yield self
        finally:
            elapsed = time.monotonic() - self.active_start
            self.active_start = None
            self.data["active_seconds"] = float(self.data["active_seconds"]) + elapsed
            self.data["segments"].append({"label": label, "seconds": elapsed, "ended_unix": time.time()})
            self.data["h100_equivalent_hours"] = self.used_hours
            self.data["cap_hours"] = self.cap
            self.path.write_text(json.dumps(self.data, indent=2) + "\n")
            commit_remote_volume()
