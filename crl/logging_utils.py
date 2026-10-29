"""Structured run logging: JSONL records plus a config snapshot.

Every run directory is self-describing: ``config.yaml`` (exact settings),
``logs.jsonl`` (one record per logged step / phase event), and any JSON
artifacts saved via :meth:`RunLogger.save_json` (e.g. the evaluation
matrix). ``analysis/plots.py`` consumes exactly these files.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import yaml


class RunLogger:
    """Append-only JSONL logger bound to one run directory."""

    def __init__(self, results_dir: str | Path, run_name: str, config: dict[str, Any],
                 append: bool = False) -> None:
        self.run_dir = Path(results_dir) / run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        with open(self.run_dir / "config.yaml", "w") as handle:
            yaml.safe_dump(config, handle, sort_keys=False)
        # Truncate: re-running a config must overwrite its log, not append
        # (appending concatenates runs and corrupts step-indexed plots). A RESUMED
        # run is the exception -- it must append, or the record of every task it is
        # skipping is erased and the run becomes unanalysable.
        self._log_file = open(self.run_dir / "logs.jsonl", "a" if append else "w")
        self._start_time = time.time()

    def log(self, record: dict[str, Any]) -> None:
        """Write one flat record; adds wall-clock seconds since run start."""
        record = {"t_wall": round(time.time() - self._start_time, 3), **record}
        self._log_file.write(json.dumps(record) + "\n")
        self._log_file.flush()

    def save_json(self, name: str, obj: Any) -> None:
        with open(self.run_dir / name, "w") as handle:
            json.dump(obj, handle, indent=2)

    def close(self) -> None:
        self._log_file.close()
