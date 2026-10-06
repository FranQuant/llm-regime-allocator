"""Committed response cache. One JSON file per call, content-addressed.

key = sha256(provider, model, params, system, user, run, attempt)
`run` separates deliberate repeat runs; `attempt` separates retries after a failure.
Files live under results/llm_cache/<model>/<key[:2]>/<key>.json and are committed,
so every number in the study can be replayed without an API key.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def cache_key(provider: str, model: str, params: dict, system: str, user: str, run: int, attempt: int) -> str:
    raw = json.dumps({"provider": provider, "model": model, "params": params, "system": system,
                      "user": user, "run": run, "attempt": attempt}, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()


def _slug(model: str) -> str:
    return model.replace("/", "__").replace(":", "_")


class ResponseCache:
    def __init__(self, root: Path | str):
        self.root = Path(root)

    def path(self, model: str, key: str) -> Path:
        return self.root / _slug(model) / key[:2] / f"{key}.json"

    def get(self, model: str, key: str) -> dict | None:
        p = self.path(model, key)
        return json.loads(p.read_text()) if p.exists() else None

    def put(self, model: str, key: str, record: dict) -> None:
        p = self.path(model, key)
        p.parent.mkdir(parents=True, exist_ok=True)
        record = {**record, "key": key, "cached_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        p.write_text(json.dumps(record, indent=1, sort_keys=True) + "\n")

    def manifest(self) -> pd.DataFrame:
        """One row per cached call; written to results/llm_cache/manifest.csv."""
        rows = []
        for p in sorted(self.root.rglob("*.json")):
            r = json.loads(p.read_text())
            rows.append({k: r.get(k) for k in ("key", "provider", "model", "model_returned", "response_id",
                                               "variant", "decision_date", "run", "attempt",
                                               "prompt_version", "parse_ok", "cached_utc")}
                        | {"file": str(p.relative_to(self.root)),
                           "sha256": hashlib.sha256(p.read_bytes()).hexdigest()})
        return pd.DataFrame(rows)
