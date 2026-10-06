"""Data manifest: what each committed data file is, where it came from, and its hash."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def file_entry(path: Path, *, source: str, rows: int, extra: dict | None = None) -> dict:
    return {
        "file": path.name,
        "sha256": sha256(path),
        "rows": rows,
        "source": source,
        "built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **(extra or {}),
    }


def update_manifest(manifest_path: Path, key: str, entry: dict) -> None:
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    manifest[key] = entry
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def verify_manifest(manifest_path: Path) -> dict[str, bool]:
    """Recompute hashes of every listed file; True where it matches."""
    manifest = json.loads(manifest_path.read_text())
    base = manifest_path.parent
    return {k: (base / v["file"]).exists() and sha256(base / v["file"]) == v["sha256"]
            for k, v in manifest.items()}
