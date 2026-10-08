"""scripts/archive_truncated.py must never overwrite an archived record."""

import importlib.util
import json
import sys

from lra.config import REPO_ROOT

spec = importlib.util.spec_from_file_location("archive_truncated", REPO_ROOT / "scripts" / "archive_truncated.py")
arch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(arch)


def _record(variant="v", **usage):
    return {"variant": variant, "parse_ok": False, "usage": usage, "key": "k"}


def _put(root, model, name, rec):
    p = root / model / name[:2] / f"{name}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(rec))
    return p


def test_free_path_adds_numbered_suffix(tmp_path):
    f = tmp_path / "abc.json"
    assert arch.free_path(f) == f
    f.write_text("x")
    assert arch.free_path(f).name == "abc.1.json"
    (tmp_path / "abc.1.json").write_text("y")
    assert arch.free_path(f).name == "abc.2.json"


def test_rerun_keeps_both_archived_records(tmp_path, monkeypatch):
    cache, archive = tmp_path / "cache", tmp_path / "archive"
    monkeypatch.setattr(arch, "CACHE", cache)
    monkeypatch.setattr(arch, "ARCHIVE", archive)
    monkeypatch.setattr(sys, "argv", ["archive_truncated.py", "--model-id", "m", "--variant", "v"])
    name = "ab" + "0" * 62
    _put(cache, "m", name, _record(finish_reason="length", marker="first"))
    arch.main()
    # a later re-run reuses the same cache key and is truncated again
    _put(cache, "m", name, _record(finish_reason="length", marker="second"))
    arch.main()
    files = sorted((archive / "m" / "ab").glob("*.json"))
    assert [f.name for f in files] == [f"{name}.1.json", f"{name}.json"]
    assert json.loads((archive / "m" / "ab" / f"{name}.json").read_text())["usage"]["marker"] == "first"
    assert json.loads((archive / "m" / "ab" / f"{name}.1.json").read_text())["usage"]["marker"] == "second"
    assert not list((cache / "m").rglob("*.json"))
