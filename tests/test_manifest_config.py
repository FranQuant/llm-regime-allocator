from lra.config import load_config
from lra.data.manifest import file_entry, update_manifest, verify_manifest


def test_manifest_detects_tampering(tmp_path):
    f = tmp_path / "x.csv"
    f.write_text("a,b\n1,2\n")
    update_manifest(tmp_path / "manifest.json", "x", file_entry(f, source="test", rows=1))
    assert verify_manifest(tmp_path / "manifest.json") == {"x": True}
    f.write_text("a,b\n1,3\n")
    assert verify_manifest(tmp_path / "manifest.json") == {"x": False}


def test_config_is_consistent():
    cfg = load_config()
    tickers = set(cfg["panel"]["tickers"])
    u = cfg["universe"]
    assert set(u["investable"]) <= tickers
    assert set(u["context"]) <= tickers
    assert set(u["benchmark_60_40"]) <= tickers
    assert u["cash"] in u["investable"]
    assert abs(sum(u["benchmark_60_40"].values()) - 1.0) < 1e-12
    for sid, spec in cfg["macro"]["series"].items():
        assert spec["mode"] in {"vintage", "lagged"}, sid
        if spec["mode"] == "lagged":
            assert "lag_days" in spec, sid
