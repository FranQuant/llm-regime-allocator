import importlib.util

import pandas as pd
import pytest

from lra.config import REPO_ROOT, load_config
from lra.llm.blind import blind_features
from lra.llm.cache import ResponseCache

spec = importlib.util.spec_from_file_location("run_live", REPO_ROOT / "scripts" / "run_live.py")
run_live = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run_live)


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setattr(run_live, "OUT", tmp_path)
    monkeypatch.setitem(run_live.MODELS, "mock", None)
    cfg = load_config(REPO_ROOT / "configs" / "llm.toml")
    b = blind_features(pd.read_csv(REPO_ROOT / "results" / "baselines" / "features.csv", parse_dates=["date"],
                                   index_col="date"))
    return cfg, b, b.index[-3:], ResponseCache(tmp_path / "cache")


def test_per_model_log_is_append_only_and_flags_unknown_cutoff(setup, tmp_path):
    cfg, b, live, cache = setup
    run_live.log_model("mock", cfg, b, live, cache)
    log = pd.read_csv(tmp_path / "log_mock.csv")
    assert len(log) == 3 and log["after_model_cutoff"].isna().all()   # mock has no cutoff -> unknown
    run_live.log_model("mock", cfg, b, live, cache)                   # re-run: verified, nothing appended
    assert len(pd.read_csv(tmp_path / "log_mock.csv")) == 3


def test_changed_answer_refuses_to_overwrite(setup, tmp_path):
    cfg, b, live, cache = setup
    run_live.log_model("mock", cfg, b, live, cache)
    log = pd.read_csv(tmp_path / "log_mock.csv")
    log.loc[0, "Goldilocks"] += 0.1
    log.to_csv(tmp_path / "log_mock.csv", index=False)
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        run_live.log_model("mock", cfg, b, live, cache)


def test_sonnet_keeps_its_phase7_log_and_cap():
    assert run_live.log_path("claude_sonnet").name == "log.csv" and run_live.MODELS["claude_sonnet"] == 1500
    assert run_live.log_path("gpt_sol").name == "log_gpt_sol.csv"
