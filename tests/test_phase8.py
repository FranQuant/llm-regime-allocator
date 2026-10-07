import sys
import types

import pandas as pd

from lra.config import REPO_ROOT, load_config
from lra.llm.blind import blind_features
from lra.llm.cache import ResponseCache
from lra.llm.clients import GoogleClient, MockClient, OpenAICompatibleClient, OpenAIClient, make_client
from lra.llm.runner import classify

LLM = load_config(REPO_ROOT / "configs" / "llm.toml")
P8 = load_config(REPO_ROOT / "configs" / "phase8.toml")


def test_phase8_models_configured():
    for name, cls in (("gpt_sol", OpenAIClient), ("gemini_flash", GoogleClient), ("glm", OpenAICompatibleClient)):
        c = make_client(name, LLM)
        assert isinstance(c, cls) and c.max_tokens == 8000 and c.params == {}
    assert make_client("claude_sonnet", LLM).max_tokens == 4000   # existing runs keep their cap
    assert make_client("glm", LLM).base_url == "https://ollama.com/v1"


def test_preregistration_matches_config():
    for name in P8["models"]["new"]:
        m = LLM["models"][name]
        assert m["params"] == {} and m.get("max_tokens") == P8["calls"]["max_tokens"]
    assert P8["calls"]["variants"] == ["blinded", "date_probe_blinded"]
    assert P8["calls"]["repeat_every"] == 4


def test_google_client_maps_response(monkeypatch):
    seen = {}

    class Models:
        def generate_content(self, model, contents, config):
            seen.update(model=model, contents=contents, config=config)
            um = types.SimpleNamespace(model_dump=lambda exclude_none: {"thoughts_token_count": 7})
            cand = types.SimpleNamespace(finish_reason="STOP")
            return types.SimpleNamespace(text='{"x":1}', model_version="gemini-x", response_id="r1",
                                         usage_metadata=um, candidates=[cand])

    genai = types.SimpleNamespace(Client=lambda api_key: types.SimpleNamespace(models=Models()))
    monkeypatch.setitem(sys.modules, "google", types.SimpleNamespace(genai=genai))
    monkeypatch.setitem(sys.modules, "google.genai", genai)
    monkeypatch.setenv("GEMINI_API_KEY", "test")
    r = GoogleClient("gemini-x", max_tokens=123).complete("SYS", "USER")
    assert seen["config"] == {"system_instruction": "SYS", "max_output_tokens": 123}
    assert (r.text, r.model_returned, r.response_id) == ('{"x":1}', "gemini-x", "r1")
    assert r.usage == {"thoughts_token_count": 7, "finish_reason": "STOP"}


def test_mock_blinded_run_end_to_end(tmp_path):
    feats = pd.read_csv(REPO_ROOT / "results" / "baselines" / "features.csv", parse_dates=["date"], index_col="date")
    b = blind_features(feats)
    dates = b.index[b.index >= "2010-01-31"][:4]
    out, _ = classify(b, dates, client=MockClient(), cache=ResponseCache(tmp_path), variant="blinded")
    assert len(out) == 4
