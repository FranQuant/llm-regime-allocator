"""Provider clients behind one protocol. Each call returns text plus metadata to log.

Provider SDKs are imported lazily so replay from cache needs neither SDKs nor keys.
`params` from configs/llm.toml are passed through verbatim.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Protocol

from lra.llm.schema import REGIMES


@dataclass
class LLMResponse:
    text: str
    model_returned: str | None = None
    response_id: str | None = None
    usage: dict = field(default_factory=dict)


class LLMClient(Protocol):
    provider: str
    model: str
    params: dict

    def complete(self, system: str, user: str) -> LLMResponse: ...


def _key(env: str) -> str:
    k = os.environ.get(env)
    if not k:
        raise EnvironmentError(f"{env} not set")
    return k


@dataclass
class AnthropicClient:
    model: str
    api_key_env: str = "ANTHROPIC_API_KEY"
    max_tokens: int = 4000
    params: dict = field(default_factory=dict)
    provider: str = "anthropic"

    def complete(self, system: str, user: str) -> LLMResponse:
        import anthropic

        client = anthropic.Anthropic(api_key=_key(self.api_key_env))
        r = client.messages.create(model=self.model, max_tokens=self.max_tokens, system=system,
                                   messages=[{"role": "user", "content": user}], **self.params)
        text = "".join(b.text for b in r.content if getattr(b, "type", None) == "text")
        usage = r.usage.model_dump() if hasattr(r.usage, "model_dump") else dict(r.usage or {})
        return LLMResponse(text, r.model, r.id, usage)


@dataclass
class OpenAIClient:
    model: str
    api_key_env: str = "OPENAI_API_KEY"
    max_tokens: int = 4000
    params: dict = field(default_factory=dict)
    provider: str = "openai"

    def complete(self, system: str, user: str) -> LLMResponse:
        import openai

        client = openai.OpenAI(api_key=_key(self.api_key_env))
        r = client.responses.create(model=self.model, instructions=system, input=user,
                                    max_output_tokens=self.max_tokens, **self.params)
        usage = r.usage.model_dump() if getattr(r, "usage", None) is not None else {}
        usage["status"] = getattr(r, "status", None)   # "incomplete" = cap hit (reasoning counts toward it)
        return LLMResponse(r.output_text, r.model, r.id, usage)


@dataclass
class OpenAICompatibleClient:
    """Chat-completions endpoint (hosted open-weight providers, local Ollama, ...)."""

    model: str
    base_url: str
    api_key_env: str = "OPENROUTER_API_KEY"
    max_tokens: int = 4000
    params: dict = field(default_factory=dict)
    provider: str = "openai_compatible"

    def complete(self, system: str, user: str) -> LLMResponse:
        import openai

        client = openai.OpenAI(api_key=os.environ.get(self.api_key_env, "unused"), base_url=self.base_url)
        r = client.chat.completions.create(model=self.model, max_tokens=self.max_tokens,
                                           messages=[{"role": "system", "content": system},
                                                     {"role": "user", "content": user}], **self.params)
        usage = r.usage.model_dump() if getattr(r, "usage", None) is not None else {}
        usage["finish_reason"] = r.choices[0].finish_reason
        return LLMResponse(r.choices[0].message.content or "", r.model, r.id, usage)


@dataclass
class GoogleClient:
    """Gemini API via the google-genai SDK. `params` are extra GenerateContentConfig fields."""

    model: str
    api_key_env: str = "GEMINI_API_KEY"
    max_tokens: int = 4000
    params: dict = field(default_factory=dict)
    provider: str = "google"

    def complete(self, system: str, user: str) -> LLMResponse:
        from google import genai

        client = genai.Client(api_key=_key(self.api_key_env))
        config = {"system_instruction": system, "max_output_tokens": self.max_tokens, **self.params}
        r = client.models.generate_content(model=self.model, contents=user, config=config)
        um = r.usage_metadata
        usage = um.model_dump(exclude_none=True) if um is not None else {}
        cands = r.candidates or []
        usage["finish_reason"] = str(cands[0].finish_reason) if cands and cands[0].finish_reason else None
        return LLMResponse(r.text or "", r.model_version, r.response_id, usage)


@dataclass
class MockClient:
    """Deterministic offline client: probabilities are a pure function of the prompt."""

    model: str = "mock-regime-v1"
    params: dict = field(default_factory=dict)
    provider: str = "mock"
    fail_every: int = 0          # >0: every n-th call returns unparseable text (tests)
    calls: int = 0

    def complete(self, system: str, user: str) -> LLMResponse:
        self.calls += 1
        if self.fail_every and self.calls % self.fail_every == 0:
            return LLMResponse("not json", self.model, f"mock-{self.calls}")
        h = hashlib.sha256((system + "\n" + user).encode()).digest()
        if '"year"' in system:  # date-recovery probe
            body = {"year": 2007 + h[0] % 20, "month": 1 + h[1] % 12, "confidence": round(h[2] / 255, 3),
                    "cues": ["mock cue"]}
            return LLMResponse(json.dumps(body), self.model, "mock-" + h.hex()[:12], {"input_tokens": len(user) // 4})
        raw = [b + 1 for b in h[:4]]
        p = [round(x / sum(raw), 4) for x in raw]
        p[-1] = round(1 - sum(p[:-1]), 4)
        body = {"probabilities": dict(zip(REGIMES, p)), "confidence": round(h[4] / 255, 3),
                "drivers": ["mock driver"], "rationale": "deterministic mock response"}
        return LLMResponse(json.dumps(body), self.model, "mock-" + h.hex()[:12], {"input_tokens": len(user) // 4})


def make_client(name: str, cfg: dict) -> LLMClient:
    m = cfg["models"][name]
    mt = m.get("max_tokens", cfg["defaults"]["max_tokens"])   # per-model cap (reasoning models need more)
    if m["provider"] == "mock":
        return MockClient(model=m["model"], params=m.get("params", {}))
    if m["provider"] == "anthropic":
        return AnthropicClient(m["model"], m["api_key_env"], mt, m.get("params", {}))
    if m["provider"] == "openai":
        return OpenAIClient(m["model"], m["api_key_env"], mt, m.get("params", {}))
    if m["provider"] == "openai_compatible":
        return OpenAICompatibleClient(m["model"], m["base_url"], m["api_key_env"], mt, m.get("params", {}))
    if m["provider"] == "google":
        return GoogleClient(m["model"], m["api_key_env"], mt, m.get("params", {}))
    raise ValueError(f"unknown provider {m['provider']!r}")
