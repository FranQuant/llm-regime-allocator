"""Free pre-flight check for Phase 8: is each configured model ID reachable with the key in .env?

Uses model-metadata endpoints only (no generation, no cost). Never prints key values.
  python scripts/check_models.py                 # gpt_sol, gemini_flash, glm
  python scripts/check_models.py --models glm
"""

from __future__ import annotations

import argparse
import os

from lra.config import REPO_ROOT, load_config


def check(name: str, m: dict) -> str:
    key = os.environ.get(m["api_key_env"])
    if not key:
        return f"{m['api_key_env']} not set in .env"
    if m["provider"] == "openai":
        import openai

        return "ok: " + openai.OpenAI(api_key=key).models.retrieve(m["model"]).id
    if m["provider"] == "google":
        from google import genai

        info = genai.Client(api_key=key).models.get(model=m["model"])
        return f"ok: {info.name} (output limit {info.output_token_limit})"
    if m["provider"] == "openai_compatible":
        import openai

        ids = sorted(x.id for x in openai.OpenAI(api_key=key, base_url=m["base_url"]).models.list())
        if m["model"] in ids:
            return "ok: " + m["model"]
        near = [i for i in ids if m["model"].split("-")[0] in i]
        return f"NOT LISTED; similar names: {near or ids[:15]}"
    return "skipped"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["gpt_sol", "gemini_flash", "glm"])
    args = ap.parse_args()
    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env")
    cfg = load_config(REPO_ROOT / "configs" / "llm.toml")
    for name in args.models:
        try:
            msg = check(name, cfg["models"][name])
        except Exception as e:  # report and carry on; the message never contains the key
            msg = f"ERROR {type(e).__name__}: {str(e)[:200]}"
        print(f"{name:14s} {cfg['models'][name]['model']:20s} {msg}")


if __name__ == "__main__":
    main()
