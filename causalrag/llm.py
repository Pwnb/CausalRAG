"""Minimal LLM + environment helpers for CausalRAG.

The base model is OpenAI ``gpt-4o-mini`` (paper Appendix C), called at
``temperature=0`` for deterministic, reproducible answers. The API key is read
from the ``OPENAI_API_KEY`` environment variable (or a local ``.env`` file); it
is never hard-coded.

Both ``build_graph`` and ``query_causal_rag`` accept a pluggable ``llm=``
callable ``(prompt: str) -> str`` so the pipeline can be tested offline or
pointed at a different provider.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

# A chat function maps a single user prompt string to the model's text reply.
LLMCallable = Callable[[str], str]

DEFAULT_MODEL = "gpt-4o-mini"


def load_env() -> None:
    """Load KEY=VALUE pairs from a local ``.env`` into ``os.environ`` (no overwrite)."""
    for candidate in (Path(".env"), Path(__file__).resolve().parents[1] / ".env"):
        if not candidate.exists():
            continue
        for raw in candidate.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip("'").strip('"'))
        break


def require_api_key(api_key: str | None = None) -> str:
    if api_key:
        return api_key
    load_env()
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Add it to a .env file or pass api_key=."
        )
    return key


def make_openai_llm(
    model: str = DEFAULT_MODEL,
    api_key: str | None = None,
    temperature: float = 0.0,
    timeout: float = 60.0,
) -> LLMCallable:
    """Return a ``(prompt) -> str`` callable backed by the OpenAI chat API."""
    key = require_api_key(api_key)
    from openai import OpenAI

    client = OpenAI(api_key=key, timeout=timeout)

    def _call(prompt: str) -> str:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        return (resp.choices[0].message.content or "").strip()

    return _call
