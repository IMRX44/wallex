"""OpenAI-compatible chat client (Liara AI gateway -> anthropic/claude-opus-5.5)."""
from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

log = logging.getLogger(__name__)


class LLMClient:
    def __init__(self, base_url: str, api_key: str, model: str, timeout: float = 120, max_tokens: int = 2000):
        from openai import AsyncOpenAI
        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key, timeout=timeout, max_retries=2)
        self.model = model
        self.max_tokens = max_tokens
        self.calls = 0
        self.failures = 0
        self.last_latency = 0.0
        self.tools_supported = True

    async def chat(self, messages: list[dict], tools: list[dict] | None = None, temperature: float = 0.2) -> Any:
        t0 = time.time()
        kw: dict[str, Any] = {"model": self.model, "messages": messages, "max_tokens": self.max_tokens,
                              "temperature": temperature}
        if tools and self.tools_supported:
            kw["tools"] = tools
            kw["tool_choice"] = "auto"
        try:
            resp = await self.client.chat.completions.create(**kw)
        except Exception as e:  # noqa: BLE001
            self.failures += 1
            msg = str(e).lower()
            if tools and self.tools_supported and ("tool" in msg or "function" in msg) and "400" in msg:
                log.warning("LLM gateway rejected tool calling; switching to JSON mode")
                self.tools_supported = False
            raise
        finally:
            self.calls += 1
            self.last_latency = time.time() - t0
        return resp.choices[0].message

    async def complete_json(self, system: str, user: str) -> dict | None:
        msg = await self.chat([{"role": "system", "content": system}, {"role": "user", "content": user}])
        return extract_json(msg.content or "")


def extract_json(text: str) -> dict | None:
    text = text.strip()
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if m:
        text = m.group(1)
    start, depth = text.find("{"), 0
    if start < 0:
        return None
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    return None
    return None
