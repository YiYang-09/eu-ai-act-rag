"""LLM clients: Ollama for real runs, a deterministic fake for tests and CI."""

from __future__ import annotations

import json

import requests


class OllamaLLM:
    def __init__(self, model: str = "qwen2.5:7b", url: str = "http://localhost:11434", num_ctx: int = 8192,
                 max_tokens: int = 512, timeout: int = 300):
        self.model, self.url, self.num_ctx, self.max_tokens, self.timeout = model, url, num_ctx, max_tokens, timeout

    def chat(self, messages: list[dict]) -> dict:
        body = {"model": self.model, "messages": messages, "stream": False, "format": "json",
                "options": {"temperature": 0, "seed": 0, "num_ctx": self.num_ctx, "num_predict": self.max_tokens}}
        r = requests.post(f"{self.url}/api/chat", json=body, timeout=self.timeout)
        r.raise_for_status()
        d = r.json()
        return {"content": d["message"]["content"], "prompt_tokens": d.get("prompt_eval_count"),
                "output_tokens": d.get("eval_count")}

    def healthy(self) -> bool:
        try:
            return requests.get(f"{self.url}/api/tags", timeout=3).ok
        except requests.RequestException:
            return False


class FakeLLM:
    """Answers with the first sentence of source 1 and cites it; refuses when no sources."""

    model = "fake"

    def chat(self, messages: list[dict]) -> dict:
        user = messages[-1]["content"]
        if "[1]" not in user:
            out = {"answer": "The provided texts do not answer this.", "sources": [], "insufficient": True}
        else:
            block = user.split("[1]", 1)[1].split("\n\n[2]", 1)[0].split("\nQuestion:", 1)[0]
            body = block.split("\n", 2)[-1].strip()  # drop the label and header lines
            out = {"answer": body[:240].rsplit(" ", 1)[0] + " …", "sources": [1], "insufficient": False}
        return {"content": json.dumps(out), "prompt_tokens": len(user.split()), "output_tokens": 20}

    def healthy(self) -> bool:
        return True
