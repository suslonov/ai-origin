"""The only module that talks to OpenAI.

We use the documented HTTP API directly so the request saved with an experiment
is exactly the request sent to the model. LangSmith instruments the paid call;
LangGraph (graphs.py) supplies its parent trace. No browser ever receives a key.
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
from typing import Any

import requests
from langsmith import traceable

from schemas import ModelInput, output_schema


class ProviderError(RuntimeError):
    """Keep the status so a configuration/auth failure can stop a whole run."""

    def __init__(self, message: str, status: int = 0) -> None:
        super().__init__(message)
        self.status = status


def model_efforts(model: str) -> list[str]:
    """Conservative defaults; /models lists IDs, not capability metadata.

    'auto' omits reasoning from the request. We never silently substitute an
    effort or a model. Unknown/new families expose all documented choices and
    the API remains the final validator of account-specific model capabilities.
    """
    base = re.sub(r"-\d{4}-\d{2}-\d{2}$", "", model)
    if base.startswith(("gpt-4", "chatgpt-4")):
        return ["auto"]
    if base == "gpt-5-pro":
        return ["auto", "high"]
    if base.startswith(("o1-pro", "o3-pro")):
        return ["auto"]
    if base.startswith(("o1", "o3", "o4")):
        return ["auto", "low", "medium", "high"]
    if base in {"gpt-5", "gpt-5-mini", "gpt-5-nano"}:
        return ["auto", "minimal", "low", "medium", "high"]
    if base.startswith("gpt-6"):
        return ["auto", "low", "medium", "high", "xhigh", "max"]
    if base.startswith("gpt-5.1"):
        return ["auto", "none", "low", "medium", "high"]
    if base.startswith("gpt-5."):
        return ["auto", "none", "low", "medium", "high", "xhigh", "max"]
    return ["auto", "none", "minimal", "low", "medium", "high", "xhigh", "max"]


def text_model(model: str) -> bool:
    """Exclude modalities/tools that cannot perform this text JSON task."""
    return bool(re.match(r"^(gpt-(?:4o|4\.1|5|6)|o[134](?:-|$)|ft:)", model)) and not any(
        word in model for word in ("audio", "realtime", "transcribe", "tts", "image", "search", "deep-research", "codex", "chat-latest")
    )


class OpenAIProvider:
    def __init__(self) -> None:
        # The user's existing dev-server environment is the authority. In
        # particular, do not search key files or repurpose AWS credentials.
        self._key = os.environ.get("CHAT_GPT_KEY") or os.environ.get("OPENAI_API_KEY", "")
        self._models: list[dict] = []
        self._models_at = 0.0
        self._lock = threading.Lock()

    @property
    def configured(self) -> bool:
        return bool(self._key)

    def _request(self, method: str, path: str, payload: dict | None = None) -> dict:
        if not self._key:
            raise ProviderError("Set CHAT_GPT_KEY in the dev server environment.", 503)
        try:
            response = requests.request(
                method, "https://api.openai.com/v1/" + path,
                headers={"Authorization": "Bearer " + self._key, "Content-Type": "application/json"},
                json=payload, timeout=(15, 600),
            )
        except requests.RequestException as exc:
            # Never repeat a paid request automatically: a timeout may happen
            # after the provider has already completed and charged for it.
            raise ProviderError(f"OpenAI connection failed ({type(exc).__name__}); no automatic retry.") from None
        if not response.ok:
            try:
                message = response.json()["error"]["message"]
            except (ValueError, KeyError, TypeError):
                message = response.reason
            message = str(message).replace(self._key, "[redacted]")
            raise ProviderError(f"OpenAI HTTP {response.status_code}: {message[:1500]}", response.status_code)
        return response.json()

    def models(self, refresh: bool = False) -> list[dict]:
        with self._lock:
            if refresh or not self._models or time.monotonic() - self._models_at > 300:
                data = self._request("GET", "models")
                self._models = [{"id": m["id"], "efforts": model_efforts(m["id"])}
                                for m in data["data"] if text_model(m["id"])]
                self._models.sort(key=lambda m: m["id"], reverse=True)
                self._models_at = time.monotonic()
            return self._models

    def validate_model(self, config: ModelInput) -> None:
        available = {model["id"]: model for model in self.models()}
        if config.model not in available:
            raise ValueError("Choose a text model returned by this account's OpenAI model list.")
        if config.effort not in available[config.model]["efforts"]:
            raise ValueError("This reasoning effort is not offered for the selected model.")

    @traceable(name="OpenAI structured response", run_type="llm",
               process_inputs=lambda values: {k: v for k, v in values.items() if k != "self"})
    def complete(self, request: dict) -> dict:
        """Return the full response, including actual model ID and token usage."""
        return self._request("POST", "responses", request)


def make_request(config: dict, instructions: str, content: str, kind: str) -> dict:
    """Store this object with the experiment before making the paid API call."""
    request: dict[str, Any] = {
        "model": config["model"], "instructions": instructions,
        "input": [{"role": "user", "content": content}],
        "store": False, "max_output_tokens": config["max_output_tokens"],
        # Disable truncation: dropping part of an exam or benchmark silently
        # would change the experiment. Oversized inputs become explicit errors.
        "truncation": "disabled",
        "text": {"format": {"type": "json_schema", "name": "ai_origin_" + kind,
                            "strict": True, "schema": output_schema(kind)}},
    }
    if config["effort"] != "auto":
        request["reasoning"] = {"effort": config["effort"]}
    return request


def parse_response(response: dict) -> dict:
    """Refusal and token-limit truncation are failures, never a 0% estimate."""
    if response.get("status") != "completed":
        raise ValueError(f"Response not completed: {response.get('status')}; {response.get('incomplete_details')}")
    chunks = []
    for output in response.get("output", []):
        for block in output.get("content", []):
            if block.get("type") == "refusal":
                raise ValueError("Model refused: " + block.get("refusal", ""))
            if block.get("type") == "output_text":
                chunks.append(block["text"])
    return json.loads("".join(chunks))
