"""Ollama HTTP client, and the llama-server client beside it.

Ollama's prompt-eval and eval durations are llama-server's prompt_ms and
predicted_ms. The second client reads those fields and reports them in the
same shape, so the two clocks can be compared.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

HOST = "http://127.0.0.1:11434"
CONTEXT_CAP = 8192
MODEL_NAME = "qwen3.8-27b-text"


@dataclass
class GenerateResult:
    response: str
    prompt_eval_count: int
    prompt_eval_duration_ns: int
    prompt_eval_cached_count: int
    eval_count: int
    eval_duration_ns: int
    load_duration_ns: int
    done_reason: str
    t_end_s: float
    wall_s: float


class Ollama:
    def __init__(self, model: str, context_cap: int = CONTEXT_CAP):
        self.model = model
        self.context_cap = context_cap

    def generate(self, prompt: str, reply_tokens: int, t0: float | None = None) -> GenerateResult:
        import time

        if t0 is None:
            t0 = time.perf_counter()
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": True,
            "raw": True,
            "think": False,
            "keep_alive": "30m",
            "options": {
                "num_predict": reply_tokens,
                "num_ctx": self.context_cap,
                "temperature": 0,
                "seed": 1,
            },
        }
        request = urllib.request.Request(
            f"{HOST}/api/generate",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        started = time.perf_counter()
        try:
            response = urllib.request.urlopen(request, timeout=900)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Ollama generate failed: {exc.code} {detail}") from exc
        final = None
        text_parts: list[str] = []
        with response:
            for line in response:
                if not line.strip():
                    continue
                message = json.loads(line)
                if message.get("error"):
                    raise RuntimeError(message["error"])
                piece = message.get("response") or ""
                if piece:
                    text_parts.append(piece)
                if message.get("done"):
                    final = message
        t_end = time.perf_counter()
        if final is None:
            raise RuntimeError("Ollama generate returned no final message")
        return GenerateResult(
            response="".join(text_parts),
            prompt_eval_count=int(final["prompt_eval_count"]),
            prompt_eval_duration_ns=int(final["prompt_eval_duration"]),
            prompt_eval_cached_count=int(final.get("prompt_eval_cached_count") or 0),
            eval_count=int(final["eval_count"]),
            eval_duration_ns=int(final["eval_duration"]),
            load_duration_ns=int(final.get("load_duration") or 0),
            done_reason=str(final.get("done_reason") or ""),
            t_end_s=t_end - t0,
            wall_s=t_end - started,
        )


def completion_to_result(message: dict, t_end_s: float, wall_s: float) -> GenerateResult:
    """Map one llama-server /completion stop message onto Ollama's generate fields.

    Ollama sets prompt_eval_count to cache_n + prompt_n, prompt_eval_duration
    from prompt_ms, and eval_duration from predicted_ms. stop_type "limit" is
    Ollama's done_reason "length".
    """
    timings = message.get("timings")
    if not isinstance(timings, dict):
        raise RuntimeError("llama-server completion had no timings")
    cache_n = int(timings.get("cache_n") or 0)
    prompt_ms = float(timings["prompt_ms"])
    predicted_ms = float(timings["predicted_ms"])
    stop_type = str(message.get("stop_type") or "")
    if stop_type == "limit" or message.get("stopped_limit") is True:
        done_reason = "length"
    else:
        done_reason = "stop"
    return GenerateResult(
        response=str(message.get("content") or ""),
        prompt_eval_count=cache_n + int(timings["prompt_n"]),
        prompt_eval_duration_ns=int(prompt_ms * 1_000_000),
        prompt_eval_cached_count=cache_n,
        eval_count=int(timings["predicted_n"]),
        eval_duration_ns=int(predicted_ms * 1_000_000),
        load_duration_ns=0,
        done_reason=done_reason,
        t_end_s=t_end_s,
        wall_s=wall_s,
    )


class LlamaServer:
    """llama-server /completion, reporting timings the way Ollama does."""

    def __init__(self, base: str, context_cap: int = CONTEXT_CAP):
        self.base = base.rstrip("/")
        self.context_cap = context_cap
        self.model = "llama.cpp"

    def generate(self, prompt: str, reply_tokens: int, t0: float | None = None) -> GenerateResult:
        import time

        if t0 is None:
            t0 = time.perf_counter()
        payload = {
            "prompt": prompt,
            "stream": True,
            "cache_prompt": True,
            "n_predict": reply_tokens,
            "temperature": 0,
            "seed": 1,
        }
        request = urllib.request.Request(
            f"{self.base}/completion",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        started = time.perf_counter()
        try:
            response = urllib.request.urlopen(request, timeout=900)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"llama-server completion failed: {exc.code} {detail}") from exc
        parts: list[str] = []
        final = None
        with response:
            for line in response:
                if not line.strip():
                    continue
                text = line.decode("utf-8", errors="replace").strip()
                if text.startswith(":"):
                    continue
                if text.startswith("data:"):
                    text = text[len("data:") :].strip()
                if not text or text == "[DONE]":
                    continue
                message = json.loads(text)
                if message.get("error"):
                    raise RuntimeError(str(message["error"]))
                piece = message.get("content") or ""
                if message.get("stop"):
                    final = message
                elif piece:
                    parts.append(piece)
        t_end = time.perf_counter()
        if final is None:
            raise RuntimeError("llama-server completion returned no final message")
        result = completion_to_result(final, t_end - t0, t_end - started)
        result.response = "".join(parts) or result.response
        return result


class RunnerTokenizer:
    def __init__(self, port: int):
        self.base = f"http://127.0.0.1:{port}"

    def tokenize(self, text: str) -> list[int]:
        body = self._post("/tokenize", {"content": text})
        return [int(token) for token in body["tokens"]]

    def detokenize(self, tokens: list[int]) -> str:
        body = self._post("/detokenize", {"tokens": tokens})
        return str(body["content"])

    def _post(self, path: str, payload: dict) -> dict:
        request = urllib.request.Request(
            self.base + path,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"llama-server {path} failed: {exc.code} {detail}") from exc


def model_manifest(model: str) -> dict:
    path = Path.home() / ".ollama/models/manifests/registry.ollama.ai/library" / model / "latest"
    if not path.is_file():
        raise FileNotFoundError(f"no local Ollama model {model!r} at {path}")
    return json.loads(path.read_text())


def weight_digest(manifest: dict) -> str:
    for layer in manifest["layers"]:
        if layer["mediaType"] == "application/vnd.ollama.image.model":
            return layer["digest"]
    raise RuntimeError("manifest has no language-model layer")


def projector_layers(manifest: dict) -> list[str]:
    return [
        layer["mediaType"]
        for layer in manifest["layers"]
        if "projector" in layer["mediaType"]
    ]


def build_prompts(tokenizer: RunnerTokenizer, count: int, content_tokens: int, salt: str) -> list[str]:
    """Build `count` prompts that each tokenize back to `content_tokens` ids.

    The windows do not overlap, and `salt` keeps this suite off any prompt already
    in the runner's cache. Ollama reuses a cached prefix and would otherwise skip prefill.
    """
    needed = count * content_tokens
    words: list[str] = []
    index = 0
    ids: list[int] = []
    while len(ids) < needed:
        words.extend(f"{salt}{index + offset}" for offset in range(1024))
        index += 1024
        ids = tokenizer.tokenize(" ".join(words))
    prompts = []
    for slot in range(count):
        start = slot * content_tokens
        window = ids[start : start + content_tokens]
        text = tokenizer.detokenize(window)
        back = tokenizer.tokenize(text)
        if back != window:
            raise RuntimeError(
                f"prompt {slot} did not round-trip through the tokenizer "
                f"({len(back)} tokens, wanted {content_tokens})"
            )
        prompts.append(text)
    return prompts
