"""Start the unmodified llama-server Ollama ships, on the shared GGUF."""

from __future__ import annotations

import json
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

# The binary Ollama 0.33.3 runs. Commit 0f3a71be1, as printed by --version.
DEFAULT_LLAMA_SERVER = Path("/Applications/Ollama.app/Contents/Resources/llama-server")
LLAMA_PORT = 8081


def llama_server_command(binary: Path, gguf: Path, port: int, context_cap: int) -> list[str]:
    """The launch line Ollama used for this model, pointed at the shared GGUF.

    Context, batch, and flash-attn are the flags that change the kernels.
    --context-shift and --keep 4 are what that runner was started with.
    Ollama's own log, 2026-10-11: -c 8192 -np 1 --flash-attn auto -b 1024 -ub 1024
    --context-shift --keep 4.
    """
    return [
        str(binary),
        "--model",
        str(gguf),
        "--port",
        str(port),
        "--host",
        "127.0.0.1",
        "--no-webui",
        "--offline",
        "-c",
        str(context_cap),
        "-np",
        "1",
        "--flash-attn",
        "auto",
        "-b",
        "1024",
        "-ub",
        "1024",
        "--context-shift",
        "--keep",
        "4",
    ]


@dataclass
class LlamaProcess:
    proc: subprocess.Popen
    port: int
    log_path: Path
    _log: object

    def stop(self) -> None:
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=10)
        self._log.close()


def start_llama_server(
    gguf: Path,
    log_path: Path,
    context_cap: int,
    port: int = LLAMA_PORT,
    binary: Path = DEFAULT_LLAMA_SERVER,
) -> LlamaProcess:
    if not binary.is_file():
        raise FileNotFoundError(f"llama-server not found at {binary}")
    if not gguf.is_file():
        raise FileNotFoundError(f"GGUF not found at {gguf}")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = log_path.open("w")
    command = llama_server_command(binary, gguf, port, context_cap)
    proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
    return LlamaProcess(proc, port, log_path, log)


def wait_until_ready(process: LlamaProcess, timeout_s: float = 300) -> None:
    """Block until /health reports the model is loaded, or the process exits."""
    deadline = time.monotonic() + timeout_s
    url = f"http://127.0.0.1:{process.port}/health"
    last = "no response yet"
    while time.monotonic() < deadline:
        if process.proc.poll() is not None:
            tail = _log_tail(process.log_path)
            raise RuntimeError(
                f"llama-server exited {process.proc.returncode} before it was ready\n{tail}"
            )
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                body = json.load(response)
            if body.get("status") == "ok":
                return
            last = str(body.get("status") or body)
        except urllib.error.HTTPError as exc:
            last = f"HTTP {exc.code}"
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
            last = str(exc)
        time.sleep(0.5)
    tail = _log_tail(process.log_path)
    raise TimeoutError(f"llama-server was not ready within {timeout_s:.0f}s ({last})\n{tail}")


def server_version(binary: Path = DEFAULT_LLAMA_SERVER) -> str:
    out = subprocess.check_output([str(binary), "--version"], text=True, stderr=subprocess.STDOUT)
    return out.strip().splitlines()[0]


def _log_tail(path: Path, lines: int = 40) -> str:
    if not path.is_file():
        return ""
    text = path.read_text(errors="replace").splitlines()
    return "\n".join(text[-lines:])
