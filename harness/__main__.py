"""Run the agreed Ollama protocol: exact token counts, thinking off, runner footprint."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from harness.client import (
    CONTEXT_CAP,
    MODEL_NAME,
    Ollama,
    RunnerTokenizer,
    build_prompts,
    model_manifest,
    projector_layers,
    weight_digest,
)
from harness.footprint import FootprintSampler, find_runner, process_memory
from harness.gguf import summarize
from harness.protocol import (
    decode_ceiling_tokens_per_second,
    mark_phases,
    mean,
    median,
    tokens_per_second,
)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_GGUF = ROOT / "models" / "Qwen3.8-27B-Text-Only-Q4_K_M.gguf"
SAMPLE_INTERVAL_S = 0.1


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure stock Ollama on the shared GGUF.")
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--gguf", type=Path, default=DEFAULT_GGUF)
    parser.add_argument("--context-cap", type=int, default=CONTEXT_CAP)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("check", help="Generate text and compare the runner footprint to the file")

    speed = sub.add_parser("speed", help="One discarded warmup, then ten 512/128 runs")
    speed.add_argument("--prompt-tokens", type=int, default=512)
    speed.add_argument("--reply-tokens", type=int, default=128)
    speed.add_argument("--runs", type=int, default=10)

    memory = sub.add_parser("memory", help="One discarded warmup, then a 4096/512 footprint trace")
    memory.add_argument("--prompt-tokens", type=int, default=4096)
    memory.add_argument("--reply-tokens", type=int, default=512)
    memory.add_argument("--runs", type=int, default=1)

    args = parser.parse_args()
    if args.command == "check":
        check(args.model, args.gguf, args.context_cap)
    elif args.command == "speed":
        suite(args, sample_footprint=False)
    else:
        suite(args, sample_footprint=True)


def check(model: str, gguf: Path, context_cap: int) -> None:
    info = summarize(gguf)
    manifest = model_manifest(model)
    projectors = projector_layers(manifest)
    client = Ollama(model, context_cap)
    result = client.generate("Say hello in one sentence.", 32)
    pid, _port = find_runner(weight_digest(manifest))
    footprint, resident = process_memory(pid)
    report = {
        "model": model,
        "architecture": info["architecture"],
        "file_type": info["file_type_name"],
        "tensor_count": info["tensor_count"],
        "vision_tensors": info["vision_tensors"],
        "projector_layers": projectors,
        "file_bytes": info["bytes"],
        "phys_footprint": footprint,
        "resident_size": resident,
        "prompt_eval_count": result.prompt_eval_count,
        "eval_count": result.eval_count,
        "response": result.response,
        "runner_pid": pid,
    }
    _write_json("check", report)
    gib = 1024**3
    print(f"file {info['bytes'] / gib:.2f} GiB  {info['file_type_name']}  arch {info['architecture']}")
    print(f"vision tensors in the file: {len(info['vision_tensors'])}")
    print(f"projector layers in the Ollama model: {len(projectors)}")
    print(f"runner pid {pid}  phys_footprint {footprint / gib:.2f} GiB  resident {resident / gib:.2f} GiB")
    print(f"reply: {result.response.strip()!r}")
    if info["vision_tensors"] or projectors:
        raise SystemExit("vision weights are loaded")
    if not result.response.strip():
        raise SystemExit("model returned an empty reply")


def suite(args, sample_footprint: bool) -> None:
    client = Ollama(args.model, args.context_cap)
    manifest = model_manifest(args.model)
    digest = weight_digest(manifest)
    delta = _calibrate(client, digest)
    content_tokens = args.prompt_tokens - delta
    if content_tokens < 1:
        raise SystemExit(f"prompt token target {args.prompt_tokens} is smaller than the BOS overhead {delta}")
    pid, port = find_runner(digest)
    tokenizer = RunnerTokenizer(port)
    # One discarded warmup, then the measured runs. Each string is new.
    salt = time.strftime("%H%M%S")
    prompts = build_prompts(tokenizer, args.runs + 1, content_tokens, salt)
    print(f"calibrated prompt overhead {delta} tokens; content tokens per prompt {content_tokens}")
    warmup = _one_run(client, prompts[0], args.reply_tokens, pid=None)
    print(
        f"warmup discarded  prefill {warmup['prefill_tps']:.2f} tok/s  "
        f"decode {warmup['decode_tps']:.2f} tok/s  "
        f"cached {warmup['prompt_eval_cached_count']}"
    )
    if warmup["prompt_eval_cached_count"] > 0:
        raise SystemExit(
            f"warmup prefill reused {warmup['prompt_eval_cached_count']} cached tokens"
        )
    measured = []
    traces = []
    for index, prompt in enumerate(prompts[1:]):
        sample_pid = pid if sample_footprint else None
        run = _one_run(client, prompt, args.reply_tokens, pid=sample_pid)
        _require_exact(run, args.prompt_tokens, args.reply_tokens)
        measured.append(run)
        if run["trace"]:
            traces.extend({"run": index, **sample} for sample in run["trace"])
        print(
            f"run {index + 1:02d}  prefill {run['prefill_tps']:.2f} tok/s  "
            f"decode {run['decode_tps']:.2f} tok/s  "
            f"cached {run['prompt_eval_cached_count']}  "
            f"load {run['load_duration_ns'] / 1e6:.1f} ms"
        )
    prefill = [run["prefill_tps"] for run in measured]
    decode = [run["decode_tps"] for run in measured]
    summary = {
        "model": args.model,
        "context_cap": args.context_cap,
        "prompt_tokens": args.prompt_tokens,
        "reply_tokens": args.reply_tokens,
        "warmup_discarded": 1,
        "runs": len(measured),
        "mean_prefill_tps": mean(prefill),
        "mean_decode_tps": mean(decode),
        "prefill_tps": prefill,
        "decode_tps": decode,
        "sample_hz": (1 / SAMPLE_INTERVAL_S) if sample_footprint else None,
    }
    if sample_footprint:
        decode_medians = []
        resident_medians = []
        peaks = []
        resident_peaks = []
        for run in measured:
            decode_samples = [sample for sample in run["trace"] if sample["phase"] == "decode"]
            if not decode_samples:
                raise SystemExit("a run produced no decode footprint samples")
            decode_medians.append(median([sample["phys_footprint"] for sample in decode_samples]))
            resident_medians.append(median([sample["resident_size"] for sample in decode_samples]))
            peaks.append(max(sample["phys_footprint"] for sample in run["trace"]))
            resident_peaks.append(max(sample["resident_size"] for sample in run["trace"]))
        summary["median_decode_phys_footprint"] = decode_medians
        summary["mean_of_median_decode_phys_footprint"] = mean(decode_medians)
        summary["peak_phys_footprint"] = peaks
        summary["median_decode_resident"] = resident_medians
        summary["mean_of_median_decode_resident"] = mean(resident_medians)
        summary["peak_resident"] = resident_peaks
    else:
        weight_bytes = args.gguf.stat().st_size
        ceiling = decode_ceiling_tokens_per_second(weight_bytes)
        summary["weight_bytes"] = weight_bytes
        summary["decode_ceiling_tps"] = ceiling
        summary["decode_fraction_of_ceiling"] = summary["mean_decode_tps"] / ceiling
        summary["prefill_over_ceiling"] = summary["mean_prefill_tps"] / ceiling
    out_dir = _write_json(args.command, summary, traces if sample_footprint else None)
    print(
        f"mean prefill {summary['mean_prefill_tps']:.2f} tok/s  "
        f"mean decode {summary['mean_decode_tps']:.2f} tok/s"
    )
    if "decode_ceiling_tps" in summary:
        print(
            f"decode ceiling {summary['decode_ceiling_tps']:.2f} tok/s "
            f"at 400 GB/s over {summary['weight_bytes'] / 1e9:.2f} GB of weights"
        )
        print(f"decode is {summary['decode_fraction_of_ceiling']:.2%} of that ceiling")
        _write_ceiling_note(summary)
    if sample_footprint:
        gib = 1024**3
        print(
            f"mean of median decode phys_footprint "
            f"{summary['mean_of_median_decode_phys_footprint'] / gib:.2f} GiB"
        )
        print(
            f"mean of median decode resident "
            f"{summary['mean_of_median_decode_resident'] / gib:.2f} GiB"
        )
        print(f"trace {out_dir / 'trace.jsonl'}")


def _calibrate(client: Ollama, digest: str) -> int:
    """How many prompt-eval tokens Ollama adds beyond the raw string (the BOS)."""
    result = client.generate("calibration probe", 1)
    _pid, port = find_runner(digest)
    content_tokens = len(RunnerTokenizer(port).tokenize("calibration probe"))
    delta = result.prompt_eval_count - content_tokens
    if delta < 0 or delta > 2:
        raise SystemExit(f"unexpected prompt overhead {delta} (eval {result.prompt_eval_count}, tokens {content_tokens})")
    return delta


def _one_run(client: Ollama, prompt: str, reply_tokens: int, pid: int | None) -> dict:
    sampler = None
    t0 = time.perf_counter()
    if pid is not None:
        sampler = FootprintSampler(pid, SAMPLE_INTERVAL_S)
        sampler.start(t0)
    try:
        result = client.generate(prompt, reply_tokens, t0=t0)
    finally:
        if sampler is not None:
            sampler.stop()
    prefill_tps = tokens_per_second(result.prompt_eval_count, result.prompt_eval_duration_ns)
    decode_tps = tokens_per_second(result.eval_count, result.eval_duration_ns)
    trace = []
    if sampler is not None:
        marked = mark_phases(
            sampler.samples,
            result.t_end_s,
            result.prompt_eval_duration_ns,
            result.eval_duration_ns,
        )
        trace = [
            {
                "t_s": round(sample.t_s, 4),
                "phys_footprint": sample.phys_footprint,
                "resident_size": sample.resident_size,
                "phase": sample.phase,
            }
            for sample in marked
        ]
    return {
        "prompt_eval_count": result.prompt_eval_count,
        "prompt_eval_duration_ns": result.prompt_eval_duration_ns,
        "prompt_eval_cached_count": result.prompt_eval_cached_count,
        "eval_count": result.eval_count,
        "eval_duration_ns": result.eval_duration_ns,
        "load_duration_ns": result.load_duration_ns,
        "done_reason": result.done_reason,
        "prefill_tps": prefill_tps,
        "decode_tps": decode_tps,
        "response": result.response,
        "trace": trace,
    }


def _require_exact(run: dict, prompt_tokens: int, reply_tokens: int) -> None:
    if run["prompt_eval_count"] != prompt_tokens:
        raise SystemExit(
            f"prompt eval count {run['prompt_eval_count']} != {prompt_tokens}"
        )
    if run["eval_count"] != reply_tokens or run["done_reason"] != "length":
        raise SystemExit(
            f"reply stopped at {run['eval_count']} tokens ({run['done_reason']}), wanted {reply_tokens}"
        )
    if run["prompt_eval_cached_count"] > 0:
        raise SystemExit(
            f"prefill reused {run['prompt_eval_cached_count']} cached tokens"
        )


def _write_json(kind: str, summary: dict, traces: list | None = None) -> Path:
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out_dir = ROOT / "results" / f"{kind}-{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if traces is not None:
        lines = [json.dumps(sample) for sample in traces]
        (out_dir / "trace.jsonl").write_text("\n".join(lines) + "\n")
    print(f"wrote {out_dir / 'summary.json'}")
    return out_dir


def _write_ceiling_note(summary: dict) -> None:
    weight_gb = summary["weight_bytes"] / 1e9
    text = (
        f"Weight file: {summary['weight_bytes']} bytes ({weight_gb:.2f} GB).\n"
        f"Decode ceiling at 400 GB/s: {summary['decode_ceiling_tps']:.2f} tok/s, "
        f"one full read of the weights per token.\n"
        f"Mean decode over {summary['runs']} warm runs: {summary['mean_decode_tps']:.2f} tok/s, "
        f"{summary['decode_fraction_of_ceiling']:.1%} of that ceiling.\n"
        f"Mean prefill: {summary['mean_prefill_tps']:.2f} tok/s, "
        f"{summary['prefill_over_ceiling']:.2f} times the decode ceiling. "
        f"Prefill reads the weights once for many tokens, so it is not held to one token per read.\n"
    )
    path = ROOT / "results" / "ceiling.txt"
    path.write_text(text)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
