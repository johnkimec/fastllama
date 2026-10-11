import struct
import tempfile
import unittest
from pathlib import Path

from harness.client import completion_to_result
from harness.footprint import Sample
from harness.gguf import summarize, vision_tensors
from harness.protocol import (
    decode_ceiling_tokens_per_second,
    inside_band,
    mark_phases,
    mean,
    median,
    parity_band,
    tokens_per_second,
)
from harness.server import llama_server_command


class ProtocolTest(unittest.TestCase):
    def test_rates_and_mean(self):
        self.assertAlmostEqual(tokens_per_second(128, 4_000_000_000), 32.0)
        self.assertAlmostEqual(mean([10.0, 20.0, 30.0]), 20.0)

    def test_median_odd_and_even(self):
        self.assertEqual(median([3, 1, 2]), 2.0)
        self.assertEqual(median([1, 100]), 50.5)

    def test_mark_phases_uses_ollama_durations(self):
        samples = [
            Sample(0.05, 10),
            Sample(0.40, 11),
            Sample(0.80, 12),
            Sample(1.40, 13),
        ]
        # Prefill is 0.40s starting at t=0.30. Decode is 0.80s starting at t=0.70, ending at t=1.50.
        marked = mark_phases(samples, t_end_s=1.50, prompt_eval_ns=400_000_000, eval_ns=800_000_000)
        self.assertEqual([sample.phase for sample in marked], ["other", "prefill", "decode", "decode"])

    def test_decode_ceiling(self):
        self.assertAlmostEqual(decode_ceiling_tokens_per_second(15_000_000_000), 400e9 / 15_000_000_000)

    def test_llama_server_clocks_match_ollama_fields(self):
        message = {
            "content": "",
            "stop": True,
            "stop_type": "limit",
            "timings": {
                "cache_n": 0,
                "prompt_n": 512,
                "prompt_ms": 4000.0,
                "predicted_n": 128,
                "predicted_ms": 10000.0,
            },
        }
        result = completion_to_result(message, t_end_s=14.0, wall_s=14.0)
        self.assertEqual(result.prompt_eval_count, 512)
        self.assertEqual(result.prompt_eval_duration_ns, 4_000_000_000)
        self.assertEqual(result.prompt_eval_cached_count, 0)
        self.assertEqual(result.eval_count, 128)
        self.assertEqual(result.eval_duration_ns, 10_000_000_000)
        self.assertEqual(result.done_reason, "length")
        self.assertAlmostEqual(
            tokens_per_second(result.prompt_eval_count, result.prompt_eval_duration_ns),
            128.0,
        )
        self.assertAlmostEqual(
            tokens_per_second(result.eval_count, result.eval_duration_ns),
            12.8,
        )

    def test_cached_prompt_tokens_count_the_way_ollama_counts_them(self):
        message = {
            "stop_type": "eos",
            "timings": {
                "cache_n": 10,
                "prompt_n": 502,
                "prompt_ms": 1000.0,
                "predicted_n": 4,
                "predicted_ms": 200.0,
            },
        }
        result = completion_to_result(message, 1.0, 1.0)
        self.assertEqual(result.prompt_eval_count, 512)
        self.assertEqual(result.prompt_eval_cached_count, 10)
        self.assertEqual(result.done_reason, "stop")

    def test_parity_band_stays_at_five_percent_until_spread_is_wider(self):
        self.assertEqual(parity_band([10.0, 10.1, 10.2]), 0.05)
        self.assertGreater(parity_band([10.0, 12.0]), 0.05)
        self.assertTrue(inside_band(10.4, 10.0, 0.05))
        self.assertFalse(inside_band(10.6, 10.0, 0.05))
        self.assertFalse(inside_band(9.4, 10.0, 0.05))


class ServerCommandTest(unittest.TestCase):
    def test_launch_matches_the_ollama_runner_flags(self):
        command = llama_server_command(Path("/bin/llama-server"), Path("/tmp/model.gguf"), 8081, 8192)
        self.assertEqual(command[command.index("-c") + 1], "8192")
        self.assertEqual(command[command.index("-b") + 1], "1024")
        self.assertEqual(command[command.index("-ub") + 1], "1024")
        self.assertEqual(command[command.index("--flash-attn") + 1], "auto")
        self.assertEqual(command[command.index("--keep") + 1], "4")
        self.assertIn("--context-shift", command)
        self.assertIn("--no-webui", command)


class GgufTest(unittest.TestCase):
    def test_header_and_vision_names(self):
        self.assertEqual(vision_tensors(["blk.0.attn_v.weight", "v.patch_embd.weight"]), ["v.patch_embd.weight"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tiny.gguf"
            path.write_bytes(_tiny_gguf())
            info = summarize(path)
        self.assertEqual(info["architecture"], "qwen35")
        self.assertEqual(info["file_type_name"], "Q4_K_M")
        self.assertEqual(info["vision_tensors"], ["v.patch_embd.weight"])
        self.assertEqual(info["tensor_count"], 2)


def _tiny_gguf() -> bytes:
    parts = [struct.pack("<4sIQQ", b"GGUF", 3, 2, 2)]
    parts.append(_string("general.architecture"))
    parts.append(struct.pack("<I", 8))
    parts.append(_string("qwen35"))
    parts.append(_string("general.file_type"))
    parts.append(struct.pack("<I", 4))
    parts.append(struct.pack("<I", 15))
    parts.append(_tensor("blk.0.attn_v.weight"))
    parts.append(_tensor("v.patch_embd.weight"))
    return b"".join(parts)


def _string(text: str) -> bytes:
    raw = text.encode()
    return struct.pack("<Q", len(raw)) + raw


def _tensor(name: str) -> bytes:
    return _string(name) + struct.pack("<I", 1) + struct.pack("<Q", 1) + struct.pack("<I", 15) + struct.pack("<Q", 0)


if __name__ == "__main__":
    unittest.main()
