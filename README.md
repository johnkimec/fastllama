# fastllama

Decode for Qwen3.8-27B on an M1 Max, on the same 4-bit file Ollama loads.

The kept change is wider loads in llama.cpp’s Metal mat-vec kernel, against commit `0f3a71be1`. Mean decode went from 13.17 to 14.24 tokens per second. Prefill stayed flat. Memory did not drop.

The patch is [`patches/mul_mv-wide-loads.patch`](patches/mul_mv-wide-loads.patch). The writeup is the [site](https://johnkimec.github.io/fastllama/).
