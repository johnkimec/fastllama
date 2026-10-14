# fastllama

## Status

approved

## Progress

Phase 5. Settled. D11 is decided: stop, and keep the step 7 kernel. Mean decode 14.24 tok/s, +8.12% versus Ollama, short of the 10% line at 14.49. Prefill did not move. Footprint was not improved. Status stays approved. Step 8 is measured and short of the 10% decode line. The edit is vector math in `kernel_mul_mv_q4_K_f32`. Mean decode 14.10 tok/s, +7.04% versus Ollama, about 0.14 below the step 7 mean of 14.24. Decode stayed between 14.01 and 14.18. Prefill 124.76 tok/s, −2.45%, inside the 5% band. Greedy token ids matched on 768 tokens, run `results/quality-20261014T011452Z`. Status stays approved. D8 is decided. Steps 1 through 7 are done. Step 7 is measured and short of the 10% decode line. D7 is decided: wide loads in the Q6_K decode kernel. Status stays approved. D6 is decided: the next change stays on decode weight traffic. Steps 1 through 6 are done. Step 6 is measured and short of the pass rule. The change is wide loads in `kernel_mul_mv_q4_K_f32`: mean decode 14.18 tok/s, +7.67% versus Ollama, about +8.1% versus the unmodified source build. Greedy token ids matched the unmodified source build on 768 tokens, run `results/quality-20261012T011849Z`. Prefill is flat. Footprint was not remeasured. The 10% decode line is 14.49 tok/s. Against the 24.17 tok/s ceiling this build is at 58.7%, up from Ollama's 54.5%. Detail is under Step 6. The source tree is `third_party/llama.cpp`, gitignored, branch `fastllama`, commit 0f3a71be1. The unmodified Apple Clang binary is `third_party/llama.cpp/build/bin-stock/` (run `results/parity-20261012T003904Z`). The current binary is `third_party/llama.cpp/build/bin/llama-server`. Note: `results/parity.txt`. Run: `results/parity-20261012T010510Z`. The ceiling sentence in `results/ceiling.txt` was appended by hand. A speed re-run rewrites that file and drops the sentence. Step 4 ran the llama-server binary Ollama ships (commit 0f3a71be1) as its own process. Stock Ollama was not modified. If the plan changes, follow the change rule.

Step 4, same GGUF and the 512/128 protocol: mean prefill 127.82 tok/s (−0.06%) and mean decode 13.20 tok/s (+0.20%) against `results/speed-20261011T221636Z`. Both inside the 5% band. Note: `results/parity.txt`. Run: `results/parity-20261011T231303Z`. Other finished runs: `results/speed-20261011T221636Z`, `results/memory-20261011T222500Z`, and `results/check-20261011T221314Z`. The model name is `qwen3.8-27b-text`, the context cap is 8192, and the file is `models/Qwen3.8-27B-Text-Only-Q4_K_M.gguf`.

Step 1's "near the file size" check matches resident size. `phys_footprint` does not include the clean weight mapping, and it stays the pass-rule counter until a new decision says otherwise. The 4096/512 footprint baseline is one warm run. The pass rule's memory comparison is the mean of ten.

## Problem

I want to run a local LLM (Qwen3.8-27B) on my M1 Max MacBook Pro with 64 GB of RAM, and get more tokens per second while using less memory than Ollama does on the same model. Ollama is my baseline.

This is a learning project as much as a performance one. I'm a hardware engineer by training, and I want to really understand the M1 Max's architecture (unified memory, memory bandwidth, the GPU) and how it limits inference speed. Then I want to use that understanding to make changes I can measure. I'll probably write it up afterward.

## Scope and non-goals

### Scope

First version, accepted:

- It generates text from a text prompt.
- 4-bit Qwen3.8-27B language-model file, same file on both sides. The vision projector is not loaded.
- One measured change in how inference runs, aimed at this M1 Max.
- Headline run 512/128, memory-trace run 4096/512, with the agreed pass rule.

### Non-goals

First version, accepted:

- It does not take images or video.
- It does not train the model.
- It does not replace Ollama's daemon, model library, or API. That stays the later phase.
- The writeup comes after the measurements. It is not part of the pass rule.

## Constraints

- M1 Max MacBook Pro with 64 GB of RAM.
- Model: Qwen3.8-27B.
- Let's use 4-bit for now. Both sides run that same format in memory.
- Baseline: Ollama on the same model.
- Baseline, agreed: Ollama loads only the 4-bit language-model file. fastllama loads that same file. The projector is in neither process.
- I want to optimize the stack solely for this specific hardware. Ollama is general.
- Benchmark an average over many runs of the same tests, in a controlled environment.

## Success criteria

<!-- What done means. The pass rule below is the wording they accepted. -->

- Faster: both prefill and decode. We can be at par if we can't seem to beat it. It's really just the idea that we can put together some novel architecture/theoretical improvement on the pre-existing Ollama solution.
- First version, then later: starting at one measured change in how inference runs, then moving to a stack that replaces Ollama. I am optimizing this inference stack for this specific hardware.
- Headline comparison: 512-token prompt, 128-token reply.
- Memory trace run: 4096-token prompt, 512-token reply.
- Less memory: peak memory is a good measure. Collect as many data points in the memory usage as possible so we can debug and deep dive into the memory behaviour and any spikes that might happen.
- Spikes can skew peak memory. Compare an average memory over many runs instead.
- Speed check, agreed: Discard one warmup. Compare the mean prefill and mean decode over 10 warm runs of the 512/128 test. At par means your mean is at least 95% of Ollama's mean. If Ollama's own spread is already wider than 5%, widen the band to that spread.
- Memory measurement, agreed: On the 4096/512 run, over the same 10 warm runs, compare the mean of the median physical footprint during decode. Keep the full trace and the peak so spikes can be debugged. They do not decide the comparison.
- Steady-state footprint sounds okay. Ideally, we would want to see noticeable improvements for this to be a success.
- Pass rule, agreed: The 5% band is the floor. Inside it, the run is at par and the first version is not a success. Worse than the band fails. Success means the mean prefill, the mean decode, and the steady-state footprint each beat Ollama by at least 10% on the agreed runs. If Ollama's own spread is wider than 5%, the floor widens to that spread. The 10% bar stays put.
- Comparison, with that baseline: thinking off on both sides, stop at the agreed token counts, the same context cap (large enough for the 4096/512 run, far below 262K), model already loaded. Ollama's prompt-eval rate is prefill. Ollama's eval rate is decode. Weight-load time stays out of both.

## Breakdown

### Harness

- **Responsibility:** Run the agreed protocol against Ollama and against the engine. Sample physical footprint from outside the process. Report prefill, decode, and the trace. It does not run the model.
- **Interface:** A run spec goes in (token counts, context cap, thinking off). Timings and a footprint trace come out.
- **Decisions:** D1, D2, and D3 decided. Skip: the protocol is already fixed in the success criteria. Skip: the first harness drives Ollama only. The engine adapter waits until the engine exists, because the first lab step is the Ollama profile. Skip: the implementation language does not change the interface.

### Engine

- **Responsibility:** Load the shared 4-bit language-model file and run prefill and decode on the M1 Max. The hardware-specific change lives here.
- **Interface:** Token ids, a generation budget, and a context cap go in. Output tokens, prefill time, and decode time come out.
- **Decisions:** D4 and D5 decided. D6, D7, D8, D9, D10, and D11 decided. Skip: the engine loads the same 4-bit GGUF from phase 1. The file format is not a new decision. Skip: the specific kernel change waits until the Ollama profile shows where the gap is.

### Shared file

- **Responsibility:** One 4-bit language-model file that Ollama and the engine both load. The vision projector is not in it.
- **Interface:** The file itself.
- **Decisions:** No decision. Phase 1 already fixed 4-bit, text-only, and the same file on both sides.

## Decision log

### D1 — How the harness drives a run

- **Date:** 2026-10-10
- **Context:** The harness has to time prefill and decode separately, using Ollama's own split, and later drive the engine with the same shape. The choice changes the harness interface and what the engine must answer. `ollama run` and the HTTP API both talk to the same local daemon. The difference is the shape of the reply, not where the model runs.
- **Options:**
  1. Ollama's HTTP API. The generate endpoint returns prompt-eval count and duration, and eval count and duration. The numbers are the ones Ollama already accounts. The engine does not have this API yet, so a later adapter would return the same two durations. Fits because the baseline is Ollama. Costs a dependency on the daemon, which this version is not replacing.
  2. The `ollama run` terminal output. Easy to start by hand. The printed rate is often one tokens-per-second figure, so prefill and decode are no longer separate. Fits a manual check. Breaks the pass rule, which needs both rates.
  3. llama.cpp's benchmark tool (`llama-bench` with the 512/128 pair). It is built for this exact token split and is repeatable. It is not Ollama. A win there can be a loss against Ollama's runner. Fits only if the baseline were llama.cpp, which phase 1 already ruled out.
- **AI recommendation:** Option 1. The pass rule is defined on Ollama's prompt-eval rate and eval rate. The engine can grow a small adapter that returns those same two durations. Option 3 would move the baseline you already signed off.
- **My decision:** Option 1. Ollama's HTTP API.
- **My reasoning:** The returning JSON is much richer than simply just using the Ollama CLI. The granularity and configurability of the API would be helpful when profiling. The performance numbers for the Ollama baseline come from its daemon, so the harness needs that process running. The footprint sample has to come from the runner that holds the weights, not the menu-bar app.
- **What I learned:** Ollama's timings come from a background daemon, and the weights live in the runner process it starts, not in the menu-bar app.
- **Note:** The menu-bar app is the Ollama icon in the strip at the top of the Mac screen. That process is the application you launch. It starts the runner. The weights are in the runner.
- **Status:** decided

### D2 — Which process the footprint sample comes from

- **Date:** 2026-10-10
- **Context:** The pass rule compares steady-state physical footprint during decode. The HTTP response carries the speed fields (prompt-eval and eval counts and durations) and does not carry memory. Footprint has to be sampled from the OS while the request runs. The weights sit in the runner. The menu-bar app is a different process with a much smaller footprint. The choice changes what the harness measures and whether a 10% memory win is real.
- **Options:**
  1. Sample the runner only. The harness finds the runner the menu-bar app started and polls that process's physical footprint from outside, during the request. This matches where the weights live. If a later build folds the runner into the menu-bar app, a harness that still looks for a separate runner samples nothing. If it follows the weights, it samples one process that contains both the app and the weights.
  2. Sum every Ollama process, menu-bar app plus runner. You still catch the weights if the process tree shifts. The number also includes the app's own bytes, so the 10% bar is partly about the application, not the model.
  3. Use a memory field from the HTTP response. No pid to find. The generate response does not report physical footprint, so this cannot feed the pass rule.
- **AI recommendation:** Option 1. The agreed number is the footprint of the weights during decode, and those bytes are in the runner. Option 2 dilutes that with the menu-bar app. Option 3 has nothing to read.
- **My decision:** Option 1. Sample the runner only.
- **My reasoning:** Do option 1 because our first version is really just getting the system down. We aren't so concerned about the application that calls the inference. The point is that we want to isolate the process solely. Once Ollama folds the runner into the menu-bar app, one sample contains the numbers for both the app and the weights.
- **What I learned:** The memory number is the footprint of the process that holds the weights. Isolating the runner keeps it on the inference system. Once that runner is folded into the menu-bar app, one sample mixes in the app.
- **Status:** decided

### D3 — How dense the footprint trace is

- **Date:** 2026-10-10
- **Context:** The pass rule uses the median footprint during decode. The trace is kept so spikes can be inspected. The HTTP API does not stream memory, so the harness polls the runner from outside. How often it polls changes whether a short spike is visible, and whether the poll itself moves the tokens-per-second number.
- **Options:**
  1. Poll at a fixed 10 times a second for the whole request, and mark which samples fall during decode. Spikes longer than about a tenth of a second show up. The poll is cheap enough that it should stay inside the 5% noise band. Shorter spikes are missed.
  2. Poll as fast as the OS call returns. Short spikes show up. The sampler uses CPU on the same machine as the model, so it can move both the tokens per second and the footprint you are trying to measure.
  3. One sample per generated token, taken inside the runner at token boundaries. The trace lines up with decode. Prefill is one long stretch with no token boundary, so spikes there are invisible. It also needs a hook inside Ollama's runner, which pulls memory sampling out of the harness and into Ollama.
- **AI recommendation:** Option 1. The trace is for debugging spikes, and the pass rule is a median over decode, so a tenth of a second is enough to see a real spike without the sampler becoming part of the result. Option 3 changes the breakdown.
- **My decision:** Option 1. Poll at a fixed 10 times a second.
- **My reasoning:** Polling too often would significantly degrade performance, and I expect steady-state memory usage for the most part. The intervals of option 1 may be enough to catch spikes and abnormalities.
- **What I learned:** Polling footprint is part of the measurement. Faster polling can move the speed and memory numbers, so 10 times a second is enough to see a real spike without the sampler becoming the result.
- **Status:** decided

### D4 — What the engine is built on

- **Date:** 2026-10-10
- **Context:** The first version is one measured change aimed at this M1 Max, on the same 4-bit language-model file Ollama runs. A later phase can replace the stack. A runtime is the program that loads those weights and runs each token on the chip. It is not the weight file, and it is not the Ollama daemon. GGUF is llama.cpp's file format for those weights: packed tensors, the quantization type, and the metadata the runtime needs. The 4-bit language-model file in phase 1 is a GGUF. MLX uses a different file. MLX is Apple's array framework for Apple silicon: arrays live in unified memory, and the GPU work is issued through Metal. mlx-lm is the language-model layer on top of MLX. llama.cpp is a different runtime, the one Ollama uses, with a Metal backend among others. The choice changes what the engine is, and whether a 10% win is that change or a different runtime.
- **Options:**
  1. A change inside llama.cpp, the runtime Ollama already uses on this Mac. Same file, same kernels, one hardware-specific change you can point at. A 10% win belongs to that change. The cost is the bias you already named: the design starts from Ollama's shape. Fits this version. A full replacement stays later.
  2. MLX, Apple's framework for unified memory on this machine. mlx-lm can run this model as text-only 4-bit, with the vision encoder left out. Memory placement and kernels are already aimed at Apple silicon, so the gap versus Ollama can be large. The cost is two mixes: MLX versus your change, and MLX's own 4-bit file versus the GGUF file Ollama loads. Phase 1 said both sides load the same file. Fits if "optimized for this chip" matters more than attributing one change on one file.
  3. A new runtime for this model and this chip. You meet every limit yourself. The cost is that the first version spends its time reaching Ollama, and the 10% bar waits until the kernels are already good. Fits the later phase, which is out of scope for this version.
- **AI recommendation:** Option 1. The pass rule is 10% over Ollama on one file. A change inside llama.cpp makes that 10% mean one thing. The Ollama profile still comes before any kernel change. Option 3 is the later phase you already set aside.
- **My decision:** Option 1. A change inside the llama.cpp runtime.
- **My reasoning:** Option 1 is a better comparison to the Ollama baseline. Both sides load the same GGUF file and the same runtime, so a 10% gap is a change inside llama.cpp, not a different file or a different runtime. The cost is that the design starts from llama.cpp's shape, so the novel part has to be a change I can point at. A new architecture stays later.
- **What I learned:** llama.cpp is the runtime Ollama already uses, and GGUF is the weight file it reads. Staying there keeps the comparison on one change. MLX would mix a new runtime and a different 4-bit file into the result. Choosing the runtime does not itself optimize the M1 Max.
- **Status:** decided

### D5 — Where the changed runtime runs

- **Date:** 2026-10-10
- **Context:** The baseline is stock Ollama on the shared GGUF. The engine is a change inside llama.cpp. Where that changed build runs decides whether the baseline stays measurable, and whether the harness talks to one server or two.
- **Options:**
  1. A separate llama.cpp process, loading the same GGUF. Stock Ollama is never modified. The harness grows a second client that reports prefill and decode durations in the same shape as Ollama's prompt-eval and eval fields. The baseline stays intact. The cost is two servers, and the second client's timings have to mean the same thing as Ollama's.
  2. Patch the llama.cpp that the installed Ollama runs, and keep using Ollama's HTTP API for both. One client. The cost is that replacing the build to test the change removes the baseline, unless two Ollama installs are kept and switched. The comparison is easy to corrupt.
  3. Call llama.cpp inside the harness process, with no server. Fewer processes. The cost is that the engine has no daemon and Ollama does, so the comparison includes that process difference, not only the kernel change. The footprint process is different too.
- **AI recommendation:** Option 1. The baseline Ollama install stays untouched, and the 10% belongs to the llama.cpp change on the same file. The harness's second client is the cost, and it is smaller than losing the baseline.
- **My decision:** Option 1. A separate llama.cpp process. Stock Ollama stays untouched.
- **My reasoning:** It's the most apples-to-apples comparison because stock Ollama stays untouched. The cost is that the harness has two clients. The second process has to report prefill and decode the same way Ollama's prompt-eval and eval fields do. If its clocks include something Ollama's clocks do not, the comparison is less apples-to-apples than the separate install suggests.
- **What I learned:** A separate llama.cpp process keeps the Ollama baseline in place. The comparison stays fair only when that process's prefill and decode clocks mean the same thing as Ollama's.
- **Status:** decided

### D6 — What the next change aims at

- **Date:** 2026-10-11
- **Context:** Step 6 was the one hardware-specific change, wide loads in `kernel_mul_mv_q4_K_f32`. Mean decode is 14.18 tok/s, 58.7% of the 24.17 tok/s ceiling, +7.67% versus Ollama and short of the 10% line at 14.49. Prefill stayed at 127.50 tok/s. The loads do not change how many bytes stay resident, and the footprint was not remeasured. Greedy token ids matched the unmodified build on 768 tokens. The pass rule still wants 10% on prefill, decode, and steady-state footprint together. A second change has to name which of those three it is allowed to move, because they are not the same bytes. The weight file is 15.41 GiB and is a clean mapping, so it is not in `phys_footprint`. On the tiny check that counter was 0.93 GiB. On the 4096/512 run it was 8.17 GiB. This file is a hybrid: 64 layers, full attention on one layer in four, 4 KV heads, key and value length 256. At f16 that cache is 64 KiB per token, about 0.3 GiB at the 4096/512 length. The recurrent state does not grow with the prompt and is a fraction of a GiB. The dirty growth of about 7 GiB is neither the weight file nor that cache.
- **Options:**
  1. Stay on decode weight traffic, past the wide loads. The kernel reads the same 16.55 GB per token and was not filling the bus: wider loads brought back about one of the eleven missing tokens per second. More work on that path can raise decode. It does not move prefill, and it does not cut `phys_footprint`. Fits this session if the goal is tokens per second. The three-part pass rule stays failed on prefill and on memory.
  2. Cut the dirty bytes in the 4096/512 footprint. That is the memory half of the pass rule. The file itself is not in the counter, and the KV cache at this length is about 0.3 GiB, so the cut has to be the other dirty buffer, which is not named yet. Fits if this session is the memory number. A wrong name edits a buffer that is not the 7 GiB. It raises decode only if those bytes are extra weight traffic.
  3. Change the prefill kernel. Prefill is 127.50 tok/s, 5.3 times one read of the weights, so it is not waiting on a second read. Fits only if the prefill line of the pass rule is the next target. It does not close the decode gap and it does not shrink the decode footprint.
- **AI recommendation:** Option 1. The hole in tokens per second is still decode weight traffic, and the wide loads closed only about one token per second of it. Prefill is already several times the one-read rate. The memory number is real, and it is the dirty footprint, not the file. Naming that buffer is a separate step. Putting it in the same change as the next kernel would mix two results.
- **My decision:** Option 1. Let's do option 1.
- **My reasoning:** The missing tokens per second are still the decode weight stream, and the wide loads showed the bus still has room. This change leaves prefill and the dirty footprint where they are, so the three-part pass rule stays failed on those two until a later change.
- **What I learned:** Wider loads made the same weight bytes faster, so decode is still short of one full read of the file. Prefill and the footprint are other bytes, and this change will not move them.
- **Status:** decided

### D7 — Which edit on the decode weight stream

- **Date:** 2026-10-11
- **Context:** Step 7 is one named edit, past the wide loads, on the kernels that move the weight bytes during decode. The file is 12.08 GB of Q4_K (73.0%) and 4.45 GB of Q6_K (26.9%). The wide loads are only in `kernel_mul_mv_q4_K_f32`. After the load, that kernel unpacks the `float4` and `uint2` values back into scalars and multiplies one nibble at a time. `kernel_mul_mv_q6_K_f32` still loads the activation one float at a time. Both kernels handle 2 rows per simdgroup (`N_R0` is 2), so each activation load is shared by 2 weight rows. A second copy of the Q4_K values, tried before the wide loads, scored 12.01 tok/s and was replaced. That pipeline is not an option here.
- **Options:**
  1. Wide loads in `kernel_mul_mv_q6_K_f32`, the same kind of change as the Q4_K loads. The activation arrives as `float4` and the weight slice arrives as a wider integer load. The arithmetic stays the one already in that kernel. This is the 4.45 GB the first edit did not touch. Fits because wider loads of the same bytes are the change that raised decode. The Q4_K math stays scalar, so if Q6_K is a small share of the token time the gain is capped by that share.
  2. Vectorize the Q4_K nibble multiply so the wide registers are consumed by vector math instead of being unpacked into scalars. This is the 12.08 GB, and it is the kernel that already moved. Fits if the remaining stall is the scalar nibble loop behind the wide load. The cost is a new shape for the arithmetic. The earlier second-copy pipeline on this kernel got slower.
  3. Raise `N_R0_Q4_K` from 2 to 4, so each activation load is shared by 4 weight rows. The repeated activation loads issue about two bytes for each quantized weight byte. They cannot all miss DRAM: at 14.18 tok/s an extra stream that large would need more than the 400 GB/s bus. Fits if the next win is fewer repeated loads of a cached activation. The cost is fewer threadgroups, and the Q6_K loads stay narrow.
- **AI recommendation:** Option 1. The Q4_K wide loads are the edit that raised decode, and a quarter of the file still uses the narrow-load kernel. Repeating that change on Q6_K follows the measurement. Vectorizing the Q4_K multiply is the follow-on if the Q6_K share turns out to be small. Raising the row count spends the step on loads that mostly hit cache.
- **My decision:** Option 1.
- **My reasoning:** Option 1 because we experienced gains on the other decode kernel, so it might make sense that we see some improvements here.
- **What I learned:** Optimizing the performance of a part of the entire computation is capped by its contribution amount.
- **Status:** decided

### D8 — Which edit follows the Q6_K loads

- **Date:** 2026-10-11
- **Context:** Step 7 is measured. Wide activation loads in `kernel_mul_mv_q6_K_f32` put mean decode at 14.24 tok/s, +8.12% versus Ollama's 13.17, about 0.06 above the Q4-only mean of 14.18. The 10% line is 14.49. Prefill stayed at 127.58 tok/s. The Q6_K weight bytes stayed byte loads, because a block is 210 bytes. Q4_K is 12.08 GB, 73.0% of the file. In `kernel_mul_mv_q4_K_f32` the activation is already a `float4` and the weight slice is already a `uint2`. The kernel then splits those registers into scalars and multiplies one nibble at a time. D6 still holds: this edit stays on decode weight traffic. Prefill and the dirty footprint are not this step. The second-copy pipeline that scored 12.01 tok/s is not an option.
- **Options:**
  1. Vectorize the Q4_K nibble multiply, so the `float4` and `uint2` are consumed by vector math instead of being unpacked into scalar arrays. This is the 12.08 GB, and it is the kernel that already moved. Fits if the scalar nibble loop is what keeps those wide loads from staying in flight on the bus. The cost is a new shape for the arithmetic. If that loop is already hidden behind the weight read, decode will not move. A wrong shape can regress, as the second-copy pipeline did.
  2. Raise `N_R0_Q4_K` from 2 to 4, so each activation load is shared by 4 weight rows. Those repeated activation loads are mostly cache hits: at 14.24 tok/s they cannot all miss the 400 GB/s bus. Fits if the next win is fewer issues of a cached activation. The cost is half as many threadgroups, which can hide less of the weight-read latency. The bytes read from DRAM per token do not change. The Q6_K kernel stays as it is.
  3. Split the Q6_K weight load into an aligned wide load on even blocks and a byte load on odd blocks. This finishes the load step 7 left narrow. Fits if those remaining Q6_K bytes are still worth a load-width edit. The cost is the cap already measured: Q6_K is 26.9% of the file, and widening its activation loads moved decode by 0.06 tok/s.
- **AI recommendation:** Option 1. The Q6_K edit showed that another wide load on the smaller share does not close the gap. The Q4_K kernel already holds the wide registers and splits them before the multiply. That is 73% of the file. Raising the row count spends the step on cached activation loads and fewer threadgroups. Splitting the Q6_K weight load spends it on the share that just failed to move the mean.
- **My decision:** Option 1. Just do it.
- **My reasoning:** The Q4_K kernel already paid for the wide load and then split that register into one nibble at a time. That split sits on 73% of the file, and the Q6_K loads showed the other 27% cannot close the 0.25 tok/s left. If the scalar loop is already hidden behind the weight read, decode stays at 14.24, and a wrong shape of this same kernel already fell to 12.01.
- **What I learned:** A wide load fills the bus only while the datapath keeps the value wide. Splitting it back into scalars spends the instruction issue the load just saved.
- **Status:** decided

### D9 — Whether the step 8 kernel stays

- **Date:** 2026-10-13
- **Context:** Step 8 is measured. The dot products in `kernel_mul_mv_q4_K_f32` put mean decode at 14.10 tok/s, between 14.01 and 14.18. Step 7, the same wide loads plus the Q6_K `float4` activations and the scalar nibble loop, was 14.24. This edit is about 0.14 tok/s slower. It is +7.04% versus Ollama's 13.17, short of 14.49. Prefill was 124.76 tok/s, inside the 5% band. Greedy token ids matched. The wide loads and the Q6_K activation loads are still in the tree under the dot products. `third_party/llama.cpp` is gitignored, so a reset to commit 0f3a71be1 would also drop those loads. D8 stays decided: it records the edit that was measured.
- **Options:**
  1. Remove the dot products and put the scalar nibble loop back. The wide loads and the Q6_K `float4` activations stay. The source matches the 14.24 build. The step 8 numbers stay in the log. Fits because the next measurement should start from the faster kernel. The cost is that this dot-product shape has to be written again if a later decision wants it.
  2. Leave the dot products in the tree. The binary stays at 14.10. Fits if the next edit is a change to those dot products. The cost is that every later run starts 0.14 tok/s behind the best build, and a gain has to climb back to 14.24 before it is new.
- **AI recommendation:** Option 1. The pass rule scores the binary that runs. The dot products lost decode and did not change the token ids, so the lesson is already in the measurement. The wide loads are the part that raised decode, and they stay.
- **My decision:** Option 1.
- **My reasoning:** Revert because it made performance worse.
- **What I learned:** It made performance worse.
- **Status:** decided

### D10 — Which edit follows the failed Q4 dot products

- **Date:** 2026-10-13
- **Context:** Step 8 measured the Q4 nibble dot products at 14.10 tok/s, about 0.14 below the scalar loop at 14.24. The wide Q4 loads stay. The bus is still about 59% of the 24.17 tok/s ceiling. Q6_K is 4.45 GB, 26.9% of the file. Its activations are already `float4`. Its weight bytes are still loaded one at a time. A Q6_K block is 210 bytes. A 4-byte load is aligned on even blocks only, and those threads share a simdgroup with the odd blocks, so a branch would run both load paths. A 2-byte load is aligned on every block. D9 puts the scalar Q4 loop back before this edit. Raising `N_R0` cuts the threadgroup count, which hides less of a weight-read stall.
- **Options:**
  1. Wide loads of the Q6_K weight bytes in `kernel_mul_mv_q6_K_f32`, using the 2-byte load that is aligned for every block. The arithmetic stays the scalar form already in that kernel. This is the 4.45 GB the Q4 edit did not touch. Fits because wider loads of weight bytes are the change that raised decode. The cost is the cap from Q6_K's share: if those bytes already move as fast as the Q4 bytes, the gain stays small.
  2. Raise `N_R0_Q4_K` from 2 to 4, so each activation load is shared by 4 weight rows. Fits if the next win is fewer issues of a cached activation. The cost is half as many threadgroups on a kernel that is still waiting on the weight read.
- **AI recommendation:** Option 1. The dot products showed the scalar Q4 math is not the stall. The remaining narrow weight loads are the Q6_K bytes. A 2-byte load keeps the simdgroup on one path. Raising the row count spends the step on fewer threadgroups.
- **My decision:** Option 1. Go ahead and do it.
- **My reasoning:** Go ahead and do it.
- **Status:** decided

### D11 — Whether to keep going

- **Date:** 2026-10-13
- **Context:** Step 7 is the fastest measured build: mean decode 14.24 tok/s, between 14.20 and 14.27, +8.12% versus Ollama's 13.17. The 10% line is 14.49. Prefill stayed at 127.58 tok/s. The Q4 dot products measured 14.10. The Q6 `ushort` weight loads measured 14.02, and the runs before the sag sat at 14.09 to 14.13. The pass rule wants 10% on prefill, decode, and steady-state footprint together. Prefill did not move. The footprint was not remeasured, and these load changes do not cut resident bytes. The ~7 GiB dirty buffer is still unnamed.
- **Options:**
  1. Stop. Put the Q6 weight loads back to bytes, so the source matches the step 7 kernel: Q4 wide loads, the scalar nibble loop, and Q6 `float4` activations. The kept numbers are the step 7 run. No new speed run. Fits because the later edits measured slower, and another run on a hot machine will not make the pass rule.
  2. Stop and leave the `ushort` loads in the tree. The source stays at the slower measurement. Fits only if the last edit should remain even though it lost decode.
- **AI recommendation:** Option 1. The binary that stays is the one that measured fastest. The later runs stay in the log.
- **My decision:** Settle now.
- **My reasoning:** This seems cooked.
- **What I learned:** This seems cooked.
- **Status:** decided

<!-- Status: open | decided | superseded by D<n>. Supersede; do not delete. -->

## Implementation plan

### Step 1 — Load the shared file in stock Ollama

- **Depends on:** none
- **Verify:** Ollama generates text from the 4-bit language-model GGUF with no vision projector loaded. The runner's physical footprint sits near the size of that file, about 15 GB plus a small overhead.

### Step 2 — Build the harness against stock Ollama

- **Depends on:** D1, D2, D3
- **Verify:** Discard one warmup. Ten warm runs of 512 prompt tokens and 128 reply tokens report mean prefill and mean decode from Ollama's prompt-eval and eval fields. A 4096/512 run writes a footprint trace of the runner at 10 samples a second, with decode samples marked.

### Step 3 — Record the baseline against the bandwidth ceiling

- **Depends on:** D1, D2, D3
- **Verify:** The mean decode rate from step 2 is written next to the ceiling for this file, about 27 tokens per second at 400 GB/s and about 15 GB of weights. The note says how far Ollama sits from that ceiling, for decode and for prefill.

### Step 4 — Run unmodified llama.cpp beside Ollama

- **Depends on:** D5
- **Verify:** This step checks close-to-parity before any kernel change. The same protocol, on the same GGUF, through the second process, lands inside 5% of Ollama's mean prefill and mean decode. If it does not, the clock mismatch is written down before any kernel change. A later 10% win then belongs to the change, not to the second process.

### Step 5 — Name the limit the gap points at

- **Depends on:** D4
- **Verify:** A short note names the limit the step 3 gap points at: extra weight traffic, prefill compute, or steady-state footprint. The kernel change is not started before that note exists.

### Step 6 — One hardware-specific change in the separate llama.cpp

- **Depends on:** D4, D5
- **Verify:** The change is designed after step 5, by reading the llama.cpp kernels that own the named limit, and then implemented in the separate process. The harness shows mean prefill, mean decode, and steady-state footprint each at least 10% better than stock Ollama, or classifies the run as inside the 5% floor or worse than that floor. The footprint trace is still written.
- **Measured:** Wide loads only, in the registers the kernel already had. The activation is read as `float4` and each weight slice as `uint2`. A second copy of those values was tried first and replaced: that pipeline scored decode 12.01 tok/s, −8.84% versus Ollama, run `results/parity-20261012T005650Z`. The wide-load build, run `results/parity-20261012T010510Z`, put every warm decode run between 14.12 and 14.21 tok/s. Ollama's own decode runs sit between 13.08 and 13.21. Mean decode is 14.18 tok/s, +7.67% versus Ollama's 13.17 and about +8.1% versus the unmodified source build's 13.12. Mean prefill is 127.50 tok/s, −0.31%. A greedy 32-token reply matched the unmodified build. A longer greedy check, run `results/quality-20261012T011849Z`, also matched: 768 token ids across a 512-token reply, a second 128-token reply, and the 512/128 protocol prompt, temperature 0, seed 1. Chosen-token logprobs are not bit-identical. The largest absolute gap was 0.00056, and the worst gap was 0.69% of the stock margin to the runner-up. The gap did not grow across the 512-token reply. Footprint was not remeasured. These loads do not change how many bytes stay resident. The pass rule still wants 10% on prefill, decode, and steady-state footprint together. This run is short of that: the 10% decode line is 14.49 tok/s, about 0.3 tok/s above this mean, and prefill did not move. Against the 24.17 tok/s ceiling, Ollama was at 54.5% and this build is at 58.7%. About one of the eleven missing tokens per second came back.

### Step 7 — A second change on the decode weight stream

- **Depends on:** D4, D5, D6, D7
- **Verify:** The named edit is wide loads in `kernel_mul_mv_q6_K_f32` (D7), past the wide loads in `kernel_mul_mv_q4_K_f32`, implemented in the separate process. One named edit. The harness, on the 512/128 protocol, classifies mean decode against Ollama as at least 10% faster, inside the 5% floor, or worse than that floor. Mean prefill is reported. A greedy reply is compared with the unmodified source build on token ids and text. The footprint is not the target of this step.
- **Measured:** The activation in `kernel_mul_mv_q6_K_f32` is loaded as `float4`. The weight bytes stay byte loads, because a Q6_K block is 210 bytes and a wider integer load is misaligned on odd blocks. Greedy check `results/quality-q6-20261012T014248Z`: all 768 token ids matched the Q4-only build and the unmodified source build. Chosen-token logprobs matched the Q4-only build exactly. A speed run taken while the machine was still hot, `results/parity-20261012T014626Z`, fell from 13.22 to 11.61 tok/s and is not the result. The cooled run, `results/parity-20261012T015145Z`, kept decode between 14.20 and 14.27 tok/s. Mean prefill 127.58 tok/s, −0.24% versus Ollama. Mean decode 14.24 tok/s, +8.12% versus Ollama's 13.17, about 0.06 above the Q4-only mean of 14.18. The two spreads meet. Short of the 10% line at 14.49 tok/s. Note: `results/parity.txt`.

### Step 8 — Vectorize the Q4_K nibble multiply

- **Depends on:** D4, D5, D6, D8
- **Verify:** The named edit is vector math in `kernel_mul_mv_q4_K_f32` (D8), so the `float4` and `uint2` are consumed without being unpacked into scalar arrays. One named edit, in the separate process. The Q6_K kernel stays as step 7 left it. The harness, on the 512/128 protocol, classifies mean decode against Ollama as at least 10% faster, inside the 5% floor, or worse than that floor. Mean prefill is reported. A greedy reply is compared with the unmodified source build on token ids and text. The footprint is not the target of this step.
- **Measured:** The nibble multiply in `kernel_mul_mv_q4_K_f32` is four `dot` products of the `float4` activations with the nibbles of the `uint2` weight load. The scalar arrays are gone. Greedy check `results/quality-20261014T011452Z`: all 768 token ids matched the unmodified source build, and the text matched. The largest chosen-token logprob gap was 0.00055. The worst gap was 0.55% of the stock margin to the runner-up. A speed run taken while the machine was still hot, `results/parity-20261014T012027Z`, held 14.12 to 14.19 for eight runs and then fell to 13.61 and 13.12. It is not the result. The later run, `results/parity-20261014T012630Z`, kept decode between 14.01 and 14.18. Mean prefill 124.76 tok/s, −2.45% versus Ollama, inside the 5% band. Mean decode 14.10 tok/s, +7.04% versus Ollama's 13.17, about 0.14 below the step 7 mean of 14.24 and about 0.08 below the Q4-only mean of 14.18. Short of the 10% line at 14.49 tok/s. Against the 24.17 tok/s ceiling this build is at 58.3%. Note: `results/parity.txt`. The harness still writes the step 4 sentence when decode is outside the 5% band, including when decode is faster than that band.

### Step 9 — Wide loads of the Q6_K weight bytes

- **Depends on:** D4, D5, D6, D9, D10
- **Verify:** The dot products from step 8 are gone, and the Q4_K wide loads stay. The named edit is a 2-byte load of the Q6_K weight bytes in `kernel_mul_mv_q6_K_f32` (D10), on every block, because that width is aligned for the whole simdgroup. One named edit, in the separate process. The harness, on the 512/128 protocol, classifies mean decode against Ollama as at least 10% faster, inside the 5% floor, or worse than that floor. Mean prefill is reported. A greedy reply is compared with the unmodified source build on token ids and text. The footprint is not the target of this step.
- **Measured:** The Q4_K nibble multiply is the scalar loop again, and the wide `float4` and `uint2` loads stay. Each Q6_K weight group is two `ushort` loads. Greedy check `results/quality-20261014T015041Z`: all 768 token ids matched the unmodified source build, and the text matched. The largest chosen-token logprob gap was 0.00027. The worst gap was 0.69% of the stock margin to the runner-up. A speed run taken while the machine was still hot, `results/parity-20261014T015609Z`, started at 14.10 and fell to 13.53. It is not the result. The later run, `results/parity-20261014T020254Z`, started at 14.10, 14.13, 14.09, and 14.13, then sagged to 13.81. Mean prefill 125.02 tok/s, −2.25% versus Ollama, inside the 5% band. Mean decode 14.02 tok/s, +6.47% versus Ollama's 13.17. The four runs before the sag are about 14.11, below the step 7 band of 14.20 to 14.27. Short of the 10% line at 14.49 tok/s. Note: `results/parity.txt`.

### Step 10 — Settle on the step 7 kernel

- **Depends on:** D11
- **Verify:** The Q6_K weight loads are bytes again. The Q4_K wide loads, the scalar nibble loop, and the Q6_K `float4` activations stay. No new speed run. The kept measurement is step 7, run `results/parity-20261012T015145Z`: mean decode 14.24 tok/s, +8.12% versus Ollama, short of 14.49. Mean prefill 127.58 tok/s. The pass rule is not met.

## Open questions

- How much headroom is left once Ollama is measured against the memory-bandwidth ceiling for the 4-bit language-model weights. Answered in `results/ceiling.txt`: mean decode is 13.17 tok/s, 54.5% of a 24.17 tok/s ceiling. The wide-load build is 14.18 tok/s, 58.7% of that ceiling. It is not a design choice.
- Which buffer the ~7 GiB dirty growth is, from 0.93 GiB `phys_footprint` on a tiny prompt to 8.17 GiB on the 4096/512 run. The weight file is a clean mapping, and the KV cache at that length is about 0.3 GiB. D6 chose decode weight traffic, so this buffer stays unnamed.

## Change history

- 2026-10-10 — Created. Phase 1 started. Problem captured in your words.
- 2026-10-10 — Success criteria started: both speeds if possible, peak memory plus a memory trace for spikes. Benchmark length and the pass rule still open.
- 2026-10-10 — Benchmark lengths agreed: 512/128 headline, 4096/512 memory trace. Speed goal relaxed to beat Ollama or be at par. What "at par" and "novel improvement" mean is still open.
- 2026-10-10 — First version is one hardware-specific change, then a later replacement of the Ollama stack. Benchmarks are averages over many controlled runs. Average memory proposed because spikes skew the peak. Numeric pass rule still open.
- 2026-10-10 — Speed check agreed: 10 warm runs of 512/128, at par within 5% of Ollama's mean, widened to Ollama's spread when that is larger. Memory measurement agreed: mean of median physical footprint during decode on 4096/512. Whether that footprint must be lower than Ollama is still open.
- 2026-10-10 — Steady-state footprint accepted as the memory number to judge. Success described as noticeable improvements. The size of that improvement is still open.
- 2026-10-10 — Pass rule agreed. The 5% band is the floor. Success means mean prefill, mean decode, and steady-state footprint each beat Ollama by at least 10%. The 10% bar does not widen with Ollama's spread.
- 2026-10-10 — Clarified the open weight question: bit width in unified memory, not disk storage. Non-goals still open.
- 2026-10-10 — First version uses 4-bit weights for both sides. Non-goals still open.
- 2026-10-10 — First-version scope accepted: text prompts only, no vision input, no training, no Ollama product replacement, writeup after the measurements. Fairness of the Ollama comparison still open.
- 2026-10-10 — Explained that the vision encoder is a separate projector file from the language-model weights. Baseline choice still open.
- 2026-10-10 — Baseline agreed: both sides load the same 4-bit language-model file. The vision projector is in neither process. Phase 1 ready for sign-off.
- 2026-10-10 — Phase 1 signed off.
- 2026-10-10 — Phase 2 signed off. Breakdown: harness, engine, and one shared 4-bit language-model file.
- 2026-10-10 — D1 opened: how the harness drives a run. Status stays draft. Building waits for approval.
- 2026-10-10 — D1 pick is option 1, Ollama's HTTP API. Reasoning and what you learned are still open.
- 2026-10-10 — D1 reasoning drafted: the JSON is richer and more configurable than the CLI, which helps profiling. Daemon dependency and what you learned are still open.
- 2026-10-10 — D1 reason and learning drafted from your words: baseline numbers come from a daemon, and OS interleaving may make the comparison less apples-to-apples. Wording not confirmed yet.
- 2026-10-10 — You flagged the D1 interleaving claim as possibly off. Revised reason not written yet.
- 2026-10-10 — D1 decided. HTTP API. Reason includes the daemon dependency and sampling the runner rather than the menu-bar app. D2 opened: which process the footprint sample comes from.
- 2026-10-10 — D2 pick is option 1, sample the runner only. Reason connects to measuring the inference system. The cost if that process is merged, and what you learned, are still open.
- 2026-10-10 — D2 reason now includes the fold-in cost: one sample would contain both the app and the weights. What you learned is still open.
- 2026-10-10 — D2 decided. Sample the runner only. D3 opened: how dense the footprint trace is.
- 2026-10-10 — D3 pick is option 1, 10 samples a second. Reason: faster polling would degrade performance, and steady-state is the expected shape. What you learned is still open.
- 2026-10-10 — D3 decided. Footprint is polled 10 times a second. D4 opened: what the engine is built on.
- 2026-10-10 — D4 pick is option 1, a change inside llama.cpp. Reason and what you learned are still open.
- 2026-10-10 — D4 decided. The engine is a change inside llama.cpp on the same GGUF. D5 opened: where that changed build runs.
- 2026-10-10 — D5 pick is option 1, a separate llama.cpp process. Reason is the fair comparison. The second client's timings, and what you learned, are still open.
- 2026-10-11 — D5 reason and learning drafted: stock Ollama stays untouched, and the second process's clocks have to match Ollama's. Wording not confirmed yet.
- 2026-10-11 — D5 decided. Changed llama.cpp runs as a separate process. Stock Ollama stays untouched.
- 2026-10-11 — Phase 4 plan written. Waiting on sign-off.
- 2026-10-11 — Step 4 purpose stated: unmodified llama.cpp must sit near stock Ollama before any kernel change.
- 2026-10-11 — Phase 4 signed off.
- 2026-10-11 — Step 6 notes that the kernel change is designed after step 5, by reading the llama.cpp kernels for the named limit.
- 2026-10-11 — Status set to approved.
- 2026-10-11 — Progress updated for a new session. Steps 1–3 are done. Step 4 is next. The step 5 note is in `results/ceiling.txt` and is not kept by a speed re-run.
- 2026-10-11 — Step 4 done. Unmodified llama.cpp (commit 0f3a71be1) on the shared GGUF sits inside 5% of Ollama: prefill −0.06%, decode +0.20%. Note in `results/parity.txt`. Step 6 is next.
- 2026-10-11 — Progress corrected. Steps 1–5 are done. Step 4 used Ollama's shipped llama-server, not a source tree. Step 6 is next.
- 2026-10-11 — llama.cpp checked out at 0f3a71be1 on branch fastllama under third_party/llama.cpp, gitignored. llama-server built with Apple Clang. Parity of that rebuild has not been run.
- 2026-10-11 — Unmodified source build (Apple Clang, commit 0f3a71be1) sits inside 5% of Ollama: prefill −0.79%, decode −0.41%. Note in `results/parity.txt`. Run: `results/parity-20261012T003904Z`. No kernel file has been edited.
- 2026-10-11 — Pipelined `kernel_mul_mv_q4_K_f32` built and measured. Greedy text matched the unmodified build. Prefill −1.23% (inside 5%). Decode 12.01 tok/s, −8.84% versus Ollama and slower than the unmodified source build's 13.12 tok/s. Run: `results/parity-20261012T005650Z`.
- 2026-10-11 — Replaced the pipeline with wide loads only in `kernel_mul_mv_q4_K_f32`. Greedy text matched. Prefill 127.50 tok/s (−0.31%). Decode 14.18 tok/s, +7.67% versus Ollama, short of the 10% bar. Run: `results/parity-20261012T010510Z`.
- 2026-10-11 — Step 6 records the wide-load result: stable decode between 14.12 and 14.21 tok/s, +7.67% versus Ollama, 58.7% of the 24.17 tok/s ceiling. Prefill flat. Footprint not remeasured. Short of the three-part 10% pass rule.
- 2026-10-11 — Greedy quality check of the wide-load kernel against the unmodified source build. All 768 token ids matched. Largest chosen-token logprob gap 0.00056. Worst gap was 0.69% of the margin to the runner-up, and the gap did not grow over 512 tokens. Run: `results/quality-20261012T011849Z`.
- 2026-10-11 — D6 opened. The next change has to name whether it aims at decode weight traffic, the dirty footprint, or prefill. Recommendation is decode weight traffic.
- 2026-10-11 — D6 pick is option 1, decode weight traffic. Reasoning and what you learned are still open.
- 2026-10-11 — D6 decided. The next change stays on the decode weight stream. Prefill and the dirty footprint stay where they are. Step 7 written, waiting on sign-off.
- 2026-10-11 — Step 7 signed off. The next edit is one named change on the decode weight stream, past the wide loads.
- 2026-10-11 — D7 opened. The file is 73.0% Q4_K and 26.9% Q6_K. The Q6_K decode kernel still loads one float at a time. Recommendation is wide loads in that kernel.
- 2026-10-11 — D7 pick is option 1, wide loads in the Q6_K decode kernel. The reason cites the Q4_K gain. The cap from Q6_K's share, and what you learned, are still open.
- 2026-10-11 — D7 decided. Wide loads in the Q6_K decode kernel. The reason is the Q4_K gain. The learning is that a part's speedup is capped by its contribution.
- 2026-10-11 — Step 7 measured. Q6_K activation loads are float4. Weight loads stay bytes because the 210-byte block is misaligned on odd blocks. Cooled decode 14.24 tok/s, +8.12% versus Ollama, about 0.06 above the Q4-only mean. Greedy logprobs matched the Q4-only build exactly. Short of 14.49 tok/s. Run: `results/parity-20261012T015145Z`.
- 2026-10-11 — D8 opened. The next edit follows the Q6_K loads and stays on the decode weight stream. Recommendation is to vectorize the Q4_K nibble multiply. The pick, the reason, and what you learned are still open.
- 2026-10-13 — D8 pick is option 1, vectorize the Q4_K nibble multiply. The reason and what you learned are still open. The kernel change waits on those two lines.
- 2026-10-13 — D8 decided. Vectorize the Q4_K nibble multiply. The reason is that the split sits on 73% of the file and the smaller share cannot close the gap. The learning is that a wide load fills the bus only while the value stays wide. Step 8 written, waiting on sign-off.
- 2026-10-13 — Step 8 signed off. The edit is vector math in `kernel_mul_mv_q4_K_f32`. The Q6_K kernel stays as step 7 left it.
- 2026-10-13 — Step 8 measured. Q4_K nibble multiply is four dot products on the wide registers. Cooled decode 14.10 tok/s, +7.04% versus Ollama, about 0.14 below the step 7 mean. Prefill −2.45%, inside the 5% band. Greedy token ids matched on 768 tokens. Short of 14.49 tok/s. Run: `results/parity-20261014T012630Z`. The hot run `results/parity-20261014T012027Z` is not the result.
- 2026-10-13 — D9 opened. Whether the step 8 kernel stays. Recommendation is to put the scalar nibble loop back and keep the wide loads. The pick, the reason, and what you learned are still open.
- 2026-10-13 — D9 pick is option 1, put the scalar nibble loop back. The reason is that the dot products made performance worse. What you learned is still open.
- 2026-10-13 — D9 decided. The dot products come out. The reason is that they made performance worse. The learning is that it made performance worse.
- 2026-10-13 — D10 decided. Wide loads of the Q6_K weight bytes. The load is 2 bytes, aligned for every block in the simdgroup. Step 9 signed off.
- 2026-10-13 — Step 9 measured. Q4 dot products reverted. Q6_K weights load as ushort. Greedy token ids matched on 768 tokens. Decode before the sag is 14.09 to 14.13, below the step 7 band. Mean decode 14.02 tok/s, +6.47% versus Ollama. Short of 14.49 tok/s. Run: `results/parity-20261014T020254Z`. The hot run `results/parity-20261014T015609Z` is not the result.
- 2026-10-13 — D11 decided. Settle. The reason is that this seems cooked. The source returns to the step 7 kernel. No new speed run. The kept measurement is 14.24 tok/s, short of the pass rule.
