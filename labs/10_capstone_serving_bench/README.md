# Lab 10 - Capstone: Multi-Precision, Multi-GPU Serving Benchmark

**Goal:** bring every precision this curriculum covered together in one
place, served the way it would actually be deployed -- through vLLM, with
tensor parallelism across multiple B200s -- and measure the real
throughput/latency/memory/quality trade-off between them.

**Runtime:** this is the capstone and genuinely takes a while: expect
**45-90 minutes** for the full sweep (4 precisions x 3 concurrency levels,
each needing its own vLLM engine startup + weight loading across multiple
GPUs). See "Scoping this down" below if you want a faster run.

## What this pulls together

| Precision | How it's produced here | Which earlier lab covered the technique |
|---|---|---|
| BF16 | vLLM's default, no quantization | the baseline in every previous lab |
| FP8 | vLLM's built-in on-the-fly FP8 weight quantization (`quantization="fp8"`) | lab 05 |
| INT4 | vLLM's built-in on-the-fly `bitsandbytes` NF4 quantization (`quantization="bitsandbytes"`) | labs 01-03 (RTN/GPTQ/AWQ) and lab 08 (NF4 specifically) |
| NVFP4 | vLLM's `modelopt`-style FP4 quantization path, **best-effort** -- see below | lab 06 |

This lab deliberately uses vLLM's **on-the-fly** quantization options
(load a normal bf16 checkpoint, quantize during engine startup) instead of
pre-producing separate quantized checkpoints with the libraries from labs
02/03/05/06 -- that keeps this already-long capstone from also needing a
32B-scale offline quantization pass per precision. The accuracy/format
trade-offs you already measured in those labs on smaller models still
apply; this lab's job is to show the *serving-time* trade-off (throughput,
latency, memory) that those techniques exist to unlock in the first place.

**NVFP4 is best-effort**: whether vLLM's installed version exposes a
working on-the-fly FP4 quantization path for this model depends heavily on
your `vllm`/`nvidia-modelopt` versions and the specific model architecture.
The script tries it, and if it fails, skips that one row of the comparison
with a clear message and a suggestion (build a pre-quantized NVFP4
checkpoint with `nvidia-modelopt`, the way lab 06's accelerated path does,
then point `--model` at that checkpoint) rather than blocking the rest of
the lab.

## What the script does

1. Loads Qwen2.5-14B-Instruct by default (see "Scoping this down" to use
   the full 32B model) with `tensor_parallel_size=4` (4 of your 8 B200s --
   change `TP_SIZE` to use more/fewer).
2. For each precision: starts a fresh vLLM engine, measures GPU memory via
   `nvidia-smi` right after weight loading, evaluates an approximate
   perplexity using vLLM's `prompt_logprobs` (teacher-forced log-likelihood
   of held-out WikiText-2 passages -- the same idea as `common/eval.py`'s
   perplexity, computed through vLLM instead of a raw HF forward pass),
   then sweeps request concurrency (`1, 8, 32` identical requests in one
   batch) measuring aggregate throughput and batch completion time.
3. Tears the engine down and moves to the next precision, writing results
   incrementally so a long run's progress isn't lost if one precision
   fails partway through.
4. Saves a combined throughput-vs-concurrency chart (one line per
   precision), a memory bar chart, and a quality (perplexity) bar chart.

## Run it

```bash
python labs/10_capstone_serving_bench/lab10_multi_precision_serving_bench.py
```

### Scoping this down

- **Faster**: edit the script's `CONCURRENCIES` list to just `[1, 8]`, or
  `PRECISION_CONFIGS` to drop the NVFP4 attempt.
- **Bigger** (uses more of your node): change `MODEL_KEY` from
  `"capstone_fallback"` (Qwen2.5-14B-Instruct) to `"capstone"`
  (Qwen2.5-32B-Instruct) and/or raise `TP_SIZE` toward 8.
- **More realistic latency numbers**: this script uses vLLM's simple
  offline batch API (`LLM.generate`), which reports batch completion time
  and aggregate throughput but not per-request percentile latencies
  (TTFT/ITL). For that, run `vllm serve <model>` as a server and drive it
  with `vllm bench serve` (or your own client) -- a natural next step once
  this lab has told you which precision is even worth serving.

## What to look for

- Memory should drop substantially from BF16 -> FP8 -> INT4, freeing up
  space that (in a real deployment) becomes more KV-cache blocks --
  i.e. more concurrent users at the same latency, which is the real
  business case for all of this, not just "the model got smaller".
- Throughput-per-request typically *increases* going from BF16 to FP8
  (Blackwell's FP8 tensor cores, lab 05) even before considering the
  memory/batching benefit; INT4's throughput story is more mixed since
  weight-only quantization still computes matmuls in bf16 after
  dequantizing (unless the batch is large enough that memory bandwidth,
  not compute, is the bottleneck -- watch how the throughput gap between
  precisions changes as concurrency increases).
- Perplexity differences between precisions here should roughly match the
  *relative* degradation you saw for each technique in its own lab, even
  though this is a different (larger) model -- that consistency is a good
  sanity check that nothing went wrong in this lab's own quantization
  setup.
