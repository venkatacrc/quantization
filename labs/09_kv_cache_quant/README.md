# Lab 09 - KV-Cache Quantization

**Goal:** quantize the *KV cache* (not the weights, not the activations
that flow through a linear layer -- the per-token key/value tensors saved
during autoregressive generation) and measure both the memory savings and
the accuracy cost, then compare a from-scratch implementation against
vLLM's native support.

**Runtime:** ~15-20 minutes on one B200 (Qwen2.5-7B-Instruct). The vLLM
comparison section needs `vllm` (already in your base environment).

## Background

During autoregressive generation, a transformer caches every previous
token's key and value projections per layer/head so it doesn't have to
recompute them at every new step -- that's the "KV cache". For long
contexts and/or large batches, this cache can dominate GPU memory more
than the model's *weights* do:

```
kv_cache_bytes = 2 (K and V) x num_layers x num_kv_heads x head_dim x seq_len x batch_size x bytes_per_element
```

Halving `bytes_per_element` (bf16 -> int8, or bf16 -> fp8) directly halves
that memory, which either lets you serve longer contexts, larger batches,
or both, with the same GPU memory. Unlike weight quantization, this cost
is paid *fresh, every request* (the cache is per-conversation, not a
one-time offline step), so KV-cache quantization has to be cheap enough to
do at generation time, not just accurate.

### Why this is a slightly different problem than weight quantization

Keys and values are activations, computed on the fly from the current
context -- there's no "calibrate once offline" step like GPTQ/AWQ get to
do. The standard approach (used by vLLM, TensorRT-LLM, and this lab) is
**per-token dynamic quantization**: whenever a new token's key/value is
computed, quantize it (with a scale derived from that token's own values)
before writing it into the cache, then dequantize when reading it back out
for attention. This lab implements exactly that, using the same
`fake_quantize_int` / `fp8_quantize_dequantize` primitives from lab 00,
applied per-token by subclassing `transformers`' `DynamicCache`.

## What the script does

1. Computes and charts the **theoretical KV-cache memory** for
   Qwen2.5-7B-Instruct at a few context lengths (4K/32K/128K tokens),
   comparing bf16 vs INT8 vs FP8 cache storage -- purely arithmetic, no
   GPU needed, but it's the clearest way to see why this matters *before*
   context length gets long.
2. Implements `QuantizedKVCache`, a small subclass of `transformers`'
   `DynamicCache` that quantizes every key/value tensor (INT8 or FP8,
   per-token dynamic scale) the moment it's written to the cache, then
   dequantizes it back before returning it for attention -- so the
   *rest* of the model is completely unaware anything is quantized.
3. Re-runs this lab's own version of the sliding-window WikiText-2
   perplexity evaluation (from `common/eval.py`), but this time passing
   the quantized cache explicitly, so every window's self-attention
   actually reads back quantized-and-dequantized keys/values -- a genuine
   test of the accuracy cost, not just a memory estimate.
4. Compares BF16 cache (baseline), INT8 cache, and FP8 (E4M3) cache.
5. **Production comparison**: launches vLLM twice (`kv_cache_dtype="auto"`
   vs `kv_cache_dtype="fp8"`) on the same model, and compares actual
   reported GPU memory usage and generation throughput.
6. Saves comparison charts + `metrics.json`.

## Run it

```bash
python labs/09_kv_cache_quant/lab09_kv_cache_quant.py
```

## What to look for

- The theoretical memory chart should make it obvious that at long enough
  context, the KV cache -- not the weights -- becomes the memory
  bottleneck; this is exactly why vLLM/TensorRT-LLM support KV-cache
  quantization as a first-class serving option, independent of whatever
  precision the weights are in.
- INT8 KV-cache perplexity degradation should be small but non-zero;
  FP8 should be even closer to the bf16 baseline, for the same reason FP8
  activations needed no smoothing in lab 05 -- keys/values have outlier
  structure too, and FP8's non-uniform grid absorbs it better than INT8's
  uniform one.
- The vLLM comparison should show a measurable memory reduction with
  `kv_cache_dtype="fp8"` at the same `gpu_memory_utilization` setting --
  in production this memory gets reinvested as more available KV cache
  blocks, i.e. more concurrent requests or longer contexts, not just a
  smaller footprint for its own sake.
