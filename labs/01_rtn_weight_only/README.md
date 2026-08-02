# Lab 01 - Round-to-Nearest (RTN) Weight-Only PTQ

**Goal:** apply the simplest possible post-training quantization (PTQ)
method -- round every weight to the nearest grid point, independently,
with no calibration data at all -- to a real language model, and measure
the actual accuracy/memory trade-off. This is the baseline every fancier
method (GPTQ, AWQ, ...) in later labs tries to beat.

**Runtime:** ~5-10 minutes on one B200 (model download + several perplexity
evaluations on Qwen2.5-0.5B-Instruct).

## Background

RTN quantization is exactly the `fake_quantize_int` function from lab 00,
applied to every `nn.Linear` weight matrix in the model (skipping the
embedding and LM head, which are usually kept at higher precision since
vocabulary-sized matrices are unusually sensitive to quantization relative
to their size). No data is needed -- just the weight tensor itself.

Because it's data-free and needs no per-layer optimization, RTN is
essentially free to run, which makes it the natural first thing to try and
a useful baseline. Its weakness is exactly what you'd expect: it treats
every weight independently and has no way to account for which weights
matter most for the model's actual outputs (that's what GPTQ and AWQ add in
labs 02-03).

## What the script does

1. Loads Qwen2.5-0.5B-Instruct in bf16 and measures baseline WikiText-2
   perplexity + real GPU memory footprint.
2. For a sweep of `(bits, granularity)` configs --
   `{8, 4, 3} x {per_tensor, per_channel, per_group(g=128), per_group(g=32)}`
   -- quantizes every linear layer's weights in place with RTN, measures the
   new perplexity, then restores the original weights before moving to the
   next config (so every config starts from the same clean baseline).
3. Estimates the *theoretical* packed memory footprint for each config
   (bits per weight + the overhead of storing one scale per group) --
   the script itself keeps weights in bf16 the whole time (since PyTorch
   doesn't have a native int4 tensor type to just drop in), so this is
   a computed estimate of what a real packed-int4 kernel would use, not a
   measurement of this script's own memory use.
4. Runs a short generation from the model at a couple of the more aggressive
   configs, so you can *read* the degradation, not just see it as a number.
5. Saves a perplexity-vs-bits line chart (one line per granularity) and a
   memory-footprint bar chart to `results/`.

## Run it

```bash
python labs/01_rtn_weight_only/lab01_rtn.py
```

## What to look for

- Perplexity should barely move at 8-bit (any granularity), start
  noticeably degrading at 4-bit per-tensor, and recover most of that
  degradation at 4-bit per-group(32) -- granularity matters more as bits
  shrink, exactly as lab 00 predicted on synthetic data.
- 3-bit per-tensor is usually where RTN starts falling apart on a 0.5B
  model -- watch the generated text at that config for repeated tokens or
  incoherence.
- The memory chart should show group-wise quantization's overhead
  (one scale per group) eating into its savings vs per-channel/per-tensor
  as `group_size` shrinks -- there's a real cost to finer granularity, not
  just a benefit.
- Keep these numbers in mind for lab 02 (GPTQ) and lab 03 (AWQ) at the same
  bit-widths -- both should beat RTN's perplexity at 4-bit and especially at
  3-bit, since they use calibration data to decide *how* to quantize instead
  of just rounding.
