# Lab 04 - SmoothQuant (W8A8: Weight *and* Activation Quantization)

**Goal:** go from weight-only quantization (labs 01-03) to quantizing
**activations too**, which is what actually lets hardware run an INT8 x
INT8 matmul instead of dequantizing back to bf16 first -- and see the real
GPU speedup that unlocks, not just a memory number.

**Runtime:** ~15-20 minutes on one B200 (Qwen2.5-3B-Instruct, larger than
labs 01-03 so the GEMM benchmark has representative matrix sizes).

## Background

Every previous lab quantized *weights* but still computed in bf16: the
quantized weight gets dequantized back to bf16 before the matmul, so you
save memory (loading weights) but not compute -- the GPU still does a bf16
GEMM. To get an actual compute speedup on hardware with fast low-precision
tensor cores, **both** operands of the matmul need to be low-precision:
this is what "W8A8" (8-bit weights *and* 8-bit activations) means.

The problem: activations in transformer LLMs have large per-channel
outliers (some input feature channels are consistently 10-100x larger than
the rest, across almost every token) -- much worse outliers than the
weights have. A naive per-tensor INT8 quantizer for activations gets
dominated by those few channels and wrecks precision for everyone else.

SmoothQuant ("SmoothQuant: Accurate and Efficient Post-Training
Quantization for Large Language Models", Xiao et al. 2022) fixes this with
the *same* per-channel-scale identity AWQ uses (lab 03), but applied with a
different goal: instead of protecting salient weight channels, it
deliberately **migrates difficulty from activations to weights** --
weights are much easier to quantize precisely (no outliers, and they're
static so you can compute an exact per-channel scale offline), so moving
the outlier problem onto them is a good trade:

```
s_c = act_scale_c ^ alpha / weight_scale_c ^ (1 - alpha)      (alpha = 0.5 by default)

W' = W * s        (folded into the weight, offline, one time)
X' = X / s        (folded into the previous layer at deployment; explicit here, see below)
```

Higher `alpha` pushes more of the outlier magnitude onto the weights
(smoother activations, lumpier weights); lower `alpha` does the opposite.
`alpha = 0.5` is the paper's default and works well across most models;
this lab sweeps a few values so you can see the accuracy trade-off
directly instead of taking that on faith.

After smoothing, both `W'` and `X'` are quantized to INT8:

- **Weights**: per-output-channel (row) scale, computed once from the
  (smoothed) weight tensor itself -- call this **static weight
  quantization**, since it needs no runtime data at all.
- **Activations**: this lab uses **static, per-tensor** quantization -- one
  scale, calibrated ahead of time from the smoothed activations' observed
  max magnitude -- which is what actually enables a real INT8 x INT8 GEMM
  with no runtime scale computation. (Some deployed systems instead use
  **dynamic** per-token activation quantization -- computing each batch's
  scale on the fly -- which is usually more accurate but needs an extra
  reduction op at runtime; this lab's script prints the dynamic-quant
  accuracy as a reference point without implementing its own fused kernel.)

### The same simplification as lab 03

As in AWQ, `X' = X / s` is normally fused into the previous layer for free.
This lab again applies it as an explicit elementwise division in a wrapper
module -- numerically identical, just not fused for maximum speed.

## What the script does

1. Loads Qwen2.5-3B-Instruct, evaluates BF16 baseline perplexity.
2. Runs calibration data through the model, recording per-channel
   activation scales (for smoothing) and the smoothed activations' max
   magnitude (for static INT8 calibration).
3. Sweeps `alpha in {0.3, 0.5, 0.7}` (quick perplexity check on a handful of
   windows), picks the best, then does the full smoothing + INT8
   weight+activation quantization at that `alpha` and replaces every target
   linear layer with a module that does a real `torch._int_mm` INT8 x INT8
   matmul on GPU.
4. Evaluates full perplexity of the resulting W8A8 model.
5. **GEMM microbenchmark**: times a raw `torch._int_mm` INT8 matmul against
   an equivalent-shape bf16 matmul, at this model's actual layer shapes, to
   measure the real hardware speedup from int8 tensor cores directly (the
   cleanest signal for "is INT8 compute actually faster on this GPU",
   independent of any Python-level overhead in the whole-model wrapper).
6. **End-to-end benchmark**: times text generation with the full W8A8 model
   vs BF16, as a rough sanity check (see the caveat below).
7. Saves comparison charts + `metrics.json`.

## Run it

```bash
python labs/04_smoothquant/lab04_smoothquant.py
```

## What to look for

- The perplexity gap between BF16 and W8A8-SmoothQuant should be small
  (typically well under 1 point on a 3B-class model) -- the entire point of
  the smoothing step is making 8-bit activations viable in the first place;
  compare against what happens if you skip smoothing (`alpha = 0`, printed
  as a reference row) to see how much it's actually buying you.
- The GEMM microbenchmark should show a clear speedup for `torch._int_mm`
  over bf16 at large matrix shapes -- this is the real payoff of W8A8 that
  weight-only methods (labs 01-03) can't give you, since B200 has native
  INT8 tensor cores at roughly double the bf16 throughput.
- **Caveat**: the end-to-end generation benchmark uses this lab's simple,
  unfused per-layer wrapper (quantize activation -> `torch._int_mm` ->
  dequantize, one Python call per linear layer), so its wall-clock number
  usually shows *less* speedup than the GEMM microbenchmark, or even a
  slowdown at small batch sizes -- a production W8A8 kernel (as used by
  vLLM/TensorRT-LLM) fuses all of that into one kernel per layer. Treat the
  GEMM microbenchmark as "what the hardware can do" and the end-to-end
  number as "what this teaching implementation does with it".
