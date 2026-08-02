# Lab 05 - FP8 Quantization (Native Blackwell Tensor Cores)

**Goal:** quantize both weights and activations to FP8 and run a *real* FP8
matmul on your B200s, comparing E4M3 vs E5M2 and per-tensor vs per-row
scaling -- and see why FP8 needs much less special handling than INT8
(lab 04) did.

**Runtime:** ~15-20 minutes on one B200 (Qwen2.5-7B-Instruct).

## Background

Lab 00 introduced FP8's two flavors: `E4M3` (more mantissa bits, less
range, max magnitude ~448) and `E5M2` (more range, less precision, max
magnitude ~57344). NVIDIA Hopper and Blackwell GPUs (your B200s) have
**native FP8 tensor cores** -- an FP8 x FP8 matmul isn't just a memory
trick, it runs at roughly double the throughput of an equivalent bf16
matmul in hardware. PyTorch exposes this directly via `torch._scaled_mm`,
which this lab calls straight from Python (no custom CUDA needed).

### Why FP8 doesn't need SmoothQuant-style smoothing

Lab 04 showed that naive INT8 activation quantization (no smoothing) badly
hurts accuracy, because a uniform integer grid gets dominated by a few
outlier channels. FP8's *floating-point* grid is non-uniform by
construction -- it naturally allocates more resolution near zero and less
far from zero, which is a much better match for the "mostly small values,
occasional large outlier" shape real activations have. This lab quantizes
activations to FP8 **directly, per-tensor, dynamically** (scale computed
fresh every forward pass, no calibration, no smoothing) and you'll see
perplexity stay close to the bf16 baseline anyway -- a direct, hands-on
illustration of why FP8 has become the default "just quantize it" format
for LLM inference, while INT8 needs much more care.

## What the scripts do

`lab05_fp8_inference.py`:

1. Loads Qwen2.5-7B-Instruct, evaluates BF16 baseline perplexity + memory.
2. **Weight-only FP8** sweep (fake-quant, like lab 01's RTN sweep but on
   the FP8 grid instead of an integer grid): `{E4M3, E5M2} x {per-tensor,
   per-row}`, measuring perplexity for each.
3. **GEMM microbenchmark**: times a real `torch._scaled_mm` FP8 x FP8
   matmul against an equivalent bf16 matmul at this model's actual layer
   shapes and a few batch sizes -- the direct hardware speedup signal.
4. **W8A8-FP8 end-to-end**: replaces every target linear layer with a
   module that dynamically quantizes its input activations to FP8 E4M3
   (per-tensor, computed fresh each forward call -- no smoothing, no
   calibration) and its weights to FP8 E4M3 (static, per-row), then
   actually calls `torch._scaled_mm`. Evaluates perplexity and benchmarks
   generation throughput/memory against the BF16 baseline.
5. Saves comparison charts + `metrics.json`.

`lab05b_compare_torchao.py` (needs `torchao`): reproduces the W8A8-FP8
setup with `torchao`'s `Float8DynamicActivationFloat8WeightConfig`
(a fused, production-grade implementation) and compares perplexity +
throughput against this lab's from-scratch version.

## Run it

```bash
python labs/05_fp8/lab05_fp8_inference.py
python labs/05_fp8/lab05b_compare_torchao.py   # needs torchao
```

## What to look for

- Weight-only FP8 should barely move perplexity at either E4M3 or E5M2 --
  7B-class models have plenty of headroom at 8 bits regardless of grid
  shape; the interesting comparison here is INT8 vs FP8 at the *activation*
  side (next point), not weights.
- W8A8-FP8 (no smoothing) perplexity should land close to lab 04's
  *smoothed* SmoothQuant INT8 result, and clearly beat lab 04's
  *unsmoothed* INT8 reference row -- direct evidence that FP8's grid
  shape is doing SmoothQuant's job for free.
- The GEMM microbenchmark should show FP8 meaningfully faster than bf16 at
  large token counts, and (if you ran lab 04) roughly comparable to or
  faster than the INT8 `torch._int_mm` numbers -- Blackwell's FP8 tensor
  cores are a first-class citizen, not a fallback path.
- `torchao`'s fused implementation should show a bigger end-to-end
  generation speedup than this lab's unfused per-layer version, for the
  same reason noted in lab 04: production kernels fuse the
  quantize/matmul/dequantize sequence into one kernel launch.
