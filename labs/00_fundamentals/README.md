# Lab 00 - Quantization Numerics Fundamentals

**Goal:** build intuition for what "quantization" actually does at the bit
level, before touching a real model. Every later lab is a variation on the
ideas here.

**Runtime:** a minute or two. CPU is fine; a GPU (if present) is used
automatically but not required.

## Background

### What is quantization?

Neural network weights and activations are normally stored as 16- or 32-bit
floating point numbers. Quantization maps those values onto a much smaller
set of representable numbers (e.g. 256 integer levels for 8-bit, 16 levels
for 4-bit) so that a tensor takes less memory and, on hardware with native
low-precision arithmetic, computes faster. The core trade-off is always:
**fewer bits -> less memory & faster compute, but more rounding error.**

### Uniform integer quantization: scale + zero-point

For `b` bits, we pick an integer grid (e.g. `[-127, 127]` for 8-bit
symmetric) and a real-valued **scale** `s` (and, for asymmetric
quantization, a **zero-point** `z`) so that:

```
quantize:    q = round(x / s + z),  clamped to the integer grid
dequantize:  x_hat = (q - z) * s
```

- **Symmetric** quantization forces `z = 0` -- simpler and what most
  weight quantizers use, at the cost of wasting one grid code if the data
  isn't centered at zero.
- **Asymmetric** quantization fits `s` and `z` to the tensor's actual
  `[min, max]`, giving a tighter fit for skewed distributions (e.g.
  post-ReLU activations, which are >= 0).

### Granularity: per-tensor vs per-channel vs per-group

A single `(s, z)` for an entire weight matrix (**per-tensor**) is cheap to
store but forces every row to share one dynamic range -- if one output
channel has much larger weights than the rest, it dominates the scale and
everyone else loses precision. **Per-channel** quantization gives every row
of a weight matrix (e.g. one scale per output feature) its own `(s, z)`.
**Per-group** goes further and gives every contiguous chunk of `group_size`
elements *within* a row its own scale -- this is what GPTQ/AWQ/QLoRA use in
practice (typical `group_size` is 32, 64, or 128), because even within one
output channel the input dimension can have very different local scales.
Finer granularity = less error, but more scale factors to store (and every
stored scale eats into the memory savings you're trying to get).

### Floating-point quantization: FP8 and FP4

Instead of a uniform integer grid, you can quantize onto a *floating-point*
grid, which spaces its representable values non-uniformly (finer resolution
near zero, coarser further out) -- often a better match for weight/activation
distributions, which are usually peaked near zero with occasional outliers.

- **FP8** comes in two flavors: `E4M3` (4 exponent bits, 3 mantissa bits,
  max magnitude ~448) trades dynamic range for precision, while `E5M2` (5
  exponent bits, 2 mantissa) trades precision for dynamic range (useful for
  gradients/activations that can spike). NVIDIA Blackwell GPUs (including
  your B200s) have **native FP8 tensor cores**, so FP8 isn't just a
  memory-saving trick here -- it's also a real speedup (see lab 05).
- **FP4** has only 16 representable magnitudes total. With 1 sign + 2
  exponent + 1 mantissa bit (`E2M1`), the non-negative grid is exactly
  `{0, 0.5, 1, 1.5, 2, 3, 4, 6}`. To keep this usable, FP4 formats share one
  extra scale factor across a small **block** of elements (16 for NVIDIA's
  NVFP4, 32 for the OCP "MX" spec's MXFP4) -- this is called **microscaling**.
  NVFP4 uses an FP8 (`E4M3`) block scale for more precision; MXFP4 uses a
  power-of-two scale (`E8M0`, i.e. just an exponent, no mantissa) which is
  coarser but even cheaper. B200s have native FP4 tensor cores too -- see
  lab 06.

## What the script does

`lab00_numerics.py`:

1. Builds a synthetic weight-like tensor (Gaussian body + a few outlier
   rows, mimicking the heavy-tailed per-channel statistics real LLM weights
   show) and, if a GPU + internet are available, also pulls one real
   attention projection weight matrix from Qwen2.5-0.5B-Instruct for a
   sanity check that the synthetic tensor isn't misleading.
2. Sweeps bit-width `{8, 4, 3, 2}` x `{symmetric, asymmetric}` at
   per-tensor granularity and records reconstruction error.
3. Sweeps granularity `{per_tensor, per_channel, per_group(32/64/128)}` at
   INT4 and records error.
4. Compares INT8 vs FP8 (E4M3, E5M2) at the same "8 bits" budget.
5. Compares INT4 vs FP4 (NVFP4-style block=16, MXFP4-style block=32 with a
   power-of-two scale) at the same "4 bits" budget.
6. Saves bar charts + an error histogram + a `metrics.json` summary into
   `results/`.

## Run it

```bash
python labs/00_fundamentals/lab00_numerics.py
```

## What to look for

- Error should drop sharply from 8->4->3->2 bits, and roughly follow
  `error ~ 2^-bits` for uniform quantization of a roughly-Gaussian tensor.
- Asymmetric quantization should look *better than or equal to* symmetric
  at every bit-width here, because the synthetic tensor isn't perfectly
  zero-centered.
- Per-group should beat per-channel should beat per-tensor at INT4 -- but by
  a shrinking margin as `group_size` grows, since a smaller group means a
  scale factor that fits the local data better.
- FP8 (`E4M3`) should have lower error than INT8 on the outlier rows
  specifically (check the printed per-row breakdown) even though both use 8
  bits, because the floating grid adapts its spacing to magnitude.
- FP4 should be *much* worse than INT4 per-element on average (only 16
  total grid points vs 16 *integer* points is the same count, but FP4's
  grid is non-uniform and its block scale is shared over many more values
  than a typical per-group INT4 setup) -- this is exactly why lab 06 spends
  time on how real NVFP4/MXFP4 recipes claw that accuracy back.
