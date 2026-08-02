# Lab 06 - FP4 / NVFP4 / MXFP4 (Blackwell-Native 4-Bit Formats)

**Goal:** understand the microscaling trick that makes 4-bit floating point
usable at all, measure it from scratch, and then try the real
hardware-accelerated path your B200s support.

**Runtime:** ~15-25 minutes on one B200 (Qwen2.5-7B-Instruct); the
accelerated-path section needs `nvidia-modelopt` and/or `torchao` and may
take longer depending on what's installed.

## Background

Lab 00 introduced the math: a 4-bit float with 2 exponent + 1 mantissa bit
(`E2M1`) has exactly 8 non-negative representable values,
`{0, 0.5, 1, 1.5, 2, 3, 4, 6}`. That's a *brutally* coarse grid on its own
-- nowhere near enough to represent a full weight tensor's dynamic range.
The fix used by both **NVFP4** (NVIDIA's format) and **MXFP4** (the Open
Compute Project's "microscaling" spec, which NVFP4 is closely related to)
is to share one extra scale factor across a small **block** of elements
(16 for NVFP4, 32 for MXFP4), so the *local* dynamic range within each
block still gets the full 8-value grid, even though the *global* range
across the whole tensor can be huge. NVFP4 uses an FP8 (`E4M3`) block
scale (finer); MXFP4 uses a power-of-two-only (`E8M0`) block scale
(coarser, cheaper -- no mantissa bits to store).

This block-sharing is exactly what makes FP4 practical -- and exactly what
makes it fundamentally different from INT4 with the same group size:
INT4's uniform integer grid + a per-group scale gives 16 *evenly spaced*
levels per group, while FP4's floating grid + per-block scale gives 16
*unevenly spaced* levels (denser near zero) per block. Which one wins
depends on how peaked the underlying data is -- you measured this
directly for a synthetic tensor in lab 00; this lab measures it on a real
7B model.

B200 (Blackwell) GPUs have **native FP4 tensor cores**, so NVFP4/MXFP4
aren't just an extreme memory-saving trick -- like FP8 in lab 05, real FP4
compute is meaningfully faster than bf16 or even FP8 in hardware, when you
use a kernel that actually exploits it (this lab's from-scratch code does
not -- see the "accelerated path" section below for what does).

## What the script does

1. Loads Qwen2.5-7B-Instruct, evaluates BF16 baseline perplexity.
2. **From-scratch weight-only quantization sweep** at the same "4 bits"
   budget: `INT4 (per-group, g=32)` vs `NVFP4 (block=16)` vs
   `MXFP4 (block=32, power-of-two scale)`, using the
   `fp4_blockwise_quantize_dequantize` function from `common/quant_ops.py`
   (the same one lab 00 exercised on synthetic data) applied to every
   linear layer's weights.
3. **Accelerated path (best-effort)**: checks for `nvidia-modelopt`
   (TensorRT Model Optimizer) and, if present, runs its NVFP4 PTQ recipe on
   the same calibration data and evaluates perplexity; separately checks
   for `torchao`'s NVFP4/MXFP4 support. Whichever is available gets run and
   compared; if neither is installed, the script prints exactly what to
   `pip install` and skips this section rather than failing.
4. Pulls FP8 (lab 05) and RTN/GPTQ/AWQ (labs 01-03) numbers from their
   `metrics.json` files if those labs were already run, and charts each as
   **relative perplexity degradation** (`(ppl - baseline_ppl) / baseline_ppl`)
   rather than raw perplexity -- labs 01-03 use a smaller model than labs
   05-06, so raw perplexity values aren't directly comparable across them,
   but "how much worse than that model's own bf16 baseline" is.
5. Saves comparison charts + `metrics.json`.

## Run it

```bash
python labs/06_fp4_nvfp4/lab06_fp4_blackwell.py
```

Run labs 01, 02, 03, and 05 first (on the same model where possible) if you
want the combined "every precision so far" chart to be fully populated --
the script works fine without them, it just has fewer bars.

## What to look for

- NVFP4 (block=16, finer per-block scale) should beat MXFP4 (block=32,
  power-of-two scale) on perplexity -- smaller blocks adapt to local
  magnitude better, and a real (non-power-of-two) scale wastes less range.
- Whether NVFP4 beats or loses to INT4 at group=32 is genuinely
  model/layer dependent -- report what you actually observe rather than
  assuming one wins; this is exactly the kind of empirical check real
  quantization engineers do before picking a format for production.
- If you have `nvidia-modelopt` or `torchao`'s FP4 path installed: their
  perplexity should beat this lab's naive weight-only FP4 (no calibration,
  no error correction) by a similar margin to how GPTQ/AWQ beat plain RTN
  in labs 02-03 -- production 4-bit recipes combine microscaling *with*
  calibration-based error correction, not just a better number format.
- Keep this lab's numbers in mind for the lab 10 capstone, which serves an
  NVFP4 checkpoint (if you have the tooling) alongside BF16/FP8/INT4 across
  multiple GPUs and measures the actual throughput difference.
