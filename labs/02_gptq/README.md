# Lab 02 - GPTQ (Hessian-Based Layer-Wise Reconstruction)

**Goal:** implement the core GPTQ algorithm from scratch and see *why* it
beats RTN at the same bit-width -- then reproduce the same idea with a
production library and compare.

**Runtime:** ~15-25 minutes on one B200 (calibration forward passes +
per-layer quantization + two perplexity evaluations on Qwen2.5-1.5B-Instruct).
Lab 02b (production comparison) needs `gptqmodel` or `llmcompressor`
installed (`pip install -r requirements-extra.txt`).

## Background

RTN (lab 01) quantizes every weight independently -- it has no way to know
that rounding one weight can be partly "corrected for" by adjusting the
*other, not-yet-quantized* weights in the same row. GPTQ ("GPTQ: Accurate
Post-Training Quantization for Generative Pre-trained Transformers", Frantar
et al. 2022) fixes this with a layer-wise reconstruction: for each linear
layer, quantize its weight matrix one column (input feature) at a time,
and after quantizing each column, adjust every *remaining* column to
compensate for the error just introduced -- using second-order (Hessian)
information from real calibration data to know exactly how much to adjust.

Concretely, for one layer with weight `W` (out_features x in_features) and
calibration input activations `X` (n_tokens x in_features), we want to
minimize the squared output error `||W Xᵀ - Ŵ Xᵀ||²`. Because that loss is
quadratic in `Ŵ`, its Hessian w.r.t. any one row of `W` is
`H = 2 XᵀX` (the same for every row, since the loss decomposes over rows).
GPTQ:

1. Adds a small damping term to `H` for numerical stability, then computes
   `H⁻¹` via Cholesky decomposition (twice: once to invert, once more to get
   the Cholesky factor of the inverse -- this specific two-step trick is
   what makes the algorithm numerically stable enough to run on full-size
   LLM layers, per the paper's appendix).
2. Processes columns left to right. For column `i`: quantize it (RTN, but
   using the group's live/current values -- see below), then propagate the
   resulting error into every later column `j > i`, weighted by
   `H⁻¹[i, j] / H⁻¹[i, i]`. This is the same "compensate the neighbors"
   update as classical Optimal Brain Quantization (OBQ, GPTQ's predecessor),
   but GPTQ's fixed column order + Cholesky trick makes it run in
   `O(d_in²)` instead of OBQ's `O(d_in³)`.
3. Re-derives the group's scale/zero-point every `group_size` columns,
   from the (partially-corrected) live weight values at that point -- not
   from the original weights -- so later groups benefit from the error
   correction already applied to them.

### A simplification this lab makes, on purpose

Real GPTQ implementations (including the ones in lab 02b) quantize
transformer blocks **sequentially**: to build the calibration activations
for block `N`, they run the calibration data through blocks `0..N-1` *after
those blocks have already been quantized*, so each layer's Hessian reflects
the actual (quantized) activations it will see at inference time. That's
important for best-possible accuracy, but it roughly doubles the code
you need (you have to interleave "run calibration forward pass one block at
a time" with "quantize this block's linears") without teaching anything new
about the *math* of GPTQ.

This lab instead captures calibration activations for **every** linear
layer in a single forward pass through the original, unquantized model, and
quantizes every layer independently from those activations. This is
sometimes called "non-sequential" GPTQ. It's slightly less accurate than
the real thing (errors from early layers aren't accounted for when
quantizing later layers) but isolates exactly the Hessian-based
reconstruction math above. Lab 02b's production-library run uses the real
sequential algorithm, so the gap you measure between the two is a decent
proxy for "how much does sequential error-correction actually buy you".

## What the scripts do

`lab02_gptq_from_scratch.py`:

1. Loads Qwen2.5-1.5B-Instruct, evaluates BF16 baseline perplexity.
2. Runs 128 calibration sequences (512 tokens each, from WikiText-2) through
   the model with forward hooks on every target linear layer, accumulating
   each layer's Hessian `H = XᵀX` on the fly (never materializing the full
   activation tensor, just the running `in_features x in_features` sum).
3. For each layer, runs the from-scratch GPTQ column-by-column
   reconstruction described above at INT4/group_size=128 (same bit budget
   as one of lab 01's RTN configs), replacing the layer's weight in place.
4. Also computes plain RTN at the same bit budget for a direct, same-script
   comparison.
5. Evaluates perplexity for both, prints per-layer reconstruction error
   (`||W - Ŵ||_F / ||W||_F`) so you can see which layers are hardest to
   quantize, and saves a comparison chart + `metrics.json`.

`lab02b_compare_production_gptq.py`:

1. Checks for `gptqmodel` (preferred) or falls back to describing what to
   install if neither is present.
2. Quantizes the same model to the same INT4/group_size=128 config using
   the real, sequential production algorithm, on the same calibration data.
3. Evaluates perplexity of the resulting checkpoint and reports wall-clock
   quantization time, appending its numbers onto lab02's `metrics.json` /
   comparison chart.

## Run it

```bash
python labs/02_gptq/lab02_gptq_from_scratch.py
# then, if you have gptqmodel or llmcompressor installed:
python labs/02_gptq/lab02b_compare_production_gptq.py
```

## What to look for

- From-scratch GPTQ should land between RTN and the production library on
  perplexity at INT4/g128 -- better than RTN (it's using calibration data
  and the Hessian), but not quite matching the production sequential
  version.
- Per-layer error should be noticeably worse for a handful of layers
  (often later MLP `down_proj` layers) -- this is the same
  "some layers are much more sensitive than others" pattern that motivates
  mixed-precision / outlier-aware methods like AWQ (lab 03).
- The production run's wall-clock time is usually *longer* than the
  from-scratch version despite being more optimized code, precisely because
  it pays for the sequential recalibration this lab skips.
