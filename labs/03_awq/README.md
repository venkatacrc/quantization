# Lab 03 - AWQ (Activation-Aware Weight Quantization)

**Goal:** implement AWQ's core insight from scratch -- that a small number
of *input channels* (not weights themselves) drive most of the
quantization error, and that you can protect them with a per-channel
rescaling instead of mixed precision -- then compare against a production
AWQ implementation.

**Runtime:** ~10-15 minutes on one B200 (Qwen2.5-1.5B-Instruct, same model
as lab 02 so the RTN/GPTQ/AWQ comparison is apples-to-apples).

## Background

GPTQ (lab 02) reduces quantization error by adjusting *weights* after the
fact, using second-order information. AWQ ("AWQ: Activation-aware Weight
Quantization for LLM Compression and Acceleration", Lin et al. 2023) takes
a different angle: it observes that a handful of **input channels** have
much larger activation magnitudes than the rest (this is the same
activation-outlier phenomenon SmoothQuant, lab 04, also exploits), and that
the weight columns multiplying those channels are disproportionately
important to get right -- *even though the weights themselves aren't
unusually large*.

The key trick is a mathematical identity that costs nothing at inference
time (once fused into the previous layer): for any per-input-channel scale
vector `s`,

```
W x = (W * s) * (x / s)         # exactly equal, for any channel-wise s > 0
```

If we pick `s_c > 1` for a salient channel `c`, the corresponding weight
column `W[:, c]` gets multiplied up before quantization -- effectively
giving it more "room" on the quantization grid relative to its rounding
error, without changing the true mathematical output at all (since the
matching `x / s` cancels it back out). The catch: scaling one column up
also raises that *row's* max value, which can widen the shared
scale/zero-point for the *other* columns in a per-channel/per-group
quantizer, making their error slightly worse. So there's a real trade-off,
and AWQ picks `s` by a small grid search that balances "protect the salient
channels" against "don't hurt everyone else":

```
s_c(alpha) = act_scale_c ^ alpha / weight_scale_c ^ (1 - alpha)
```

where `act_scale_c` is channel `c`'s average calibration activation
magnitude and `weight_scale_c` is channel `c`'s average weight magnitude.
`alpha` is searched over a small grid (this lab uses 9 points from 0 to 1)
and picked by actually measuring reconstruction error on calibration data
for each candidate.

### A simplification this lab makes, on purpose

In production, the `x / s` division is *fused* into whatever produced `x`
(usually the preceding LayerNorm's weight, or the previous linear's output
rows) so there's zero extra runtime cost. Implementing that fusion requires
walking the specific pairs of adjacent modules in each transformer block
(attention QKV comes after an RMSNorm, `down_proj` comes after an
activation function applied to `gate_proj`/`up_proj`, etc.) -- useful
production engineering, but it doesn't teach anything new about *why* AWQ
works. This lab instead wraps each quantized linear layer in a small module
that explicitly divides its input by `s` before the matmul. This is
numerically identical to the fused version (same weights, same effective
math) -- just with one extra elementwise op per forward pass instead of a
free one, which is the right trade to make for a learning lab.

## What the scripts do

`lab03_awq_from_scratch.py`:

1. Loads Qwen2.5-1.5B-Instruct, evaluates BF16 baseline perplexity (should
   match lab 02's baseline).
2. Runs calibration data through the model once, recording each target
   layer's mean per-channel activation magnitude and a small cached batch
   of raw activations (for the scale search's reconstruction-error check).
3. For each layer: searches `alpha` in `{0, 0.125, ..., 1.0}`, and for each
   candidate computes the scaled+quantized weight and measures output
   reconstruction error directly (`||Wx - Ŵ'x'||`) on the cached
   calibration batch; keeps the best `alpha`.
4. Replaces the layer with a small wrapper that applies the input scaling
   + holds the scaled/quantized weight, at the same INT4/group_size=128
   budget used in labs 01/02.
5. Evaluates perplexity, prints the picked `alpha` per layer (salient
   layers should end up wanting `alpha` closer to 1), and saves a
   comparison chart against lab 02's RTN/GPTQ numbers.

`lab03b_compare_production_awq.py` mirrors lab 02b: quantizes the same
model with `autoawq` (falling back to `llmcompressor`'s `AWQModifier`) and
adds that number to the same comparison chart.

## Run it

```bash
python labs/03_awq/lab03_awq_from_scratch.py
python labs/03_awq/lab03b_compare_production_awq.py   # needs autoawq or llmcompressor
```

## What to look for

- From-scratch AWQ should beat RTN and land close to (sometimes better
  than, on this simplified non-sequential setup) from-scratch GPTQ from
  lab 02, despite being conceptually much simpler -- AWQ protects the
  *few* channels that matter most instead of trying to optimize
  every weight.
- Printed per-layer best `alpha` values: attention output projections and
  MLP `down_proj` layers (which see the widest activation distributions)
  usually pick larger `alpha` than QKV projections.
- The production AWQ run should be noticeably *faster* to quantize than
  production GPTQ (lab 02b) for the same reason it's simpler to implement
  here: no Hessian, no Cholesky, just a bounded grid search per layer.
