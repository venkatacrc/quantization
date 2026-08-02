# Lab 07 - Quantization-Aware Training (QAT) with a Straight-Through Estimator

**Goal:** instead of quantizing an already-trained model and hoping it
still works (PTQ -- everything in labs 01-06), fine-tune the model *while*
simulating quantization in the forward pass, so its weights learn to be
robust to the rounding they'll experience at inference time.

**Runtime:** ~10-15 minutes on one B200 (Qwen2.5-0.5B-Instruct; small on
purpose, since QAT needs a training loop, not just a forward pass).

## Background

Every PTQ method so far (RTN, GPTQ, AWQ, SmoothQuant) starts from trained
full-precision weights and tries to quantize them with minimum damage.
QAT instead keeps training the model with the quantization operation
*inside* the forward pass, so gradient descent can push the underlying
full-precision weights toward values that happen to round well -- not just
values that were good in full precision.

The obstacle: `round()` (which every quantizer in this repo uses) has zero
gradient almost everywhere and an undefined one at the rounding
boundaries. If you literally backprop through `round()`, no weight ever
updates. The fix, used since the earliest quantized neural network work,
is the **straight-through estimator (STE)**: in the forward pass, actually
quantize (round, clamp, dequantize -- exactly the `fake_quantize_int` from
lab 00/01); in the backward pass, *pretend the quantization step was the
identity function* and let the gradient flow straight through to the
underlying full-precision weight, unchanged. It's a biased gradient
estimator (the true function's gradient is zero almost everywhere, and STE
uses 1 instead), but empirically it works well enough that it's still the
standard approach.

```
forward:   w_q = dequantize(quantize(w))         # real rounding happens here
backward:  d(loss)/dw  ~=  d(loss)/d(w_q)          # STE: treat quantize+dequantize as identity
```

This lab implements that directly: `common/quant_ops.py`'s
`ste_fake_quantize` is a tiny `torch.autograd.Function` whose `forward`
calls the exact same `fake_quantize_int` used everywhere else in this
repo, and whose `backward` just returns the incoming gradient unchanged.

## What the script does

1. Loads Qwen2.5-0.5B-Instruct in full precision and evaluates a baseline
   perplexity (before any quantization).
2. Wraps every target linear layer in a `QATLinear` module that keeps a
   full-precision trainable weight but always runs it through
   `ste_fake_quantize` (INT4, per-channel) before the matmul -- so from
   this point on, the model always "sees" the quantization at both
   training and eval time.
3. Immediately evaluates perplexity in this state, *before any training
   step* -- this is exactly a PTQ/RTN baseline at the same bit-width,
   computed for free as a side effect of the wrapper.
4. Fine-tunes for a few hundred steps on WikiText-2 (next-token loss, small
   batches, low learning rate), periodically re-evaluating perplexity on a
   quick subset so you get a training curve, not just a before/after
   number.
5. Evaluates final perplexity on the full WikiText-2 test set.
6. Saves a loss curve, a perplexity-over-training-steps curve, and a
   before/after bar chart.

## Run it

```bash
python labs/07_qat/lab07_qat_lsq.py
```

## What to look for

- The "before training" perplexity (step 0) should look like a typical
  aggressive PTQ result -- noticeably worse than full precision, since
  INT4 per-channel with no calibration data is one of the harsher configs
  this curriculum has tried.
- Perplexity should visibly recover over the first ~100-200 steps as the
  underlying full-precision weights adjust to round more favorably, then
  plateau -- QAT doesn't need nearly as many steps as original
  pretraining, since it's fine-tuning around an already-good minimum.
- The loss curve should be noisier early on (large rounding jumps from a
  weight distribution the model wasn't trained for) and smooth out as
  training progresses.
- This is real compute you're spending (a training loop, with an
  optimizer) to recover accuracy, unlike calibration-only methods (GPTQ,
  AWQ) which are a single forward-pass-worth of work -- QAT is the
  "spend more to get more" point on this curriculum's cost/accuracy
  spectrum. Compare final perplexity here against lab 02's GPTQ number on
  a similarly-sized model to see whether the extra training cost is worth
  it for your use case.
