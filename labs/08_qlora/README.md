# Lab 08 - QLoRA (NF4 4-Bit Base + LoRA Fine-Tuning)

**Goal:** see how quantization enables fine-tuning models that wouldn't
otherwise fit in memory for training at all -- by freezing a 4-bit base
model and only training a small set of full-precision adapter weights on
top of it.

**Runtime:** ~15-20 minutes on one B200 (Qwen2.5-7B-Instruct). Needs
`bitsandbytes` (`pip install -r requirements-extra.txt`).

## Background

Labs 01-07 were all about *inference*: quantize a trained model, run it
faster/smaller, measure the accuracy cost. QLoRA ("QLoRA: Efficient
Finetuning of Quantized LLMs", Dettmers et al. 2023) uses quantization to
make *training* cheaper instead: it freezes the entire base model in 4-bit
and only trains a small number of extra parameters (LoRA adapters -- low
rank matrices inserted alongside each frozen linear layer), so training
memory is dominated by the tiny frozen base model plus the adapters'
gradients/optimizer state, not the full model's.

Two things specific to QLoRA, beyond "just" 4-bit weights:

- **NF4 ("NormalFloat4")**: instead of a uniform integer grid (INT4, labs
  01-03) or a floating-point exponent/mantissa grid (FP4, lab 06), NF4
  picks its 16 representable values as **quantiles of a standard normal
  distribution** -- i.e. the grid points that would be information-optimal
  if the data were exactly Gaussian. Real neural network weights are
  *approximately* Gaussian, so this tends to beat both INT4 and FP4 at the
  same bit-width for weight-only quantization specifically (this lab
  verifies that on a real weight tensor before doing anything else).
- **LoRA on top of a frozen quantized base**: `peft`'s
  `prepare_model_for_kbit_training` + `LoraConfig` inserts trainable
  low-rank `A`/`B` matrices (rank 16 by default here) next to each target
  linear layer; the frozen 4-bit base weight `W` and the LoRA update
  `BA` are summed at forward time (`y = Wx + BAx`), so only `A`/`B` (and
  their optimizer state) need full-precision memory and gradients.

## What the script does

1. Quick from-scratch check: quantizes one real weight tensor from
   Qwen2.5-7B-Instruct to INT4 (per-group, g=64), NF4 (block=64, matching
   bitsandbytes' default), and NVFP4 (block=16, its own native block size)
   and reports reconstruction error for each -- a direct, real-model
   version of lab 00's format comparison.
2. Loads the full model with `transformers`' `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)`
   and measures actual GPU memory used, compared against an estimate of
   what loading the same model in bf16 would take.
3. Wraps the model with `peft`'s `prepare_model_for_kbit_training` +
   LoRA adapters (rank 16, targeting every attention/MLP projection),
   reports how many parameters are actually trainable (should be well
   under 1% of the total).
4. Fine-tunes on a small slice of the Alpaca instruction dataset (a few
   hundred steps, small batches) with a plain PyTorch training loop,
   tracking the loss curve and peak GPU memory during training.
5. Compares peak training memory actually observed against a rough
   estimate of what full-precision full fine-tuning (weights + gradients +
   Adam optimizer state, no quantization, no LoRA) would require for the
   same model.
6. Generates a couple of instruction-following completions before/after
   fine-tuning so you can see the adapter's effect qualitatively, not just
   as a loss number.
7. Saves comparison charts + `metrics.json`.

## Run it

```bash
python labs/08_qlora/lab08_qlora_finetune.py
```

## What to look for

- NF4 should show the lowest reconstruction error of the three 4-bit
  formats on a real weight tensor (step 1) -- this is the concrete payoff
  of designing a codebook around the actual data distribution instead of
  a generic uniform/floating-point grid.
- The measured 4-bit memory footprint should be roughly a quarter of the
  bf16 estimate (4 bits vs 16, plus a little overhead for
  double-quantized scale factors) -- and the *trainable* parameter count
  should be tiny (LoRA rank 16 on a 7B model is typically well under
  1% of total parameters).
- Peak training memory should be dramatically lower than the full
  fine-tuning estimate -- this is the whole point of QLoRA: it's not just
  "a smaller model", it's "fine-tuning became possible on hardware that
  couldn't otherwise hold this model's gradients and optimizer state".
- Loss should decrease over training steps even with this few adapters
  training and this little data -- LoRA's whole premise is that most
  fine-tuning updates live in a low-rank subspace, so a small number of
  steps is often enough to see a clear, visible shift in behavior.
