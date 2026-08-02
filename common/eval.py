"""Evaluation helpers shared by every lab.

``perplexity()`` implements the standard sliding-window WikiText-2
perplexity used in the GPTQ / AWQ / SmoothQuant papers, so numbers you get
out of these labs are comparable to numbers reported in the literature (up
to tokenizer differences).
"""

from __future__ import annotations

import torch
from tqdm import tqdm


@torch.no_grad()
def perplexity(
    model,
    input_ids: torch.Tensor,
    seq_len: int = 2048,
    device: str | torch.device | None = None,
    show_progress: bool = True,
    max_windows: int | None = None,
) -> float:
    """Sliding-window perplexity over a full token sequence.

    Splits ``input_ids`` (1D LongTensor) into non-overlapping windows of
    ``seq_len`` tokens, runs the model on each window, and accumulates
    per-token negative log-likelihood exactly like the reference
    implementations in the GPTQ / AWQ papers (each window's loss is
    weighted by its own token count, then perplexity = exp(total_nll / total_tokens)).

    ``max_windows`` caps how many windows are evaluated -- handy for a quick
    sweep over many candidate configs before doing one full evaluation on
    the winner.
    """
    if device is None:
        device = next(model.parameters()).device

    n_tokens = input_ids.numel()
    n_windows = n_tokens // seq_len
    if max_windows is not None:
        n_windows = min(n_windows, max_windows)

    nll_sum = 0.0
    token_count = 0

    iterator = range(n_windows)
    if show_progress:
        iterator = tqdm(iterator, desc="perplexity", leave=False)

    for i in iterator:
        chunk = input_ids[i * seq_len : (i + 1) * seq_len].unsqueeze(0).to(device)
        outputs = model(chunk, labels=chunk)
        # HF's default loss already averages over (seq_len - 1) predicted
        # tokens; recover the summed NLL so windows combine correctly.
        n_pred = chunk.shape[1] - 1
        nll_sum += outputs.loss.float().item() * n_pred
        token_count += n_pred

    return float(torch.exp(torch.tensor(nll_sum / token_count)))


@torch.no_grad()
def quick_generate(model, tokenizer, prompt: str, max_new_tokens: int = 64) -> str:
    """Small sanity-check helper: generate a short completion so you can
    eyeball that a quantized model still produces coherent text, not just
    look at a perplexity number.
    """
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )
    return tokenizer.decode(out[0], skip_special_tokens=True)
