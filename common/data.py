"""Calibration + evaluation data loaders shared by every lab.

We standardize on WikiText-2 (raw) for both calibration and perplexity
evaluation across labs so results are directly comparable -- the same
metric, computed the same way, appears in labs 01 through 10. C4 is offered
as an alternative calibration set since a few quantization papers (GPTQ,
AWQ) calibrate on C4 rather than WikiText-2.
"""

from __future__ import annotations

import random

import torch


def _load_wikitext2_text(split: str = "train") -> str:
    from datasets import load_dataset

    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)
    return "\n\n".join(ds["text"])


def _load_c4_text(n_docs: int = 2000) -> str:
    from datasets import load_dataset

    ds = load_dataset("allenai/c4", "en", split="train", streaming=True)
    texts = []
    for i, ex in enumerate(ds):
        if i >= n_docs:
            break
        texts.append(ex["text"])
    return "\n\n".join(texts)


def get_calibration_samples(
    tokenizer,
    n_samples: int = 128,
    seq_len: int = 512,
    dataset: str = "wikitext2",
    seed: int = 0,
) -> list[torch.Tensor]:
    """Return a list of ``n_samples`` token-id tensors of shape (1, seq_len),
    randomly sliced from the chosen corpus. This is the standard "calibration
    set" used by PTQ methods (GPTQ, AWQ, SmoothQuant) to observe realistic
    activations without needing labels.
    """
    text = _load_wikitext2_text("train") if dataset == "wikitext2" else _load_c4_text()
    enc = tokenizer(text, return_tensors="pt")
    input_ids = enc["input_ids"][0]

    rng = random.Random(seed)
    max_start = input_ids.numel() - seq_len - 1
    if max_start <= 0:
        raise ValueError("corpus too short for the requested seq_len")

    samples = []
    for _ in range(n_samples):
        start = rng.randint(0, max_start)
        samples.append(input_ids[start : start + seq_len].unsqueeze(0))
    return samples


def get_raw_text(split: str = "test") -> str:
    """The raw (untokenized) WikiText-2 text for a split -- handy for labs
    that need text prompts directly (e.g. feeding a serving engine like
    vLLM) rather than pre-tokenized ids.
    """
    return _load_wikitext2_text(split)


def get_eval_ids(tokenizer, split: str = "test") -> torch.Tensor:
    """Full tokenized WikiText-2 test split, as a single 1D LongTensor, for
    sliding-window perplexity evaluation (see ``common/eval.py``).
    """
    text = _load_wikitext2_text(split)
    enc = tokenizer(text, return_tensors="pt")
    return enc["input_ids"][0]
