"""Model registry + loading helpers shared by every lab.

We default to fully open, ungated Qwen2.5 checkpoints so nothing needs an
``HF_TOKEN`` or license click-through. Swap ``MODEL_REGISTRY`` entries for
gated families (e.g. Llama 3.x) if you have access -- everything downstream
(tokenizer, config, generation) works the same way since they're all
standard causal LMs on the Hugging Face Hub.
"""

from __future__ import annotations

import torch

# Logical lab name -> HF Hub model id. Sizes scale up with what each lab
# needs to demonstrate; see the top-level README for the reasoning.
MODEL_REGISTRY: dict[str, str] = {
    "tiny": "Qwen/Qwen2.5-0.5B-Instruct",       # labs 00/01/07 - fast iteration
    "small": "Qwen/Qwen2.5-1.5B-Instruct",      # labs 02/03 - GPTQ/AWQ from scratch
    "medium": "Qwen/Qwen2.5-3B-Instruct",       # lab 04 - SmoothQuant W8A8
    "large": "Qwen/Qwen2.5-7B-Instruct",        # labs 05/06/08/09 - FP8/FP4/QLoRA/KV-cache
    "capstone": "Qwen/Qwen2.5-32B-Instruct",    # lab 10 - multi-GPU serving benchmark
    "capstone_fallback": "Qwen/Qwen2.5-14B-Instruct",  # lab 10 - if 32B is too slow to prep
}


def resolve(name_or_id: str) -> str:
    """Resolve a logical name (e.g. "tiny") or pass through a literal HF id."""
    return MODEL_REGISTRY.get(name_or_id, name_or_id)


def load_tokenizer(name_or_id: str):
    from transformers import AutoTokenizer

    model_id = resolve(name_or_id)
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_model(
    name_or_id: str,
    dtype: torch.dtype = torch.bfloat16,
    device_map: str | dict | None = "auto",
    **kwargs,
):
    """Load a causal LM in the given dtype. Extra kwargs (e.g.
    ``quantization_config``) are forwarded to ``from_pretrained``.
    """
    from transformers import AutoModelForCausalLM

    model_id = resolve(name_or_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=dtype,
        device_map=device_map,
        **kwargs,
    )
    model.eval()
    return model


def model_num_params(model) -> int:
    return sum(p.numel() for p in model.parameters())


def named_linear_layers(model, skip_substrings: tuple[str, ...] = ("embed_tokens", "lm_head")):
    """Yield (name, module) for every ``nn.Linear`` in the model, skipping
    the embedding / LM head by default (both are usually kept at higher
    precision -- vocabulary-sized matrices are unusually sensitive to
    quantization relative to their parameter count).
    """
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and not any(s in name for s in skip_substrings):
            yield name, module


def set_module_by_name(model, name: str, new_module) -> None:
    """Replace a submodule addressed by its dotted name (as returned by
    ``model.named_modules()``) with ``new_module``. Used by labs that need
    to swap in a wrapper module (e.g. AWQ's scaled linear in lab 03) rather
    than just mutating an existing module's weight in place.
    """
    parts = name.split(".")
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], new_module)


def model_memory_footprint_gb(model) -> float:
    """Sum of parameter + buffer byte sizes, in GiB. Useful as a quick,
    dtype-aware "how big is this checkpoint in memory" sanity check --
    independent of whatever the CUDA allocator happens to be caching.
    """
    total_bytes = 0
    for p in model.parameters():
        total_bytes += p.numel() * p.element_size()
    for b in model.buffers():
        total_bytes += b.numel() * b.element_size()
    return total_bytes / (1024**3)
