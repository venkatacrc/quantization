"""Lab 01 - Round-to-Nearest (RTN) weight-only PTQ.

Run: python labs/01_rtn_weight_only/lab01_rtn.py
See README.md in this directory for background + what to look for.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from common import quant_ops as qo
from common.data import get_eval_ids
from common.env_check import print_banner
from common.eval import perplexity, quick_generate
from common.models import load_model, load_tokenizer, model_memory_footprint_gb, named_linear_layers
from common.plotting import results_dir, save_bar_chart, save_json, save_line_chart

MODEL = "tiny"  # Qwen2.5-0.5B-Instruct - see common/models.py
SEQ_LEN = 1024

CONFIGS = [
    # (label, bits, granularity, group_size)
    ("8b/per_tensor", 8, "per_tensor", None),
    ("4b/per_tensor", 4, "per_tensor", None),
    ("4b/per_channel", 4, "per_channel", None),
    ("4b/per_group(128)", 4, "per_group", 128),
    ("4b/per_group(32)", 4, "per_group", 32),
    ("3b/per_tensor", 3, "per_tensor", None),
    ("3b/per_group(32)", 3, "per_group", 32),
]


@torch.no_grad()
def quantize_linears_(model, bits: int, granularity: str, group_size: int | None) -> dict[str, torch.Tensor]:
    """Fake-quantize every linear layer's weight in place; return the
    original weights so the caller can restore them afterwards.
    """
    originals = {}
    kwargs = {"group_size": group_size} if group_size else {}
    for name, module in named_linear_layers(model):
        originals[name] = module.weight.data.clone()
        module.weight.data = qo.fake_quantize_int(
            module.weight.data.float(), bits=bits, symmetric=True, granularity=granularity, **kwargs
        ).to(module.weight.dtype)
    return originals


@torch.no_grad()
def restore_linears_(model, originals: dict[str, torch.Tensor]) -> None:
    for name, module in named_linear_layers(model):
        module.weight.data = originals[name]


def estimate_memory_gb(model, bits: int, group_size: int | None) -> float:
    total_bytes = 0
    for name, module in named_linear_layers(model):
        total_bytes += qo.estimate_packed_size_bytes(module.weight.numel(), bits, group_size)
    # embedding/lm_head kept at bf16 (2 bytes/param), as noted in the README.
    skipped = sum(
        p.numel() for n, p in model.named_parameters() if any(s in n for s in ("embed_tokens", "lm_head"))
    )
    total_bytes += skipped * 2
    return total_bytes / (1024**3)


def main() -> None:
    print_banner("Lab 01: RTN weight-only PTQ")

    out_dir = results_dir(__file__)
    tokenizer = load_tokenizer(MODEL)
    model = load_model(MODEL, dtype=torch.bfloat16)
    eval_ids = get_eval_ids(tokenizer, split="test")

    print("\nEvaluating BF16 baseline...")
    baseline_ppl = perplexity(model, eval_ids, seq_len=SEQ_LEN)
    baseline_mem = model_memory_footprint_gb(model)
    print(f"  baseline perplexity = {baseline_ppl:.3f}")
    print(f"  baseline memory     = {baseline_mem:.3f} GiB (bf16)")

    sample_prompt = "The history of the Roman Empire begins with"
    print(f"\n  baseline generation: {quick_generate(model, tokenizer, sample_prompt, 40)!r}")

    results = [
        {"config": "bf16 (baseline)", "bits": 16, "granularity": "n/a", "ppl": baseline_ppl, "mem_gb": baseline_mem}
    ]

    print("\nSweeping RTN configs...")
    for label, bits, granularity, group_size in CONFIGS:
        originals = quantize_linears_(model, bits, granularity, group_size)
        ppl = perplexity(model, eval_ids, seq_len=SEQ_LEN)
        mem_gb = estimate_memory_gb(model, bits, group_size)
        print(f"  {label:>20s}  ppl={ppl:8.3f}  (Δ={ppl - baseline_ppl:+.3f})  est_packed_mem={mem_gb:.3f} GiB")
        entry = {"config": label, "bits": bits, "granularity": granularity, "group_size": group_size, "ppl": ppl, "mem_gb": mem_gb}
        results.append(entry)

        if label in ("4b/per_tensor", "3b/per_tensor"):
            text = quick_generate(model, tokenizer, sample_prompt, 40)
            print(f"      generation @ {label}: {text!r}")
            entry["sample_generation"] = text

        restore_linears_(model, originals)

    # --- Plots -------------------------------------------------------
    by_granularity: dict[str, list[tuple[int, float]]] = {}
    for r in results[1:]:
        by_granularity.setdefault(r["granularity"], []).append((r["bits"], r["ppl"]))

    all_bits = sorted({b for pts in by_granularity.values() for b, _ in pts}, reverse=True)
    series = {}
    for gran, pts in by_granularity.items():
        lookup = dict(pts)
        series[gran] = [lookup.get(b, float("nan")) for b in all_bits]

    save_line_chart(
        out_dir / "01_ppl_vs_bits.png",
        all_bits,
        series,
        xlabel="bits",
        ylabel="WikiText-2 perplexity",
        title="RTN: perplexity vs bit-width, by granularity (baseline dashed)",
    )
    save_bar_chart(
        out_dir / "02_memory_footprint.png",
        [r["config"] for r in results],
        [r["mem_gb"] for r in results],
        ylabel="estimated packed memory (GiB)",
        title="RTN: estimated real packed memory footprint by config",
    )

    save_json(out_dir / "metrics.json", {"baseline_ppl": baseline_ppl, "baseline_mem_gb": baseline_mem, "sweep": results})
    print(f"\nDone. Plots + metrics.json written to {out_dir}")


if __name__ == "__main__":
    main()
