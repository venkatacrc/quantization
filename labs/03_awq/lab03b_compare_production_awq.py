"""Lab 03b - compare the from-scratch AWQ in lab03 against a production
quantization library (autoawq, falling back to llmcompressor's AWQModifier).

Run lab03_awq_from_scratch.py first -- this script reads its metrics.json
and appends the production-library numbers onto the same comparison chart.

NOTE: as in lab02b, quantization-library APIs move fast; this is written
against autoawq's `AutoAWQForCausalLM` API and llmcompressor's `oneshot`
API as of when this lab was written.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from common.data import get_eval_ids
from common.env_check import check_optional, print_banner
from common.eval import perplexity
from common.models import MODEL_REGISTRY, load_tokenizer
from common.plotting import results_dir, save_bar_chart, save_json

MODEL = "small"  # Qwen2.5-1.5B-Instruct, must match lab03_awq_from_scratch.py
SEQ_LEN = 1024
BITS = 4
GROUP_SIZE = 128


def try_autoawq(model_id: str, tokenizer, save_dir: Path):
    try:
        from awq import AutoAWQForCausalLM
    except ImportError:
        return None

    print("  using autoawq...")
    model = AutoAWQForCausalLM.from_pretrained(model_id, safetensors=True)
    quant_config = {"zero_point": True, "q_group_size": GROUP_SIZE, "w_bit": BITS, "version": "GEMM"}

    t0 = time.time()
    model.quantize(tokenizer, quant_config=quant_config)
    elapsed = time.time() - t0
    model.save_quantized(str(save_dir))
    tokenizer.save_pretrained(str(save_dir))
    return {"backend": "autoawq", "seconds": elapsed, "save_dir": str(save_dir)}


def try_llmcompressor(model_id: str, tokenizer, save_dir: Path):
    try:
        from llmcompressor import oneshot
        from llmcompressor.modifiers.quantization import AWQModifier
        from transformers import AutoModelForCausalLM
        from datasets import load_dataset
    except ImportError:
        return None

    print("  using llmcompressor...")
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, device_map="auto")
    calib_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train").select(range(128))
    recipe = AWQModifier(targets="Linear", scheme=f"W{BITS}A16", group_size=GROUP_SIZE, ignore=["lm_head"])

    t0 = time.time()
    oneshot(model=model, dataset=calib_dataset, recipe=recipe, output_dir=str(save_dir))
    elapsed = time.time() - t0
    return {"backend": "llmcompressor", "seconds": elapsed, "save_dir": str(save_dir)}


def load_quantized_for_eval(backend: str, save_dir: Path):
    if backend == "autoawq":
        from awq import AutoAWQForCausalLM

        wrapped = AutoAWQForCausalLM.from_quantized(str(save_dir), fuse_layers=False)
        # AutoAWQ wraps a standard transformers causal LM at `.model`.
        return getattr(wrapped, "model", wrapped)
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(save_dir, device_map="auto")


def main() -> None:
    print_banner("Lab 03b: production AWQ comparison")
    out_dir = results_dir(__file__)
    save_dir = out_dir / "awq_checkpoint"

    from_scratch_metrics_path = out_dir / "metrics.json"
    if not from_scratch_metrics_path.exists():
        print(
            f"\n{from_scratch_metrics_path} not found -- run lab03_awq_from_scratch.py first "
            "so this script has a baseline/RTN/from-scratch-AWQ perplexity to compare against."
        )
        sys.exit(1)
    baseline_metrics = json.loads(from_scratch_metrics_path.read_text())

    print("\nChecking for a production AWQ library...")
    autoawq_check = check_optional("autoawq")
    print(f"  autoawq: {'OK' if autoawq_check.available else 'missing'} -- {autoawq_check.detail}")

    model_id = MODEL_REGISTRY[MODEL]
    tokenizer = load_tokenizer(MODEL)

    print("\nQuantizing with a production backend...")
    result = try_autoawq(model_id, tokenizer, save_dir) or try_llmcompressor(model_id, tokenizer, save_dir)

    if result is None:
        print(
            "\nNeither autoawq nor llmcompressor is installed. Install one and re-run:\n"
            "  pip install autoawq\n"
            "  # or\n"
            "  pip install llmcompressor\n"
        )
        sys.exit(1)

    print(f"  quantized with {result['backend']} in {result['seconds']:.1f}s, saved to {result['save_dir']}")

    print("\nLoading the quantized checkpoint and evaluating perplexity...")
    quant_model = load_quantized_for_eval(result["backend"], save_dir)
    eval_ids = get_eval_ids(tokenizer, split="test")
    prod_ppl = perplexity(quant_model, eval_ids, seq_len=SEQ_LEN)
    print(f"  {result['backend']} perplexity = {prod_ppl:.3f}")

    labels = ["BF16", f"RTN INT{BITS}/g{GROUP_SIZE}", f"AWQ-scratch INT{BITS}/g{GROUP_SIZE}", f"{result['backend']} INT{BITS}/g{GROUP_SIZE}"]
    values = [baseline_metrics["baseline_ppl"], baseline_metrics["rtn_ppl"], baseline_metrics["awq_from_scratch_ppl"], prod_ppl]

    save_bar_chart(
        out_dir / "03_ppl_comparison_with_production.png",
        labels,
        values,
        ylabel="WikiText-2 perplexity",
        title="RTN vs from-scratch AWQ vs production AWQ",
    )

    baseline_metrics["production_backend"] = result["backend"]
    baseline_metrics["production_ppl"] = prod_ppl
    baseline_metrics["production_seconds"] = result["seconds"]
    save_json(from_scratch_metrics_path, baseline_metrics)
    print(f"\nDone. Updated {from_scratch_metrics_path} and wrote 03_ppl_comparison_with_production.png")


if __name__ == "__main__":
    main()
