"""Lab 02b - compare the from-scratch GPTQ in lab02 against a production
quantization library (gptqmodel, falling back to llmcompressor).

Run lab02_gptq_from_scratch.py first -- this script reads its metrics.json
and appends the production-library numbers onto the same comparison chart.

NOTE: quantization-library APIs move fast. This script is written against
gptqmodel's `GPTQModel.load/.quantize/.save` API and llmcompressor's
`oneshot(...)` API as of when this lab was written. If either has changed
on your installed version, the error message from the failing call plus
that library's own README/examples should get you unstuck quickly -- the
important thing this lab teaches is *what number to compare against*, not
memorizing one library's call signature.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from common.data import get_calibration_samples, get_eval_ids
from common.env_check import check_optional, print_banner
from common.eval import perplexity
from common.models import MODEL_REGISTRY, load_tokenizer
from common.plotting import results_dir, save_bar_chart, save_json

MODEL = "small"  # Qwen2.5-1.5B-Instruct, must match lab02_gptq_from_scratch.py
SEQ_LEN = 1024
BITS = 4
GROUP_SIZE = 128
SAVE_DIR = None  # set below, inside results/


def build_calibration_texts(tokenizer, n_samples: int = 128, seq_len: int = 512) -> list[str]:
    samples = get_calibration_samples(tokenizer, n_samples, seq_len, dataset="wikitext2")
    return [tokenizer.decode(s[0]) for s in samples]


def try_gptqmodel(model_id: str, calib_texts: list[str], save_dir: Path) -> dict | None:
    try:
        from gptqmodel import GPTQModel, QuantizeConfig
    except ImportError:
        return None

    print("  using gptqmodel...")
    quant_config = QuantizeConfig(bits=BITS, group_size=GROUP_SIZE)
    model = GPTQModel.load(model_id, quant_config)
    t0 = time.time()
    model.quantize(calib_texts, batch_size=4)
    elapsed = time.time() - t0
    model.save(str(save_dir))
    return {"backend": "gptqmodel", "seconds": elapsed, "save_dir": str(save_dir)}


def try_llmcompressor(model_id: str, calib_texts: list[str], save_dir: Path) -> dict | None:
    try:
        from llmcompressor import oneshot
        from llmcompressor.modifiers.quantization import GPTQModifier
        from datasets import Dataset
        from transformers import AutoModelForCausalLM
    except ImportError:
        return None
    print("  using llmcompressor...")
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, device_map="auto")
    calib_dataset = Dataset.from_dict({"text": calib_texts})
    recipe = GPTQModifier(
        targets="Linear",
        ignore=["lm_head"],
        config_groups={
            "group_0": {
                "targets": ["Linear"],
                "weights": {
                    "num_bits": BITS,
                    "type": "int",
                    "symmetric": True,
                    "strategy": "group",
                    "group_size": GROUP_SIZE,
                },
                "input_activations": None,
                "output_activations": None,
            }
        },
    )

    t0 = time.time()
    oneshot(model=model, dataset=calib_dataset, recipe=recipe, output_dir=str(save_dir))
    elapsed = time.time() - t0
    return {"backend": "llmcompressor", "seconds": elapsed, "save_dir": str(save_dir)}


def main() -> None:
    print_banner("Lab 02b: production GPTQ comparison")
    out_dir = results_dir(__file__)
    save_dir = out_dir / "gptq_checkpoint"

    from_scratch_metrics_path = out_dir / "metrics.json"
    if not from_scratch_metrics_path.exists():
        print(
            f"\n{from_scratch_metrics_path} not found -- run lab02_gptq_from_scratch.py first "
            "so this script has a baseline/RTN/from-scratch-GPTQ perplexity to compare against."
        )
        sys.exit(1)
    baseline_metrics = json.loads(from_scratch_metrics_path.read_text())

    print("\nChecking for a production GPTQ library...")
    gptqmodel_check = check_optional("gptqmodel")
    print(f"  gptqmodel: {'OK' if gptqmodel_check.available else 'missing'} -- {gptqmodel_check.detail}")

    model_id = MODEL_REGISTRY[MODEL]
    tokenizer = load_tokenizer(MODEL)
    print("\nBuilding calibration text set (same WikiText-2 sampling as lab02)...")
    calib_texts = build_calibration_texts(tokenizer)

    print("\nQuantizing with a production backend...")
    result = try_gptqmodel(model_id, calib_texts, save_dir) or try_llmcompressor(model_id, calib_texts, save_dir)

    if result is None:
        print(
            "\nNeither gptqmodel nor llmcompressor is installed. Install one and re-run:\n"
            "  pip install gptqmodel\n"
            "  # or\n"
            "  pip install llmcompressor\n"
        )
        sys.exit(1)

    print(f"  quantized with {result['backend']} in {result['seconds']:.1f}s, saved to {result['save_dir']}")

    print("\nLoading the quantized checkpoint and evaluating perplexity...")
    from transformers import AutoModelForCausalLM

    quant_model = AutoModelForCausalLM.from_pretrained(save_dir, device_map="auto")
    eval_ids = get_eval_ids(tokenizer, split="test")
    prod_ppl = perplexity(quant_model, eval_ids, seq_len=SEQ_LEN)
    print(f"  {result['backend']} perplexity = {prod_ppl:.3f}")

    labels = ["BF16", f"RTN INT{BITS}/g{GROUP_SIZE}", f"GPTQ-scratch INT{BITS}/g{GROUP_SIZE}", f"{result['backend']} INT{BITS}/g{GROUP_SIZE}"]
    values = [baseline_metrics["baseline_ppl"], baseline_metrics["rtn_ppl"], baseline_metrics["gptq_from_scratch_ppl"], prod_ppl]

    save_bar_chart(
        out_dir / "03_ppl_comparison_with_production.png",
        labels,
        values,
        ylabel="WikiText-2 perplexity",
        title="RTN vs from-scratch GPTQ vs production GPTQ",
    )

    baseline_metrics["production_backend"] = result["backend"]
    baseline_metrics["production_ppl"] = prod_ppl
    baseline_metrics["production_seconds"] = result["seconds"]
    baseline_metrics["from_scratch_seconds"] = baseline_metrics.get("gptq_from_scratch_seconds")
    save_json(from_scratch_metrics_path, baseline_metrics)
    print(f"\nDone. Updated {from_scratch_metrics_path} and wrote 03_ppl_comparison_with_production.png")


if __name__ == "__main__":
    main()
