"""Lab 05b - compare the from-scratch FP8 W8A8 model in lab05 against
torchao's fused Float8 dynamic-activation/weight quantization.

Run lab05_fp8_inference.py first -- this script reads its metrics.json and
appends torchao's numbers onto the same comparison.

NOTE: as in labs 02b/03b, this is written against torchao's
`quantize_` + `Float8DynamicActivationFloat8WeightConfig` API as of when
this lab was written; check torchao's own README/examples if it has moved.
"""

from __future__ import annotations

import json
import sys

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from common.bench import benchmark_generate, print_results_table
from common.data import get_eval_ids
from common.env_check import check_optional, print_banner
from common.eval import perplexity
from common.models import load_model, load_tokenizer, model_memory_footprint_gb
from common.plotting import results_dir, save_bar_chart, save_json

MODEL = "large"  # Qwen2.5-7B-Instruct, must match lab05_fp8_inference.py
SEQ_LEN = 1024


def main() -> None:
    print_banner("Lab 05b: torchao FP8 comparison")
    out_dir = results_dir(__file__)

    from_scratch_metrics_path = out_dir / "metrics.json"
    if not from_scratch_metrics_path.exists():
        print(
            f"\n{from_scratch_metrics_path} not found -- run lab05_fp8_inference.py first "
            "so this script has a baseline/from-scratch FP8 perplexity to compare against."
        )
        sys.exit(1)
    baseline_metrics = json.loads(from_scratch_metrics_path.read_text())

    torchao_check = check_optional("torchao")
    print(f"\ntorchao: {'OK' if torchao_check.available else 'missing'} -- {torchao_check.detail}")
    if not torchao_check.available:
        print("\nInstall it and re-run:\n  pip install torchao\n")
        sys.exit(1)

    from torchao.quantization import quantize_
    from torchao.quantization.quant_api import Float8DynamicActivationFloat8WeightConfig

    tokenizer = load_tokenizer(MODEL)
    print("\nLoading model and applying torchao Float8DynamicActivationFloat8WeightConfig...")
    model = load_model(MODEL, dtype=torch.bfloat16)
    quantize_(model, Float8DynamicActivationFloat8WeightConfig())

    eval_ids = get_eval_ids(tokenizer, split="test")
    torchao_ppl = perplexity(model, eval_ids, seq_len=SEQ_LEN)
    torchao_mem = model_memory_footprint_gb(model)
    print(f"  torchao FP8 perplexity = {torchao_ppl:.3f}, memory = {torchao_mem:.2f} GiB")

    sample_prompt = "The history of the Roman Empire begins with"
    bench = benchmark_generate(model, tokenizer, sample_prompt, "torchao-FP8", batch_size=8, max_new_tokens=128)
    print_results_table([bench])

    labels = ["BF16", "FP8-W8A8 (from scratch)", "FP8-W8A8 (torchao)"]
    values = [baseline_metrics["baseline_ppl"], baseline_metrics["fp8_w8a8_ppl"], torchao_ppl]

    save_bar_chart(
        out_dir / "04_ppl_comparison_with_torchao.png",
        labels,
        values,
        ylabel="WikiText-2 perplexity",
        title="From-scratch vs torchao FP8 W8A8 perplexity",
    )

    e2e = baseline_metrics.get("e2e_benchmarks", [])
    tput_labels = [r["label"] for r in e2e] + ["torchao-FP8"]
    tput_values = [r["tokens_per_s"] for r in e2e] + [bench.tokens_per_s]
    save_bar_chart(
        out_dir / "05_tokens_per_s_with_torchao.png",
        tput_labels,
        tput_values,
        ylabel="decode tokens/sec",
        title="Generation throughput: BF16 vs from-scratch FP8 vs torchao FP8",
    )

    baseline_metrics["torchao_ppl"] = torchao_ppl
    baseline_metrics["torchao_mem_gb"] = torchao_mem
    baseline_metrics["torchao_bench"] = bench.as_row()
    save_json(from_scratch_metrics_path, baseline_metrics)
    print(f"\nDone. Updated {from_scratch_metrics_path}")


if __name__ == "__main__":
    main()
