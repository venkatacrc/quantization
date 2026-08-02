"""Lab 06 - FP4 / NVFP4 / MXFP4 (Blackwell-native 4-bit formats).

Run: python labs/06_fp4_nvfp4/lab06_fp4_blackwell.py
See README.md in this directory for background + what to look for.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from common import quant_ops as qo
from common.data import get_calibration_samples, get_eval_ids
from common.env_check import print_banner
from common.eval import perplexity, quick_generate
from common.models import load_model, load_tokenizer, named_linear_layers
from common.plotting import results_dir, save_bar_chart, save_json

MODEL = "large"  # Qwen2.5-7B-Instruct, matches lab 05 for a fair FP8/FP4 comparison
SEQ_LEN = 1024
N_CALIB_SAMPLES = 64
CALIB_SEQ_LEN = 512

LABS_ROOT = Path(__file__).resolve().parents[1]


def apply_and_eval(model, eval_ids, layer_names, modules_by_name, originals, quant_fn) -> float:
    for name in layer_names:
        module = modules_by_name[name]
        module.weight.data = quant_fn(module.weight.data.float()).to(module.weight.dtype)
    ppl = perplexity(model, eval_ids, seq_len=SEQ_LEN, show_progress=False)
    for name in layer_names:
        modules_by_name[name].weight.data = originals[name].clone()
    return ppl


def try_modelopt_nvfp4(model, calib_samples, device) -> str | None:
    try:
        import modelopt.torch.quantization as mtq
    except ImportError:
        print("  nvidia-modelopt not installed; skipping this backend.")
        return None
    cfg = getattr(mtq, "NVFP4_DEFAULT_CFG", None)
    if cfg is None:
        print("  nvidia-modelopt is installed but this version has no NVFP4_DEFAULT_CFG; skipping.")
        return None
    def forward_loop(m):
        with torch.no_grad():
            for sample in calib_samples:
                m(sample.to(device))
    print("  running nvidia-modelopt NVFP4 PTQ...")
    try:
        mtq.quantize(model, cfg, forward_loop)
        return "nvidia-modelopt (NVFP4)"
    except Exception as e:
        print(f"  nvidia-modelopt NVFP4 PTQ failed ({type(e).__name__}: {e}); skipping this backend.")
        return None


def try_torchao_fp4(model) -> str | None:
    try:
        from torchao.quantization import quantize_
        import torchao.quantization.quant_api as qapi
    except ImportError:
        print("  torchao not installed; skipping this backend.")
        return None

    for name in ("NVFP4WeightOnlyConfig", "NVFP4InferenceConfig", "MXFP4WeightOnlyConfig"):
        cfg_cls = getattr(qapi, name, None)
        if cfg_cls is None:
            continue
        try:
            print(f"  trying torchao.{name}...")
            quantize_(model, cfg_cls())
            return f"torchao.{name}"
        except Exception as e:  # noqa: BLE001 - best-effort across fast-moving API
            print(f"    failed: {e}")
    print("  no working NVFP4/MXFP4 config found in this torchao version; skipping.")
    return None


def load_other_lab_metrics() -> dict:
    out = {}
    for lab_dir, key in [
        ("01_rtn_weight_only", "lab01"),
        ("02_gptq", "lab02"),
        ("03_awq", "lab03"),
        ("05_fp8", "lab05"),
    ]:
        p = LABS_ROOT / lab_dir / "results" / "metrics.json"
        if p.exists():
            out[key] = json.loads(p.read_text())
    return out


def main() -> None:
    print_banner("Lab 06: FP4 / NVFP4 / MXFP4")
    out_dir = results_dir(__file__)

    tokenizer = load_tokenizer(MODEL)
    model = load_model(MODEL, dtype=torch.bfloat16)
    device = next(model.parameters()).device
    eval_ids = get_eval_ids(tokenizer, split="test")
    layer_names = [name for name, _ in named_linear_layers(model)]
    print(f"\n{len(layer_names)} target linear layers found.")

    print("\nEvaluating BF16 baseline...")
    baseline_ppl = perplexity(model, eval_ids, seq_len=SEQ_LEN)
    print(f"  baseline perplexity = {baseline_ppl:.3f}")

    modules_by_name = dict(model.named_modules())
    originals = {name: modules_by_name[name].weight.data.clone() for name in layer_names}

    print("\n[1/2] From-scratch weight-only sweep: INT4(g=32) vs NVFP4(blk=16) vs MXFP4(blk=32)...")
    configs = {
        "INT4 (g=32)": lambda w: qo.fake_quantize_int(w, bits=4, symmetric=True, granularity="per_group", group_size=32),
        "NVFP4 (blk=16)": lambda w: qo.fp4_blockwise_quantize_dequantize(w, block_size=16, power_of_two_scale=False)[0],
        "MXFP4 (blk=32)": lambda w: qo.fp4_blockwise_quantize_dequantize(w, block_size=32, power_of_two_scale=True)[0],
    }
    scratch_results = []
    for label, fn in configs.items():
        ppl = apply_and_eval(model, eval_ids, layer_names, modules_by_name, originals, fn)
        print(f"    {label:>16s}  ppl={ppl:.3f}  (Δ={ppl - baseline_ppl:+.3f})")
        scratch_results.append({"config": label, "ppl": ppl})

    # Keep the best from-scratch FP4 config applied for a qualitative sample.
    best_fp4 = min((r for r in scratch_results if "FP4" in r["config"]), key=lambda r: r["ppl"])
    for name in layer_names:
        modules_by_name[name].weight.data = configs[best_fp4["config"]](originals[name].float()).to(modules_by_name[name].weight.dtype)
    sample_prompt = "The history of the Roman Empire begins with"
    print(f"\n  generation @ {best_fp4['config']}: {quick_generate(model, tokenizer, sample_prompt, 40)!r}")
    for name in layer_names:
        modules_by_name[name].weight.data = originals[name].clone()

    print("\n[2/2] Attempting an accelerated NVFP4/MXFP4 path (nvidia-modelopt, then torchao)...")
    calib_samples = get_calibration_samples(tokenizer, N_CALIB_SAMPLES, CALIB_SEQ_LEN, dataset="wikitext2")
    accel_backend = try_modelopt_nvfp4(model, calib_samples, device) or try_torchao_fp4(model)

    accel_ppl = None
    if accel_backend is not None:
        accel_ppl = perplexity(model, eval_ids, seq_len=SEQ_LEN)
        print(f"  {accel_backend} perplexity = {accel_ppl:.3f}  (Δ={accel_ppl - baseline_ppl:+.3f})")
    else:
        print(
            "\n  Neither nvidia-modelopt nor a working torchao NVFP4/MXFP4 config was found.\n"
            "  Install one and re-run to see the accelerated-path comparison:\n"
            "    pip install nvidia-modelopt\n"
            "    # or\n"
            "    pip install --upgrade torchao\n"
        )

    print("\nBuilding cross-lab relative-degradation comparison (if other labs were run)...")
    other = load_other_lab_metrics()
    degradation_labels, degradation_values = [], []
    for label, ppl in [(c["config"], c["ppl"]) for c in scratch_results]:
        degradation_labels.append(f"lab06/{label}")
        degradation_values.append((ppl - baseline_ppl) / baseline_ppl * 100)
    if accel_ppl is not None:
        degradation_labels.append(f"lab06/{accel_backend}")
        degradation_values.append((accel_ppl - baseline_ppl) / baseline_ppl * 100)
    if "lab05" in other:
        m = other["lab05"]
        degradation_labels.append("lab05/FP8-W8A8")
        degradation_values.append((m["fp8_w8a8_ppl"] - m["baseline_ppl"]) / m["baseline_ppl"] * 100)
    if "lab01" in other:
        m = other["lab01"]
        int4_rows = [r for r in m["sweep"] if r["bits"] == 4 and "group" in r["granularity"]]
        if int4_rows:
            best = min(int4_rows, key=lambda r: r["ppl"])
            degradation_labels.append("lab01/RTN-INT4")
            degradation_values.append((best["ppl"] - m["baseline_ppl"]) / m["baseline_ppl"] * 100)
    if "lab02" in other:
        m = other["lab02"]
        degradation_labels.append("lab02/GPTQ-INT4")
        degradation_values.append((m["gptq_from_scratch_ppl"] - m["baseline_ppl"]) / m["baseline_ppl"] * 100)
    if "lab03" in other:
        m = other["lab03"]
        degradation_labels.append("lab03/AWQ-INT4")
        degradation_values.append((m["awq_from_scratch_ppl"] - m["baseline_ppl"]) / m["baseline_ppl"] * 100)

    save_bar_chart(
        out_dir / "01_scratch_ppl.png",
        [r["config"] for r in scratch_results],
        [r["ppl"] for r in scratch_results],
        ylabel="WikiText-2 perplexity",
        title="From-scratch 4-bit weight-only: INT4 vs NVFP4 vs MXFP4",
    )
    if degradation_values:
        save_bar_chart(
            out_dir / "02_cross_lab_relative_degradation.png",
            degradation_labels,
            degradation_values,
            ylabel="% perplexity degradation vs that lab's own BF16 baseline",
            title="Relative degradation across every precision covered so far",
        )

    save_json(
        out_dir / "metrics.json",
        {
            "model": MODEL,
            "baseline_ppl": baseline_ppl,
            "scratch_sweep": scratch_results,
            "accelerated_backend": accel_backend,
            "accelerated_ppl": accel_ppl,
        },
    )
    print(f"\nDone. Plots + metrics.json written to {out_dir}")


if __name__ == "__main__":
    main()
