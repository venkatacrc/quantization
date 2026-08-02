"""Lab 00 - Quantization Numerics Fundamentals.

Run: python labs/00_fundamentals/lab00_numerics.py
See README.md in this directory for the full background + what to look for.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from common import quant_ops as qo
from common.env_check import print_banner
from common.plotting import results_dir, save_bar_chart, save_histogram, save_json


def make_synthetic_weight(out_features: int = 256, in_features: int = 512, seed: int = 0) -> torch.Tensor:
    """A Gaussian weight matrix with a handful of "outlier" output channels
    scaled up, which is roughly what real LLM projection weights look like
    (see e.g. the SmoothQuant/AWQ papers' activation-outlier analyses --
    weights themselves are milder but still not perfectly uniform).
    """
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(out_features, in_features, generator=g) * 0.02
    n_outliers = max(1, out_features // 32)
    outlier_rows = torch.randperm(out_features, generator=g)[:n_outliers]
    w[outlier_rows] *= 8.0
    return w


def try_load_real_weight() -> torch.Tensor | None:
    try:
        from common.models import load_model

        model = load_model("tiny", dtype=torch.float32, device_map="cpu")
        for name, p in model.named_parameters():
            if "o_proj.weight" in name or "down_proj.weight" in name:
                print(f"  using real weight tensor: {name} {tuple(p.shape)}")
                return p.detach().clone()
    except Exception as e:  # noqa: BLE001 - best-effort, purely illustrative
        print(f"  (skipping real-weight sanity check: {e})")
    return None


def sweep_bits_symmetry(w: torch.Tensor, out_dir: Path) -> dict:
    print("\n[1/5] Sweeping bit-width x symmetric/asymmetric (per-tensor)...")
    rows = []
    for bits in (8, 4, 3, 2):
        for symmetric in (True, False):
            deq = qo.fake_quantize_int(w, bits=bits, symmetric=symmetric, granularity="per_tensor")
            stats = qo.quantization_error_stats(w, deq)
            tag = f"{bits}b-{'sym' if symmetric else 'asym'}"
            rows.append({"config": tag, "bits": bits, "symmetric": symmetric, **stats.__dict__})
            print(f"    {tag:>10s}  rel_l2_err={stats.relative_l2_err:.4f}  mse={stats.mse:.3e}")

    save_bar_chart(
        out_dir / "01_bits_vs_symmetry.png",
        [r["config"] for r in rows],
        [r["relative_l2_err"] for r in rows],
        ylabel="relative L2 error",
        title="Per-tensor quantization error vs bit-width & symmetry",
    )
    return {"bits_symmetry_sweep": rows}


def sweep_granularity(w: torch.Tensor, out_dir: Path) -> dict:
    print("\n[2/5] Sweeping granularity at INT4 (per-tensor / per-channel / per-group)...")
    configs = [
        ("per_tensor", None),
        ("per_channel", None),
        ("per_group(g=128)", 128),
        ("per_group(g=64)", 64),
        ("per_group(g=32)", 32),
    ]
    rows = []
    for label, group_size in configs:
        granularity = "per_group" if group_size else label
        kwargs = {"group_size": group_size} if group_size else {}
        deq = qo.fake_quantize_int(w, bits=4, symmetric=True, granularity=granularity, **kwargs)
        stats = qo.quantization_error_stats(w, deq)
        rows.append({"config": label, **stats.__dict__})
        print(f"    {label:>18s}  rel_l2_err={stats.relative_l2_err:.4f}  mse={stats.mse:.3e}")

    save_bar_chart(
        out_dir / "02_granularity_at_int4.png",
        [r["config"] for r in rows],
        [r["relative_l2_err"] for r in rows],
        ylabel="relative L2 error",
        title="INT4 quantization error vs granularity",
    )
    return {"granularity_sweep": rows}


def compare_int8_vs_fp8(w: torch.Tensor, out_dir: Path) -> dict:
    print("\n[3/5] Comparing INT8 (per-tensor) vs FP8 E4M3 vs FP8 E5M2...")
    int8_deq = qo.fake_quantize_int(w, bits=8, symmetric=True, granularity="per_tensor")
    e4m3_deq, _ = qo.fp8_quantize_dequantize(w, fmt="e4m3")
    e5m2_deq, _ = qo.fp8_quantize_dequantize(w, fmt="e5m2")

    rows = []
    for label, deq in [("INT8", int8_deq), ("FP8-E4M3", e4m3_deq), ("FP8-E5M2", e5m2_deq)]:
        stats = qo.quantization_error_stats(w, deq)
        rows.append({"config": label, **stats.__dict__})
        print(f"    {label:>10s}  rel_l2_err={stats.relative_l2_err:.4f}  mse={stats.mse:.3e}")

    # Outlier-row breakdown: where does each format actually win/lose?
    row_norms = w.norm(dim=1)
    outlier_idx = int(row_norms.argmax())
    normal_idx = int(row_norms.argmin())
    for tag, idx in [("outlier row", outlier_idx), ("typical row", normal_idx)]:
        line = f"      {tag} (|row|={row_norms[idx]:.3f}): "
        for label, deq in [("INT8", int8_deq), ("FP8-E4M3", e4m3_deq)]:
            err = (w[idx] - deq[idx]).norm().item()
            line += f"{label}_err={err:.4f}  "
        print(line)

    save_bar_chart(
        out_dir / "03_int8_vs_fp8.png",
        [r["config"] for r in rows],
        [r["relative_l2_err"] for r in rows],
        ylabel="relative L2 error",
        title="8-bit budget: INT8 vs FP8 (E4M3/E5M2)",
    )
    return {"int8_vs_fp8": rows}


def compare_int4_vs_fp4(w: torch.Tensor, out_dir: Path) -> dict:
    print("\n[4/5] Comparing INT4 vs FP4 (NVFP4-style block=16, MXFP4-style block=32)...")
    int4_pergroup16 = qo.fake_quantize_int(w, bits=4, symmetric=True, granularity="per_group", group_size=16)
    int4_pergroup128 = qo.fake_quantize_int(w, bits=4, symmetric=True, granularity="per_group", group_size=128)
    nvfp4_deq, _ = qo.fp4_blockwise_quantize_dequantize(w, block_size=16, power_of_two_scale=False)
    mxfp4_deq, _ = qo.fp4_blockwise_quantize_dequantize(w, block_size=32, power_of_two_scale=True)

    rows = []
    for label, deq in [
        ("INT4 g=16", int4_pergroup16),
        ("INT4 g=128", int4_pergroup128),
        ("NVFP4 (blk=16)", nvfp4_deq),
        ("MXFP4 (blk=32)", mxfp4_deq),
    ]:
        stats = qo.quantization_error_stats(w, deq)
        rows.append({"config": label, **stats.__dict__})
        print(f"    {label:>16s}  rel_l2_err={stats.relative_l2_err:.4f}  mse={stats.mse:.3e}")

    save_bar_chart(
        out_dir / "04_int4_vs_fp4.png",
        [r["config"] for r in rows],
        [r["relative_l2_err"] for r in rows],
        ylabel="relative L2 error",
        title="4-bit budget: INT4 (per-group) vs FP4 microscaling",
    )
    return {"int4_vs_fp4": rows}, nvfp4_deq


def plot_error_histogram(w: torch.Tensor, nvfp4_deq: torch.Tensor, out_dir: Path) -> None:
    print("\n[5/5] Plotting per-element error histogram (INT4 per-tensor vs NVFP4 blk=16)...")
    int4_deq = qo.fake_quantize_int(w, bits=4, symmetric=True, granularity="per_tensor")
    err_int4 = (w - int4_deq).flatten().numpy()
    err_nvfp4 = (w - nvfp4_deq).flatten().numpy()

    save_histogram(
        out_dir / "05_error_hist_int4_per_tensor.png",
        err_int4,
        bins=80,
        xlabel="quantization error (x - dequant(x))",
        title="INT4 per-tensor: per-element error distribution",
    )
    save_histogram(
        out_dir / "06_error_hist_nvfp4.png",
        err_nvfp4,
        bins=80,
        xlabel="quantization error (x - dequant(x))",
        title="NVFP4 (block=16): per-element error distribution",
    )


def main() -> None:
    print_banner("Lab 00: Quantization Numerics Fundamentals")
    out_dir = results_dir(__file__)

    print("\nBuilding synthetic weight-like tensor...")
    w = make_synthetic_weight()
    print(f"  shape={tuple(w.shape)}  mean={w.mean():.4f}  std={w.std():.4f}  max_abs={w.abs().max():.4f}")

    real_w = try_load_real_weight()
    if real_w is not None:
        real_stats = qo.quantization_error_stats(
            real_w, qo.fake_quantize_int(real_w, bits=4, symmetric=True, granularity="per_channel")
        )
        print(f"  (sanity check) real weight INT4 per-channel rel_l2_err={real_stats.relative_l2_err:.4f}")

    metrics: dict = {}
    metrics.update(sweep_bits_symmetry(w, out_dir))
    metrics.update(sweep_granularity(w, out_dir))
    metrics.update(compare_int8_vs_fp8(w, out_dir))
    fp4_metrics, nvfp4_deq = compare_int4_vs_fp4(w, out_dir)
    metrics.update(fp4_metrics)
    plot_error_histogram(w, nvfp4_deq, out_dir)

    save_json(out_dir / "metrics.json", metrics)
    print(f"\nDone. Plots + metrics.json written to {out_dir}")


if __name__ == "__main__":
    main()
