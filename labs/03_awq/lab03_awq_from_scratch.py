"""Lab 03 - AWQ from scratch (activation-aware per-channel weight scaling).

Run: python labs/03_awq/lab03_awq_from_scratch.py
See README.md in this directory for background + what to look for.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import torch.nn as nn
import torch.nn.functional as F

from common import quant_ops as qo
from common.data import get_calibration_samples, get_eval_ids
from common.env_check import print_banner
from common.eval import perplexity, quick_generate
from common.models import load_model, load_tokenizer, named_linear_layers, set_module_by_name
from common.plotting import results_dir, save_bar_chart, save_json

MODEL = "small"  # Qwen2.5-1.5B-Instruct, same as lab 02
SEQ_LEN = 1024
N_CALIB_SAMPLES = 128
CALIB_SEQ_LEN = 512
CACHE_BATCHES_PER_LAYER = 2
BITS = 4
GROUP_SIZE = 128
ALPHAS = [i / 8 for i in range(9)]  # 0, 0.125, ..., 1.0


# ---------------------------------------------------------------------------
# Step 1: collect per-channel activation magnitude + a small cached batch of
# raw activations (used for the reconstruction-error scale search below).
# ---------------------------------------------------------------------------


class ActivationStats:
    def __init__(self, in_features: int, device, cache_batches: int = CACHE_BATCHES_PER_LAYER):
        self.abs_sum = torch.zeros(in_features, device=device, dtype=torch.float32)
        self.n_tokens = 0
        self.cache_batches = cache_batches
        self.cached: list[torch.Tensor] = []

    def update(self, x: torch.Tensor) -> None:
        x = x.reshape(-1, x.shape[-1]).float()
        if self.abs_sum.device != x.device:
            self.abs_sum = self.abs_sum.to(x.device)
        self.abs_sum += x.abs().sum(dim=0)
        self.n_tokens += x.shape[0]
        if len(self.cached) < self.cache_batches:
            self.cached.append(x.clone())

    @property
    def act_scale(self) -> torch.Tensor:
        return self.abs_sum / max(self.n_tokens, 1)

    @property
    def cached_x(self) -> torch.Tensor:
        return torch.cat(self.cached, dim=0)


def collect_activation_stats(model, tokenizer, layer_names: list[str]) -> dict[str, ActivationStats]:
    device = next(model.parameters()).device
    stats: dict[str, ActivationStats] = {}
    modules_by_name = dict(model.named_modules())
    hooks = []

    def make_hook(name):
        def hook(module, inputs, output):
            x = inputs[0]
            if name not in stats:
                stats[name] = ActivationStats(module.in_features, device)
            stats[name].update(x)

        return hook

    for name in layer_names:
        hooks.append(modules_by_name[name].register_forward_hook(make_hook(name)))

    print(f"  running {N_CALIB_SAMPLES} calibration sequences ({CALIB_SEQ_LEN} tokens each) through the model...")
    samples = get_calibration_samples(tokenizer, N_CALIB_SAMPLES, CALIB_SEQ_LEN, dataset="wikitext2")
    with torch.no_grad():
        for i, sample in enumerate(samples):
            model(sample.to(device))
            if (i + 1) % 32 == 0:
                print(f"    calibration pass {i + 1}/{len(samples)}")

    for h in hooks:
        h.remove()
    return stats


# ---------------------------------------------------------------------------
# Step 2: per-layer AWQ scale search + quantization.
# ---------------------------------------------------------------------------


class AWQScaledLinear(nn.Module):
    """Drop-in replacement for `nn.Linear` that divides its input by the
    AWQ scale vector before the matmul against the (already scaled +
    quantized) weight -- see the README for why this is numerically
    equivalent to fusing the division into the previous layer.
    """

    def __init__(self, quantized_weight: torch.Tensor, scale: torch.Tensor, bias: torch.Tensor | None, dtype: torch.dtype):
        super().__init__()
        self.weight = nn.Parameter(quantized_weight.to(dtype), requires_grad=False)
        self.register_buffer("inv_scale", (1.0 / scale).to(dtype))
        self.bias = nn.Parameter(bias.to(dtype), requires_grad=False) if bias is not None else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x * self.inv_scale.to(x.dtype), self.weight, self.bias)


@torch.no_grad()
def search_awq_scale(
    weight: torch.Tensor, act_scale: torch.Tensor, cached_x: torch.Tensor, bits: int, group_size: int
) -> tuple[float, torch.Tensor, torch.Tensor]:
    """Grid-search alpha, return (best_mse, best_alpha, best_scale, best_quantized_weight)."""
    W = weight.float()
    w_scale = W.abs().mean(dim=0).clamp(min=1e-5)
    act_scale = act_scale.clamp(min=1e-5)
    y_true = cached_x @ W.T

    best = None
    for alpha in ALPHAS:
        s = (act_scale.pow(alpha) / w_scale.pow(1 - alpha)).clamp(min=1e-4, max=1e4)
        s = s / s.mean()  # keep the overall magnitude stable across alphas
        w_scaled = W * s.unsqueeze(0)
        w_q = qo.fake_quantize_int(w_scaled, bits=bits, symmetric=True, granularity="per_group", group_size=group_size)
        y_hat = (cached_x / s.unsqueeze(0)) @ w_q.T
        mse = (y_true - y_hat).pow(2).mean().item()
        if best is None or mse < best[0]:
            best = (mse, alpha, s.clone(), w_q.clone())
    return best


def main() -> None:
    print_banner("Lab 03: AWQ from scratch")
    out_dir = results_dir(__file__)

    tokenizer = load_tokenizer(MODEL)
    model = load_model(MODEL, dtype=torch.bfloat16)
    eval_ids = get_eval_ids(tokenizer, split="test")
    layer_names = [name for name, _ in named_linear_layers(model)]
    print(f"\n{len(layer_names)} target linear layers found.")

    print("\nEvaluating BF16 baseline...")
    baseline_ppl = perplexity(model, eval_ids, seq_len=SEQ_LEN)
    print(f"  baseline perplexity = {baseline_ppl:.3f}")

    print("\n[1/3] Collecting per-channel activation statistics...")
    t0 = time.time()
    stats = collect_activation_stats(model, tokenizer, layer_names)
    print(f"  done in {time.time() - t0:.1f}s")

    modules_by_name = dict(model.named_modules())
    originals = {name: modules_by_name[name].weight.data.clone() for name in layer_names}

    print(f"\n[2/3] Plain RTN at INT{BITS}/group_size={GROUP_SIZE} for comparison...")
    for name in layer_names:
        module = modules_by_name[name]
        module.weight.data = qo.fake_quantize_int(
            module.weight.data.float(), bits=BITS, symmetric=True, granularity="per_group", group_size=GROUP_SIZE
        ).to(module.weight.dtype)
    rtn_ppl = perplexity(model, eval_ids, seq_len=SEQ_LEN)
    print(f"  RTN perplexity = {rtn_ppl:.3f}  (Δ={rtn_ppl - baseline_ppl:+.3f})")
    for name in layer_names:
        modules_by_name[name].weight.data = originals[name].clone()

    print(f"\n[3/3] AWQ scale search + quantization at INT{BITS}/group_size={GROUP_SIZE}...")
    t0 = time.time()
    layer_alphas = []
    for i, name in enumerate(layer_names):
        module = modules_by_name[name]
        s = stats[name]
        mse, alpha, scale, w_q = search_awq_scale(
            originals[name], s.act_scale, s.cached_x, bits=BITS, group_size=GROUP_SIZE
        )
        new_module = AWQScaledLinear(w_q, scale, module.bias.data if module.bias is not None else None, module.weight.dtype)
        set_module_by_name(model, name, new_module)
        layer_alphas.append({"layer": name, "alpha": alpha, "recon_mse": mse})
        del stats[name]  # free the cached activation batch as we go
        if (i + 1) % 32 == 0:
            print(f"    quantized {i + 1}/{len(layer_names)} layers")
    awq_elapsed = time.time() - t0
    print(f"  done in {awq_elapsed:.1f}s")

    awq_ppl = perplexity(model, eval_ids, seq_len=SEQ_LEN)
    print(f"  AWQ (from scratch) perplexity = {awq_ppl:.3f}  (Δ={awq_ppl - baseline_ppl:+.3f})")

    sample_prompt = "The history of the Roman Empire begins with"
    print(f"\n  generation @ AWQ-from-scratch: {quick_generate(model, tokenizer, sample_prompt, 40)!r}")

    layer_alphas.sort(key=lambda r: r["alpha"], reverse=True)
    print("\n  layers that picked the largest alpha (most activation-outlier-sensitive):")
    for r in layer_alphas[:5]:
        print(f"    {r['layer']:>40s}  alpha={r['alpha']:.3f}  recon_mse={r['recon_mse']:.3e}")

    labels = ["BF16", f"RTN INT{BITS}/g{GROUP_SIZE}", f"AWQ-scratch INT{BITS}/g{GROUP_SIZE}"]
    values = [baseline_ppl, rtn_ppl, awq_ppl]

    gptq_metrics_path = Path(__file__).resolve().parents[1] / "02_gptq" / "results" / "metrics.json"
    if gptq_metrics_path.exists():
        gptq_metrics = json.loads(gptq_metrics_path.read_text())
        labels.append(f"GPTQ-scratch INT{gptq_metrics['bits']}/g{gptq_metrics['group_size']}")
        values.append(gptq_metrics["gptq_from_scratch_ppl"])

    save_bar_chart(
        out_dir / "01_ppl_comparison.png",
        labels,
        values,
        ylabel="WikiText-2 perplexity",
        title="RTN vs AWQ-from-scratch (and GPTQ, if lab 02 was run)",
    )
    save_bar_chart(
        out_dir / "02_top_alpha_layers.png",
        [r["layer"].replace("model.layers.", "L") for r in layer_alphas[:10]],
        [r["alpha"] for r in layer_alphas[:10]],
        ylabel="best alpha",
        title="AWQ: 10 layers with the largest activation-aware scale exponent",
    )

    save_json(
        out_dir / "metrics.json",
        {
            "model": MODEL,
            "bits": BITS,
            "group_size": GROUP_SIZE,
            "baseline_ppl": baseline_ppl,
            "rtn_ppl": rtn_ppl,
            "awq_from_scratch_ppl": awq_ppl,
            "awq_from_scratch_seconds": awq_elapsed,
            "layer_alphas": layer_alphas,
        },
    )
    print(f"\nDone. Plots + metrics.json written to {out_dir}")
    print("Next: run lab03b_compare_production_awq.py to compare against a production AWQ library.")


if __name__ == "__main__":
    main()
