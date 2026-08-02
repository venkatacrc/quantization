"""Lab 04 - SmoothQuant (W8A8) + real INT8 GEMM benchmark.

Run: python labs/04_smoothquant/lab04_smoothquant.py
See README.md in this directory for background + what to look for.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import torch.nn as nn

from common.bench import cuda_timer
from common.data import get_calibration_samples, get_eval_ids
from common.env_check import print_banner
from common.eval import perplexity, quick_generate
from common.models import load_model, load_tokenizer, model_memory_footprint_gb, named_linear_layers, set_module_by_name
from common.plotting import results_dir, save_bar_chart, save_json

MODEL = "medium"  # Qwen2.5-3B-Instruct
SEQ_LEN = 1024
N_CALIB_SAMPLES = 128
CALIB_SEQ_LEN = 512
ALPHA_SWEEP = [0.3, 0.5, 0.7]
SWEEP_MAX_WINDOWS = 4
EPS = 1e-5


# ---------------------------------------------------------------------------
# Step 1: per-channel activation max-abs statistics (for the smoothing scale).
# ---------------------------------------------------------------------------


class MaxAbsStats:
    def __init__(self, in_features: int, device):
        self.max_abs = torch.zeros(in_features, device=device, dtype=torch.float32)

    def update(self, x: torch.Tensor) -> None:
        x = x.reshape(-1, x.shape[-1]).float().abs()
        self.max_abs = torch.maximum(self.max_abs, x.amax(dim=0))


def collect_max_abs_stats(model, tokenizer, layer_names: list[str]) -> dict[str, torch.Tensor]:
    device = next(model.parameters()).device
    stats: dict[str, MaxAbsStats] = {}
    modules_by_name = dict(model.named_modules())
    hooks = []

    def make_hook(name):
        def hook(module, inputs, output):
            x = inputs[0]
            if name not in stats:
                stats[name] = MaxAbsStats(module.in_features, device)
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
    return {name: s.max_abs for name, s in stats.items()}


# ---------------------------------------------------------------------------
# Step 2: smoothing + static INT8 weight+activation quantization.
# ---------------------------------------------------------------------------


class SmoothQuantW8A8Linear(nn.Module):
    """Real INT8 x INT8 matmul (via `torch._int_mm`) with SmoothQuant-style
    per-channel weight smoothing folded in. See the README for why the
    `inv_smooth_scale` multiply is applied explicitly here instead of being
    fused into the previous layer.
    """

    def __init__(
        self,
        w_int8_t: torch.Tensor,     # (in_features, out_features), int8
        w_row_scale: torch.Tensor,  # (out_features,)
        inv_smooth_scale: torch.Tensor,  # (in_features,)
        act_scale: float,
        bias: torch.Tensor | None,
        dtype: torch.dtype,
        dynamic_activation: bool = False,
    ):
        super().__init__()
        self.register_buffer("w_int8", w_int8_t)
        self.register_buffer("w_row_scale", w_row_scale.to(torch.float32))
        self.register_buffer("inv_smooth_scale", inv_smooth_scale.to(dtype))
        self.act_scale = act_scale
        self.bias = nn.Parameter(bias.to(dtype), requires_grad=False) if bias is not None else None
        self.dtype = dtype
        self.dynamic_activation = dynamic_activation
        self._int_mm_ok = torch.cuda.is_available()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        x_smooth = (x * self.inv_smooth_scale).reshape(-1, orig_shape[-1]).float()

        act_scale = x_smooth.abs().max().item() / 127.0 if self.dynamic_activation else self.act_scale
        act_scale = max(act_scale, EPS)
        x_q = torch.clamp(torch.round(x_smooth / act_scale), -127, 127).to(torch.int8)

        if self._int_mm_ok:
            try:
                out_i32 = torch._int_mm(x_q.contiguous(), self.w_int8)
            except Exception:
                self._int_mm_ok = False
                out_i32 = x_q.float() @ self.w_int8.float()
        else:
            out_i32 = x_q.float() @ self.w_int8.float()

        out = out_i32.float() * act_scale * self.w_row_scale.unsqueeze(0)
        out = out.to(self.dtype).reshape(*orig_shape[:-1], -1)
        if self.bias is not None:
            out = out + self.bias
        return out


def build_smoothquant_module(
    weight: torch.Tensor, bias: torch.Tensor | None, act_max_abs: torch.Tensor, alpha: float | None, dtype: torch.dtype
) -> SmoothQuantW8A8Linear:
    """``alpha=None`` builds the "no smoothing" (s=1) reference case."""
    W = weight.float()
    if alpha is None:
        s = torch.ones(W.shape[1], device=W.device)
    else:
        weight_max_abs = W.abs().amax(dim=0).clamp(min=EPS)
        s = (act_max_abs.clamp(min=EPS).pow(alpha) / weight_max_abs.pow(1 - alpha)).clamp(min=1e-4, max=1e4)

    w_smooth = W * s.unsqueeze(0)
    w_row_scale = w_smooth.abs().amax(dim=1).clamp(min=EPS) / 127.0
    w_int8 = torch.clamp(torch.round(w_smooth / w_row_scale.unsqueeze(1)), -127, 127).to(torch.int8)
    w_int8_t = w_int8.T.contiguous()

    act_scale_smoothed = (act_max_abs / s).clamp(min=EPS)
    act_scale = (act_scale_smoothed.max() / 127.0).item()

    return SmoothQuantW8A8Linear(w_int8_t, w_row_scale, 1.0 / s, act_scale, bias, dtype)


# ---------------------------------------------------------------------------
# Benchmarks.
# ---------------------------------------------------------------------------


def benchmark_gemm(tokens: int, k: int, n: int, device: str = "cuda", iters: int = 30, warmup: int = 5) -> dict:
    if not torch.cuda.is_available():
        return {"tokens": tokens, "k": k, "n": n, "bf16_ms": None, "int8_ms": None, "speedup": None}

    a_bf16 = torch.randn(tokens, k, device=device, dtype=torch.bfloat16)
    b_bf16 = torch.randn(k, n, device=device, dtype=torch.bfloat16)
    a_i8 = torch.randint(-127, 128, (tokens, k), device=device, dtype=torch.int8)
    b_i8 = torch.randint(-127, 128, (k, n), device=device, dtype=torch.int8)

    for _ in range(warmup):
        _ = a_bf16 @ b_bf16
        _ = torch._int_mm(a_i8, b_i8)
    torch.cuda.synchronize()

    with cuda_timer() as t:
        for _ in range(iters):
            _ = a_bf16 @ b_bf16
    bf16_ms = t["elapsed"] / iters * 1000

    with cuda_timer() as t:
        for _ in range(iters):
            _ = torch._int_mm(a_i8, b_i8)
    int8_ms = t["elapsed"] / iters * 1000

    return {"tokens": tokens, "k": k, "n": n, "bf16_ms": bf16_ms, "int8_ms": int8_ms, "speedup": bf16_ms / int8_ms}


def main() -> None:
    print_banner("Lab 04: SmoothQuant W8A8")
    out_dir = results_dir(__file__)

    tokenizer = load_tokenizer(MODEL)
    model = load_model(MODEL, dtype=torch.bfloat16)
    eval_ids = get_eval_ids(tokenizer, split="test")
    layer_names = [name for name, _ in named_linear_layers(model)]
    print(f"\n{len(layer_names)} target linear layers found.")

    print("\nEvaluating BF16 baseline...")
    baseline_ppl = perplexity(model, eval_ids, seq_len=SEQ_LEN)
    baseline_mem = model_memory_footprint_gb(model)
    print(f"  baseline perplexity = {baseline_ppl:.3f}, memory = {baseline_mem:.2f} GiB")

    print("\n[1/4] Collecting per-channel activation max-abs statistics...")
    stats = collect_max_abs_stats(model, tokenizer, layer_names)

    modules_by_name = dict(model.named_modules())
    originals = {name: modules_by_name[name].weight.data.clone() for name in layer_names}
    biases = {name: (modules_by_name[name].bias.data.clone() if modules_by_name[name].bias is not None else None) for name in layer_names}

    def rebuild(alpha):
        for name in layer_names:
            new_module = build_smoothquant_module(originals[name], biases[name], stats[name], alpha, torch.bfloat16)
            set_module_by_name(model, name, new_module)

    def restore():
        for name in layer_names:
            lin = nn.Linear(originals[name].shape[1], originals[name].shape[0], bias=biases[name] is not None)
            lin.weight.data = originals[name].clone()
            if biases[name] is not None:
                lin.bias.data = biases[name].clone()
            lin = lin.to(next(model.parameters()).device, torch.bfloat16)
            set_module_by_name(model, name, lin)

    print("\n[2/4] alpha sweep (quick perplexity check, no-smoothing reference included)...")
    sweep_results = []
    for label, alpha in [("no-smoothing (s=1)", None)] + [(f"alpha={a}", a) for a in ALPHA_SWEEP]:
        rebuild(alpha)
        ppl = perplexity(model, eval_ids, seq_len=SEQ_LEN, max_windows=SWEEP_MAX_WINDOWS, show_progress=False)
        print(f"    {label:>20s}  ppl(subset)={ppl:.3f}")
        sweep_results.append({"label": label, "alpha": alpha, "ppl_subset": ppl})
        restore()

    best = min((r for r in sweep_results if r["alpha"] is not None), key=lambda r: r["ppl_subset"])
    best_alpha = best["alpha"]
    print(f"  best alpha = {best_alpha}")

    print(f"\n[3/4] Building final static W8A8 model at alpha={best_alpha} and evaluating full perplexity...")
    rebuild(best_alpha)
    w8a8_ppl = perplexity(model, eval_ids, seq_len=SEQ_LEN)
    print(f"  SmoothQuant W8A8 perplexity = {w8a8_ppl:.3f}  (Δ={w8a8_ppl - baseline_ppl:+.3f})")

    sample_prompt = "The history of the Roman Empire begins with"
    print(f"  generation @ W8A8: {quick_generate(model, tokenizer, sample_prompt, 40)!r}")

    print("\n  (reference) dynamic per-batch activation quantization, evaluated on a subset for speed...")
    for name in layer_names:
        module = dict(model.named_modules())[name]
        if isinstance(module, SmoothQuantW8A8Linear):
            module.dynamic_activation = True
    dynamic_ppl = perplexity(model, eval_ids, seq_len=SEQ_LEN, max_windows=SWEEP_MAX_WINDOWS, show_progress=False)
    print(f"  dynamic-activation ppl(subset) = {dynamic_ppl:.3f}  (static ppl(subset) was ~{best['ppl_subset']:.3f})")

    print("\n[4/4] Benchmarking real INT8 GEMM vs BF16 at this model's actual layer shapes...")
    hidden = model.config.hidden_size
    intermediate = getattr(model.config, "intermediate_size", hidden * 4)
    gemm_results = []
    for tokens in (256, 2048):
        gemm_results.append(benchmark_gemm(tokens, hidden, hidden))
        gemm_results.append(benchmark_gemm(tokens, hidden, intermediate))
        print(f"    tokens={tokens}: {gemm_results[-2]}")
        print(f"    tokens={tokens}: {gemm_results[-1]}")

    save_bar_chart(
        out_dir / "01_ppl_comparison.png",
        [r["label"] for r in sweep_results] + ["W8A8 full-eval (best alpha)"],
        [r["ppl_subset"] for r in sweep_results] + [w8a8_ppl],
        ylabel="WikiText-2 perplexity",
        title="SmoothQuant: alpha sweep + full-eval result",
    )
    valid_gemm = [g for g in gemm_results if g["speedup"] is not None]
    if valid_gemm:
        save_bar_chart(
            out_dir / "02_gemm_speedup.png",
            [f"tok={g['tokens']},k={g['k']},n={g['n']}" for g in valid_gemm],
            [g["speedup"] for g in valid_gemm],
            ylabel="bf16_time / int8_time (speedup)",
            title="Real torch._int_mm INT8 GEMM speedup vs BF16",
        )

    save_json(
        out_dir / "metrics.json",
        {
            "model": MODEL,
            "baseline_ppl": baseline_ppl,
            "baseline_mem_gb": baseline_mem,
            "alpha_sweep": sweep_results,
            "best_alpha": best_alpha,
            "w8a8_ppl": w8a8_ppl,
            "dynamic_activation_ppl_subset": dynamic_ppl,
            "gemm_benchmarks": gemm_results,
        },
    )
    print(f"\nDone. Plots + metrics.json written to {out_dir}")


if __name__ == "__main__":
    main()
