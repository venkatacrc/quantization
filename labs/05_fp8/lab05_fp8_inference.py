"""Lab 05 - FP8 quantization + real Blackwell FP8 GEMM benchmark.

Run: python labs/05_fp8/lab05_fp8_inference.py
See README.md in this directory for background + what to look for.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import torch.nn as nn

from common import quant_ops as qo
from common.bench import benchmark_generate, cuda_timer, print_results_table
from common.data import get_eval_ids
from common.env_check import print_banner
from common.eval import perplexity, quick_generate
from common.models import load_model, load_tokenizer, model_memory_footprint_gb, named_linear_layers, set_module_by_name
from common.plotting import results_dir, save_bar_chart, save_json

MODEL = "large"  # Qwen2.5-7B-Instruct
SEQ_LEN = 1024

FP8_DTYPES = {"e4m3": torch.float8_e4m3fn, "e5m2": torch.float8_e5m2}
FP8_MAX = {"e4m3": 448.0, "e5m2": 57344.0}


# ---------------------------------------------------------------------------
# Real FP8 x FP8 matmul via torch._scaled_mm, with a graceful software
# fallback if it's unavailable/unsupported on a given torch build.
# ---------------------------------------------------------------------------


class FP8W8A8Linear(nn.Module):
    def __init__(
        self,
        w_fp8_t: torch.Tensor,      # (in_features, out_features), fp8
        w_col_scale: torch.Tensor,  # (out_features,)
        bias: torch.Tensor | None,
        dtype: torch.dtype,
        fmt: str = "e4m3",
    ):
        super().__init__()
        self.register_buffer("w_fp8", w_fp8_t)
        self.register_buffer("w_col_scale", w_col_scale.to(torch.float32))
        self.bias = nn.Parameter(bias.to(dtype), requires_grad=False) if bias is not None else None
        self.dtype = dtype
        self.fp8_dtype = FP8_DTYPES[fmt]
        self.fmax = FP8_MAX[fmt]
        self._scaled_mm_ok = torch.cuda.is_available()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        x2 = x.reshape(-1, orig_shape[-1]).float()
        act_scale = (x2.abs().amax() / self.fmax).clamp(min=1e-12)
        x_fp8 = (x2 / act_scale).to(self.fp8_dtype)

        if self._scaled_mm_ok:
            try:
                scale_a = act_scale.reshape(1, 1).float()
                scale_b = self.w_col_scale.reshape(1, -1).float()
                out = torch._scaled_mm(x_fp8, self.w_fp8, scale_a=scale_a, scale_b=scale_b, out_dtype=self.dtype)
            except Exception:
                self._scaled_mm_ok = False
                out = ((x_fp8.float() * act_scale) @ (self.w_fp8.float() * self.w_col_scale.unsqueeze(0))).to(self.dtype)
        else:
            out = ((x_fp8.float() * act_scale) @ (self.w_fp8.float() * self.w_col_scale.unsqueeze(0))).to(self.dtype)

        out = out.reshape(*orig_shape[:-1], -1)
        if self.bias is not None:
            out = out + self.bias
        return out


def build_fp8_module(weight: torch.Tensor, bias: torch.Tensor | None, dtype: torch.dtype, fmt: str = "e4m3") -> FP8W8A8Linear:
    W = weight.float()
    fmax = FP8_MAX[fmt]
    w_row_scale = (W.abs().amax(dim=1).clamp(min=1e-12) / fmax)
    w_fp8 = (W / w_row_scale.unsqueeze(1)).to(FP8_DTYPES[fmt])
    return FP8W8A8Linear(w_fp8.T.contiguous(), w_row_scale, bias, dtype, fmt)


def benchmark_fp8_gemm(tokens: int, k: int, n: int, device: str = "cuda", iters: int = 30, warmup: int = 5) -> dict:
    if not torch.cuda.is_available():
        return {"tokens": tokens, "k": k, "n": n, "bf16_ms": None, "fp8_ms": None, "speedup": None}

    a_bf16 = torch.randn(tokens, k, device=device, dtype=torch.bfloat16)
    b_bf16 = torch.randn(k, n, device=device, dtype=torch.bfloat16)
    a_fp8 = torch.randn(tokens, k, device=device).to(torch.float8_e4m3fn)
    b_fp8 = torch.randn(k, n, device=device).to(torch.float8_e4m3fn)
    scale_a = torch.tensor(1.0, device=device)
    scale_b = torch.tensor(1.0, device=device)

    fp8_ok = True
    try:
        for _ in range(warmup):
            _ = torch._scaled_mm(a_fp8, b_fp8, scale_a=scale_a, scale_b=scale_b, out_dtype=torch.bfloat16)
    except Exception as e:
        print(f"    (torch._scaled_mm unavailable/unsupported here: {e}; skipping FP8 GEMM timing)")
        fp8_ok = False

    for _ in range(warmup):
        _ = a_bf16 @ b_bf16
    torch.cuda.synchronize()

    with cuda_timer() as t:
        for _ in range(iters):
            _ = a_bf16 @ b_bf16
    bf16_ms = t["elapsed"] / iters * 1000

    fp8_ms = None
    if fp8_ok:
        with cuda_timer() as t:
            for _ in range(iters):
                _ = torch._scaled_mm(a_fp8, b_fp8, scale_a=scale_a, scale_b=scale_b, out_dtype=torch.bfloat16)
        fp8_ms = t["elapsed"] / iters * 1000

    return {
        "tokens": tokens,
        "k": k,
        "n": n,
        "bf16_ms": bf16_ms,
        "fp8_ms": fp8_ms,
        "speedup": (bf16_ms / fp8_ms) if fp8_ms else None,
    }


def main() -> None:
    print_banner("Lab 05: FP8 quantization")
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

    modules_by_name = dict(model.named_modules())
    originals = {name: modules_by_name[name].weight.data.clone() for name in layer_names}
    biases = {name: (modules_by_name[name].bias.data.clone() if modules_by_name[name].bias is not None else None) for name in layer_names}

    print("\n[1/3] Weight-only FP8 sweep (E4M3/E5M2 x per-tensor/per-row)...")
    weight_only_results = []
    for fmt in ("e4m3", "e5m2"):
        for granularity, dim in (("per_tensor", None), ("per_row", 1)):
            for name in layer_names:
                module = modules_by_name[name]
                deq, _ = qo.fp8_quantize_dequantize(module.weight.data.float(), fmt=fmt, dim=dim)
                module.weight.data = deq.to(module.weight.dtype)
            ppl = perplexity(model, eval_ids, seq_len=SEQ_LEN, show_progress=False)
            label = f"FP8-{fmt.upper()}/{granularity}"
            print(f"    {label:>20s}  ppl={ppl:.3f}  (Δ={ppl - baseline_ppl:+.3f})")
            weight_only_results.append({"config": label, "ppl": ppl})
            for name in layer_names:
                modules_by_name[name].weight.data = originals[name].clone()

    print("\n[2/3] Real FP8 GEMM microbenchmark (torch._scaled_mm) vs BF16...")
    hidden = model.config.hidden_size
    intermediate = getattr(model.config, "intermediate_size", hidden * 4)
    gemm_results = []
    for tokens in (256, 2048):
        gemm_results.append(benchmark_fp8_gemm(tokens, hidden, hidden))
        gemm_results.append(benchmark_fp8_gemm(tokens, hidden, intermediate))
        print(f"    tokens={tokens}: {gemm_results[-2]}")
        print(f"    tokens={tokens}: {gemm_results[-1]}")

    print("\n[3/3] End-to-end W8A8-FP8 model (dynamic activations, no smoothing needed)...")
    for name in layer_names:
        new_module = build_fp8_module(originals[name], biases[name], torch.bfloat16, fmt="e4m3")
        set_module_by_name(model, name, new_module)

    fp8_w8a8_ppl = perplexity(model, eval_ids, seq_len=SEQ_LEN)
    print(f"  FP8 W8A8 (E4M3, no smoothing) perplexity = {fp8_w8a8_ppl:.3f}  (Δ={fp8_w8a8_ppl - baseline_ppl:+.3f})")

    sample_prompt = "The history of the Roman Empire begins with"
    print(f"  generation @ FP8 W8A8: {quick_generate(model, tokenizer, sample_prompt, 40)!r}")

    print("\n  benchmarking end-to-end generation (BF16 baseline reloaded fresh vs FP8 W8A8)...")
    bench_results = [benchmark_generate(model, tokenizer, sample_prompt, "FP8-W8A8", batch_size=8, max_new_tokens=128)]
    del model
    torch.cuda.empty_cache()
    bf16_model = load_model(MODEL, dtype=torch.bfloat16)
    bench_results.insert(0, benchmark_generate(bf16_model, tokenizer, sample_prompt, "BF16", batch_size=8, max_new_tokens=128))
    print_results_table(bench_results)

    save_bar_chart(
        out_dir / "01_weight_only_ppl.png",
        [r["config"] for r in weight_only_results],
        [r["ppl"] for r in weight_only_results],
        ylabel="WikiText-2 perplexity",
        title="Weight-only FP8: E4M3 vs E5M2, per-tensor vs per-row",
    )
    valid_gemm = [g for g in gemm_results if g["speedup"] is not None]
    if valid_gemm:
        save_bar_chart(
            out_dir / "02_fp8_gemm_speedup.png",
            [f"tok={g['tokens']},k={g['k']},n={g['n']}" for g in valid_gemm],
            [g["speedup"] for g in valid_gemm],
            ylabel="bf16_time / fp8_time (speedup)",
            title="Real torch._scaled_mm FP8 GEMM speedup vs BF16",
        )
    save_bar_chart(
        out_dir / "03_e2e_tokens_per_s.png",
        [r.label for r in bench_results],
        [r.tokens_per_s for r in bench_results],
        ylabel="decode tokens/sec",
        title="End-to-end generation throughput: BF16 vs FP8 W8A8",
    )

    save_json(
        out_dir / "metrics.json",
        {
            "model": MODEL,
            "baseline_ppl": baseline_ppl,
            "baseline_mem_gb": baseline_mem,
            "weight_only_sweep": weight_only_results,
            "gemm_benchmarks": gemm_results,
            "fp8_w8a8_ppl": fp8_w8a8_ppl,
            "e2e_benchmarks": [r.as_row() for r in bench_results],
        },
    )
    print(f"\nDone. Plots + metrics.json written to {out_dir}")


if __name__ == "__main__":
    main()
