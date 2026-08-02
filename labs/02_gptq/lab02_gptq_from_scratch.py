"""Lab 02 - GPTQ from scratch (Hessian-based layer-wise reconstruction).

Run: python labs/02_gptq/lab02_gptq_from_scratch.py
See README.md in this directory for background + what to look for.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from common import quant_ops as qo
from common.data import get_calibration_samples, get_eval_ids
from common.env_check import print_banner
from common.eval import perplexity, quick_generate
from common.models import load_model, load_tokenizer, named_linear_layers
from common.plotting import results_dir, save_bar_chart, save_json

MODEL = "small"  # Qwen2.5-1.5B-Instruct
SEQ_LEN = 1024
N_CALIB_SAMPLES = 128
CALIB_SEQ_LEN = 512
BITS = 4
GROUP_SIZE = 128
PERCDAMP = 0.01


# ---------------------------------------------------------------------------
# Step 1: collect per-layer Hessians H = X^T X from calibration activations.
# ---------------------------------------------------------------------------


class HessianAccumulator:
    def __init__(self, in_features: int, device):
        self.H = torch.zeros(in_features, in_features, device=device, dtype=torch.float32)
        self.n_tokens = 0

    def update(self, x: torch.Tensor) -> None:
        x = x.reshape(-1, x.shape[-1]).float()
        if self.H.device != x.device:
            self.H = self.H.to(x.device)
        self.H += x.T @ x
        self.n_tokens += x.shape[0]


def collect_hessians(model, tokenizer, layer_names: list[str]) -> dict[str, HessianAccumulator]:
    device = next(model.parameters()).device
    accumulators: dict[str, HessianAccumulator] = {}
    modules_by_name = dict(model.named_modules())

    hooks = []

    def make_hook(name):
        def hook(module, inputs, output):
            x = inputs[0]
            if name not in accumulators:
                accumulators[name] = HessianAccumulator(module.in_features, device)
            accumulators[name].update(x)

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
    return accumulators


# ---------------------------------------------------------------------------
# Step 2: the GPTQ column-by-column reconstruction itself.
# ---------------------------------------------------------------------------


@torch.no_grad()
def gptq_quantize_weight(
    weight: torch.Tensor,
    H: torch.Tensor,
    bits: int = 4,
    group_size: int = 128,
    symmetric: bool = True,
    percdamp: float = 0.01,
) -> torch.Tensor:
    """Quantize a (out_features, in_features) weight matrix given its
    calibration Hessian ``H`` (in_features x in_features), following the
    GPTQ algorithm (Frantar et al. 2022). Returns the dequantized
    (fake-quantized) weight, same shape/dtype semantics as the input.
    """
    W = weight.clone().float()
    rows, columns = W.shape
    H = H.clone()

    # Columns with zero calibration signal (e.g. unused padding) would make
    # H singular; give them an identity contribution so the Cholesky step
    # below doesn't blow up. Their quantization error doesn't matter since
    # they never fire.
    dead = torch.diag(H) == 0
    H[dead, dead] = 1.0
    W[:, dead] = 0.0

    damp = percdamp * torch.mean(torch.diag(H))
    diag_idx = torch.arange(columns, device=H.device)
    H[diag_idx, diag_idx] += damp

    # Two-step Cholesky trick (see README): get the upper-triangular
    # Cholesky factor of H^-1 directly, which is what makes this stable
    # enough to run on full LLM-sized layers.
    L = torch.linalg.cholesky(H)
    Hinv_full = torch.cholesky_inverse(L)
    Hinv = torch.linalg.cholesky(Hinv_full, upper=True)

    Q = torch.zeros_like(W)
    scale = zero = None

    for col in range(columns):
        if col % group_size == 0:
            group_end = min(col + group_size, columns)
            scale, zero = qo.compute_qparams(W[:, col:group_end], bits, symmetric, dim=1)

        w_col = W[:, col : col + 1]
        q_col = qo.quantize_int(w_col, scale, zero, bits, symmetric)
        dq_col = qo.dequantize_int(q_col, scale, zero)
        Q[:, col] = dq_col.squeeze(1)

        d = Hinv[col, col]
        err = (w_col.squeeze(1) - dq_col.squeeze(1)) / d
        if col < columns - 1:
            W[:, col + 1 :] -= err.unsqueeze(1) * Hinv[col, col + 1 :].unsqueeze(0)

    return Q.to(weight.dtype)


def main() -> None:
    print_banner("Lab 02: GPTQ from scratch")
    out_dir = results_dir(__file__)

    tokenizer = load_tokenizer(MODEL)
    model = load_model(MODEL, dtype=torch.bfloat16)
    eval_ids = get_eval_ids(tokenizer, split="test")
    layer_names = [name for name, _ in named_linear_layers(model)]
    print(f"\n{len(layer_names)} target linear layers found.")

    print("\nEvaluating BF16 baseline...")
    baseline_ppl = perplexity(model, eval_ids, seq_len=SEQ_LEN)
    print(f"  baseline perplexity = {baseline_ppl:.3f}")

    print("\n[1/3] Collecting per-layer Hessians from calibration data...")
    t0 = time.time()
    accumulators = collect_hessians(model, tokenizer, layer_names)
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

    print(f"\n[3/3] GPTQ from-scratch reconstruction at INT{BITS}/group_size={GROUP_SIZE}...")
    t0 = time.time()
    layer_errors = []
    for i, name in enumerate(layer_names):
        module = modules_by_name[name]
        H = accumulators[name].H / max(accumulators[name].n_tokens, 1)
        quantized = gptq_quantize_weight(module.weight.data, H, bits=BITS, group_size=GROUP_SIZE, percdamp=PERCDAMP)
        rel_err = ((module.weight.data.float() - quantized.float()).norm() / module.weight.data.float().norm()).item()
        layer_errors.append({"layer": name, "rel_l2_err": rel_err})
        module.weight.data = quantized
        del accumulators[name]  # free the (possibly large) Hessian as we go
        if (i + 1) % 32 == 0:
            print(f"    quantized {i + 1}/{len(layer_names)} layers")
    gptq_elapsed = time.time() - t0
    print(f"  done in {gptq_elapsed:.1f}s")

    gptq_ppl = perplexity(model, eval_ids, seq_len=SEQ_LEN)
    print(f"  GPTQ (from scratch) perplexity = {gptq_ppl:.3f}  (Δ={gptq_ppl - baseline_ppl:+.3f})")

    sample_prompt = "The history of the Roman Empire begins with"
    print(f"\n  generation @ GPTQ-from-scratch: {quick_generate(model, tokenizer, sample_prompt, 40)!r}")

    layer_errors.sort(key=lambda r: r["rel_l2_err"], reverse=True)
    print("\n  worst 5 layers by reconstruction error:")
    for r in layer_errors[:5]:
        print(f"    {r['layer']:>40s}  rel_l2_err={r['rel_l2_err']:.4f}")

    save_bar_chart(
        out_dir / "01_ppl_comparison.png",
        ["BF16", f"RTN INT{BITS}/g{GROUP_SIZE}", f"GPTQ INT{BITS}/g{GROUP_SIZE}"],
        [baseline_ppl, rtn_ppl, gptq_ppl],
        ylabel="WikiText-2 perplexity",
        title="RTN vs GPTQ-from-scratch at the same bit budget",
    )
    save_bar_chart(
        out_dir / "02_worst_layers.png",
        [r["layer"].replace("model.layers.", "L") for r in layer_errors[:10]],
        [r["rel_l2_err"] for r in layer_errors[:10]],
        ylabel="relative L2 reconstruction error",
        title="GPTQ: 10 hardest-to-quantize layers",
    )

    save_json(
        out_dir / "metrics.json",
        {
            "model": MODEL,
            "bits": BITS,
            "group_size": GROUP_SIZE,
            "baseline_ppl": baseline_ppl,
            "rtn_ppl": rtn_ppl,
            "gptq_from_scratch_ppl": gptq_ppl,
            "gptq_from_scratch_seconds": gptq_elapsed,
            "layer_errors": layer_errors,
        },
    )
    print(f"\nDone. Plots + metrics.json written to {out_dir}")
    print("Next: run lab02b_compare_production_gptq.py to compare against a production GPTQ library.")


if __name__ == "__main__":
    main()
