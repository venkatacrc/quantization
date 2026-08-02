"""Lab 07 - Quantization-aware training (QAT) with a straight-through
estimator.

Run: python labs/07_qat/lab07_qat_lsq.py
See README.md in this directory for background + what to look for.
"""

from __future__ import annotations

import sys
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
from common.plotting import results_dir, save_bar_chart, save_json, save_line_chart

MODEL = "tiny"  # Qwen2.5-0.5B-Instruct
BITS = 4
GRANULARITY = "per_channel"
GROUP_SIZE = 128  # unused for per_channel, kept for API symmetry

SEQ_LEN = 1024          # full eval window length (matches other labs)
TRAIN_SEQ_LEN = 256      # shorter windows for fast training steps
BATCH_SIZE = 4
N_STEPS = 300
EVAL_EVERY = 50
EVAL_SUBSET_WINDOWS = 4
LR = 1e-4


class QATLinear(nn.Module):
    """Drop-in `nn.Linear` replacement that always fake-quantizes its
    (trainable, full-precision) weight through a straight-through
    estimator before the matmul.
    """

    def __init__(self, weight: torch.Tensor, bias: torch.Tensor | None, bits: int, granularity: str, group_size: int):
        super().__init__()
        self.weight = nn.Parameter(weight.clone().float())
        self.bias = nn.Parameter(bias.clone().float()) if bias is not None else None
        self.bits = bits
        self.granularity = granularity
        self.group_size = group_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_q = qo.ste_fake_quantize(self.weight, self.bits, True, self.granularity, self.group_size)
        return F.linear(x, w_q, self.bias)


def build_train_batches(tokenizer, n_steps: int, batch_size: int, seq_len: int) -> list[torch.Tensor]:
    samples = get_calibration_samples(tokenizer, n_steps * batch_size, seq_len, dataset="wikitext2")
    return [torch.cat(samples[i * batch_size : (i + 1) * batch_size], dim=0) for i in range(n_steps)]


def main() -> None:
    print_banner("Lab 07: QAT with straight-through estimator")
    out_dir = results_dir(__file__)

    tokenizer = load_tokenizer(MODEL)
    print("\nLoading model in full precision (small model, so this is cheap and numerically simpler for training)...")
    model = load_model(MODEL, dtype=torch.float32)
    device = next(model.parameters()).device
    eval_ids = get_eval_ids(tokenizer, split="test")
    layer_names = [name for name, _ in named_linear_layers(model)]
    print(f"  {len(layer_names)} target linear layers found.")

    print("\nEvaluating full-precision baseline...")
    baseline_ppl = perplexity(model, eval_ids, seq_len=SEQ_LEN)
    print(f"  baseline perplexity = {baseline_ppl:.3f}")

    print(f"\nWrapping every target layer in QATLinear (INT{BITS}, {GRANULARITY})...")
    modules_by_name = dict(model.named_modules())
    for name in layer_names:
        module = modules_by_name[name]
        new_module = QATLinear(
            module.weight.data, module.bias.data if module.bias is not None else None, BITS, GRANULARITY, GROUP_SIZE
        ).to(device)
        set_module_by_name(model, name, new_module)

    print("\nEvaluating perplexity BEFORE any training (this is exactly a PTQ/RTN baseline at this bit-width)...")
    pretrain_ppl = perplexity(model, eval_ids, seq_len=SEQ_LEN)
    print(f"  pre-training (PTQ-equivalent) perplexity = {pretrain_ppl:.3f}  (Δ={pretrain_ppl - baseline_ppl:+.3f})")

    sample_prompt = "The history of the Roman Empire begins with"
    print(f"  generation before training: {quick_generate(model, tokenizer, sample_prompt, 40)!r}")

    print(f"\nBuilding {N_STEPS} training batches (batch_size={BATCH_SIZE}, seq_len={TRAIN_SEQ_LEN})...")
    batches = build_train_batches(tokenizer, N_STEPS, BATCH_SIZE, TRAIN_SEQ_LEN)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    loss_history = []
    ppl_curve = [{"step": 0, "ppl": pretrain_ppl}]

    print(f"\nTraining for {N_STEPS} steps...")
    model.train()
    for step, batch in enumerate(batches, start=1):
        batch = batch.to(device)
        outputs = model(batch, labels=batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        loss_history.append(loss.item())

        if step % EVAL_EVERY == 0 or step == N_STEPS:
            model.eval()
            with torch.no_grad():
                subset_ppl = perplexity(model, eval_ids, seq_len=SEQ_LEN, max_windows=EVAL_SUBSET_WINDOWS, show_progress=False)
            model.train()
            ppl_curve.append({"step": step, "ppl": subset_ppl})
            print(f"  step {step:4d}/{N_STEPS}  loss={loss.item():.4f}  ppl(subset)={subset_ppl:.3f}")

    model.eval()
    print("\nEvaluating final perplexity on the full test set...")
    qat_ppl = perplexity(model, eval_ids, seq_len=SEQ_LEN)
    print(f"  QAT (post-training) perplexity = {qat_ppl:.3f}  (Δ={qat_ppl - baseline_ppl:+.3f})")
    print(f"  generation after training: {quick_generate(model, tokenizer, sample_prompt, 40)!r}")

    save_line_chart(
        out_dir / "01_training_loss.png",
        list(range(1, len(loss_history) + 1)),
        {"train loss": loss_history},
        xlabel="step",
        ylabel="loss",
        title="QAT training loss",
    )
    save_line_chart(
        out_dir / "02_ppl_vs_step.png",
        [p["step"] for p in ppl_curve],
        {"perplexity (subset)": [p["ppl"] for p in ppl_curve]},
        xlabel="training step",
        ylabel="WikiText-2 perplexity (subset)",
        title="QAT: perplexity recovers as training progresses",
    )
    save_bar_chart(
        out_dir / "03_before_after.png",
        ["full precision", f"PTQ-equivalent INT{BITS}\n(pre-training)", f"QAT INT{BITS}\n(post-training)"],
        [baseline_ppl, pretrain_ppl, qat_ppl],
        ylabel="WikiText-2 perplexity",
        title="QAT recovers most of the PTQ accuracy gap",
    )

    save_json(
        out_dir / "metrics.json",
        {
            "model": MODEL,
            "bits": BITS,
            "granularity": GRANULARITY,
            "baseline_ppl": baseline_ppl,
            "pretrain_ppl": pretrain_ppl,
            "qat_ppl": qat_ppl,
            "loss_history": loss_history,
            "ppl_curve": ppl_curve,
        },
    )
    print(f"\nDone. Plots + metrics.json written to {out_dir}")


if __name__ == "__main__":
    main()
