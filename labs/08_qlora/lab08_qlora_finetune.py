"""Lab 08 - QLoRA: NF4 4-bit base + LoRA fine-tuning.

Run: python labs/08_qlora/lab08_qlora_finetune.py
See README.md in this directory for background + what to look for.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from common import quant_ops as qo
from common.bench import peak_memory_gb, reset_peak_memory
from common.data import get_eval_ids
from common.env_check import print_banner, require
from common.eval import perplexity
from common.models import MODEL_REGISTRY, load_model, load_tokenizer, model_num_params
from common.plotting import results_dir, save_bar_chart, save_json, save_line_chart

MODEL = "large"  # Qwen2.5-7B-Instruct
SEQ_LEN = 1024

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

N_STEPS = 200
BATCH_SIZE = 2
MAX_LEN = 256
LR = 2e-4


def compare_weight_formats() -> list[dict]:
    print("\n[1/5] Real-weight INT4 vs NF4 vs NVFP4 comparison...")
    tmp_model = load_model(MODEL, dtype=torch.bfloat16, device_map="cpu")
    sample_weight = None
    sample_name = None
    for name, p in tmp_model.named_parameters():
        if "down_proj.weight" in name:
            sample_weight = p.detach().float().clone()
            sample_name = name
            break
    del tmp_model

    configs = {
        "INT4 (g=64)": lambda w: qo.fake_quantize_int(w, bits=4, symmetric=True, granularity="per_group", group_size=64),
        "NF4 (blk=64)": lambda w: qo.nf4_blockwise_quantize_dequantize(w, block_size=64)[0],
        "NVFP4 (blk=16)": lambda w: qo.fp4_blockwise_quantize_dequantize(w, block_size=16, power_of_two_scale=False)[0],
    }
    results = []
    print(f"  using {sample_name} ({tuple(sample_weight.shape)})")
    for label, fn in configs.items():
        deq = fn(sample_weight)
        stats = qo.quantization_error_stats(sample_weight, deq)
        print(f"    {label:>16s}  rel_l2_err={stats.relative_l2_err:.4f}  mse={stats.mse:.3e}")
        results.append({"config": label, **stats.__dict__})
    return results


def format_alpaca_example(ex: dict) -> str:
    if ex.get("input"):
        return (
            f"### Instruction:\n{ex['instruction']}\n\n### Input:\n{ex['input']}\n\n### Response:\n{ex['output']}"
        )
    return f"### Instruction:\n{ex['instruction']}\n\n### Response:\n{ex['output']}"


def load_alpaca_texts(n_examples: int) -> list[str]:
    from datasets import load_dataset

    ds = load_dataset("tatsu-lab/alpaca", split="train").select(range(n_examples))
    return [format_alpaca_example(ex) for ex in ds]


@torch.no_grad()
def generate_completion(model, tokenizer, instruction: str, max_new_tokens: int = 60) -> str:
    prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.pad_token_id)
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)


def estimate_full_finetune_memory_gb(n_params: int) -> float:
    """Rough estimate: bf16 weights (2B) + bf16 grads (2B) + fp32 Adam
    moments m,v (4B + 4B) = 12 bytes/param, ignoring activations -- a lower
    bound, since real full fine-tuning also needs activation memory that
    scales with batch size/sequence length, which this estimate skips.
    """
    return n_params * 12 / (1024**3)


def main() -> None:
    print_banner("Lab 08: QLoRA (NF4 + LoRA)")
    require("bitsandbytes")
    out_dir = results_dir(__file__)

    weight_format_results = compare_weight_formats()

    print("\n[2/5] Loading the model in 4-bit (NF4, double-quantized) with BitsAndBytesConfig...")
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    model_id = MODEL_REGISTRY[MODEL]
    tokenizer = load_tokenizer(MODEL)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    reset_peak_memory()
    model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map="auto")
    load_mem_gb = peak_memory_gb()
    n_params = model_num_params(model)
    bf16_estimate_gb = n_params * 2 / (1024**3)
    print(f"  4-bit load: {load_mem_gb:.2f} GiB actual GPU memory  vs  ~{bf16_estimate_gb:.2f} GiB estimated for bf16")

    print("\n[3/5] Wrapping with peft LoRA adapters (prepare_model_for_kbit_training + LoraConfig)...")
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.3f}%)")

    device = next(model.parameters()).device
    sample_instruction = "Give three tips for staying productive while working from home."
    print(f"\n  completion BEFORE fine-tuning: {generate_completion(model, tokenizer, sample_instruction)!r}")

    print(f"\n[4/5] Fine-tuning on {N_STEPS * BATCH_SIZE} Alpaca examples ({N_STEPS} steps, batch_size={BATCH_SIZE})...")
    texts = load_alpaca_texts(N_STEPS * BATCH_SIZE)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)

    reset_peak_memory()
    model.train()
    loss_history = []
    for step in range(N_STEPS):
        batch_texts = texts[step * BATCH_SIZE : (step + 1) * BATCH_SIZE]
        enc = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LEN)
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = out.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        loss_history.append(loss.item())

        if (step + 1) % 25 == 0:
            print(f"  step {step + 1:4d}/{N_STEPS}  loss={loss.item():.4f}")

    train_peak_mem_gb = peak_memory_gb()
    model.eval()
    print(f"  peak training memory: {train_peak_mem_gb:.2f} GiB")
    print(f"\n  completion AFTER fine-tuning: {generate_completion(model, tokenizer, sample_instruction)!r}")

    print("\n[5/5] Evaluating WikiText-2 perplexity (sanity check, not the main point of this lab)...")
    eval_ids = get_eval_ids(tokenizer, split="test")
    ppl = perplexity(model, eval_ids, seq_len=SEQ_LEN)
    print(f"  perplexity after QLoRA fine-tuning = {ppl:.3f}")

    full_ft_estimate_gb = estimate_full_finetune_memory_gb(n_params)
    print(
        f"\n  estimated full-precision full fine-tune memory (weights+grads+Adam, no activations): "
        f"~{full_ft_estimate_gb:.1f} GiB, vs {train_peak_mem_gb:.2f} GiB actually observed for QLoRA here."
    )

    save_bar_chart(
        out_dir / "01_weight_format_error.png",
        [r["config"] for r in weight_format_results],
        [r["relative_l2_err"] for r in weight_format_results],
        ylabel="relative L2 error",
        title="Real weight tensor: INT4 vs NF4 vs NVFP4 (each at its typical block size)",
    )
    save_bar_chart(
        out_dir / "02_memory_footprint.png",
        ["bf16 estimate", "4-bit NF4 (actual)", f"full FT estimate\n(bf16, {n_params/1e9:.1f}B params)", "QLoRA train (actual)"],
        [bf16_estimate_gb, load_mem_gb, full_ft_estimate_gb, train_peak_mem_gb],
        ylabel="GPU memory (GiB)",
        title="QLoRA memory footprint vs bf16 loading / full fine-tuning estimates",
    )
    save_line_chart(
        out_dir / "03_training_loss.png",
        list(range(1, len(loss_history) + 1)),
        {"train loss": loss_history},
        xlabel="step",
        ylabel="loss",
        title="QLoRA training loss",
    )

    save_json(
        out_dir / "metrics.json",
        {
            "model": MODEL,
            "weight_format_comparison": weight_format_results,
            "load_mem_gb": load_mem_gb,
            "bf16_estimate_gb": bf16_estimate_gb,
            "n_params": n_params,
            "trainable_params": trainable,
            "train_peak_mem_gb": train_peak_mem_gb,
            "full_ft_estimate_gb": full_ft_estimate_gb,
            "loss_history": loss_history,
            "final_ppl": ppl,
        },
    )
    print(f"\nDone. Plots + metrics.json written to {out_dir}")


if __name__ == "__main__":
    main()
