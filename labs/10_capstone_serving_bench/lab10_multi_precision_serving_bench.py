"""Lab 10 - Capstone: multi-precision, multi-GPU vLLM serving benchmark.

Run: python labs/10_capstone_serving_bench/lab10_multi_precision_serving_bench.py
See README.md in this directory for background, expected runtime, and how
to scope this down.
"""

from __future__ import annotations

import gc
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from common.data import get_raw_text
from common.env_check import print_banner
from common.models import MODEL_REGISTRY, load_tokenizer
from common.plotting import results_dir, save_bar_chart, save_json, save_line_chart

MODEL_KEY = "capstone_fallback"  # Qwen2.5-14B-Instruct; set to "capstone" for the full 32B model
TP_SIZE = 4                       # number of GPUs for tensor parallelism; raise toward 8 if you want
MAX_MODEL_LEN = 4096

CONCURRENCIES = [1, 8, 32]
PROMPT_TARGET_TOKENS = 512
MAX_NEW_TOKENS = 128

N_QUALITY_PROMPTS = 8
QUALITY_PROMPT_TOKENS = 256

PRECISION_CONFIGS = [
    {"label": "BF16", "quantization": None},
    {"label": "FP8", "quantization": "fp8"},
    {"label": "INT4 (bitsandbytes)", "quantization": "bitsandbytes"},
    {"label": "NVFP4 (modelopt, best-effort)", "quantization": "modelopt_fp4"},
]


def read_gpu_memory_used_mb(n_gpus: int) -> float:
    """Sum of `nvidia-smi`-reported memory.used across the first `n_gpus`
    GPUs -- works regardless of whether vLLM's tensor-parallel workers run
    as separate processes (which they do), unlike `torch.cuda.*` stats
    queried from this driver process alone.
    """
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True
        )
        values = [float(x) for x in out.strip().splitlines()]
        return sum(values[:n_gpus])
    except Exception as e:  # noqa: BLE001
        print(f"    (couldn't read nvidia-smi memory: {e})")
        return float("nan")


def build_quality_prompts(tokenizer, n_prompts: int, n_tokens: int) -> list[str]:
    text = get_raw_text("test")
    enc = tokenizer(text, return_tensors="pt")
    ids = enc["input_ids"][0]
    step = ids.numel() // n_prompts
    prompts = []
    for i in range(n_prompts):
        start = i * step
        prompts.append(tokenizer.decode(ids[start : start + n_tokens]))
    return prompts


def vllm_perplexity(llm, quality_prompts: list[str]) -> float:
    from vllm import SamplingParams

    params = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=0)
    outputs = llm.generate(quality_prompts, params)

    total_logprob, total_tokens = 0.0, 0
    for out in outputs:
        for entry in out.prompt_logprobs[1:]:  # first token has no logprob (no context yet)
            if entry is None:
                continue
            logprob_obj = next(iter(entry.values()))
            total_logprob += logprob_obj.logprob
            total_tokens += 1
    return float(torch.exp(torch.tensor(-total_logprob / max(total_tokens, 1))))


def benchmark_concurrency(llm, prompt: str, concurrency: int, max_new_tokens: int) -> dict:
    from vllm import SamplingParams

    params = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)
    prompts = [prompt] * concurrency
    t0 = time.time()
    outputs = llm.generate(prompts, params)
    elapsed = time.time() - t0
    total_new_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    return {
        "concurrency": concurrency,
        "elapsed_s": elapsed,
        "total_new_tokens": total_new_tokens,
        "tokens_per_s": total_new_tokens / elapsed,
    }


def run_one_precision(model_id: str, cfg: dict, quality_prompts: list[str], long_prompt: str) -> dict:
    from vllm import LLM

    label, quantization = cfg["label"], cfg["quantization"]
    print(f"\n=== {label} (quantization={quantization!r}) ===")
    try:
        t0 = time.time()
        llm = LLM(
            model=model_id,
            tensor_parallel_size=TP_SIZE,
            dtype="bfloat16",
            quantization=quantization,
            max_model_len=MAX_MODEL_LEN,
            gpu_memory_utilization=0.85,
        )
        load_s = time.time() - t0
    except Exception as e:  # noqa: BLE001 - especially expected for the best-effort NVFP4 config
        print(f"  failed to start vLLM with quantization={quantization!r}: {e}")
        print("  skipping this precision.")
        return {"label": label, "quantization": quantization, "error": str(e)}

    mem_mb = read_gpu_memory_used_mb(TP_SIZE)
    print(f"  loaded in {load_s:.1f}s, GPU memory used across {TP_SIZE} GPU(s): {mem_mb / 1024:.1f} GiB")

    try:
        ppl = vllm_perplexity(llm, quality_prompts)
        print(f"  approx. perplexity (vLLM prompt_logprobs): {ppl:.3f}")
    except Exception as e:  # noqa: BLE001
        print(f"  quality check failed: {e}")
        ppl = None

    concurrency_results = []
    for c in CONCURRENCIES:
        try:
            r = benchmark_concurrency(llm, long_prompt, c, MAX_NEW_TOKENS)
            concurrency_results.append(r)
            print(f"    concurrency={c:>3d}  elapsed={r['elapsed_s']:.2f}s  aggregate_tokens/s={r['tokens_per_s']:.1f}")
        except Exception as e:  # noqa: BLE001
            print(f"    concurrency={c} failed: {e}")

    result = {
        "label": label,
        "quantization": quantization,
        "load_s": load_s,
        "mem_gb": mem_mb / 1024,
        "ppl": ppl,
        "concurrency_results": concurrency_results,
    }

    del llm
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(5)  # best-effort settling time for multi-process TP workers to release GPU memory
    return result


def main() -> None:
    print_banner("Lab 10: Capstone multi-precision serving benchmark")
    out_dir = results_dir(__file__)

    model_id = MODEL_REGISTRY[MODEL_KEY]
    print(f"\nModel: {model_id}  |  tensor_parallel_size={TP_SIZE}")
    tokenizer = load_tokenizer(MODEL_KEY)

    print("\nBuilding quality-check prompts from WikiText-2...")
    quality_prompts = build_quality_prompts(tokenizer, N_QUALITY_PROMPTS, QUALITY_PROMPT_TOKENS)
    long_prompt = quality_prompts[0][: PROMPT_TARGET_TOKENS * 4]  # rough char-length proxy for ~PROMPT_TARGET_TOKENS tokens

    all_results = []
    for cfg in PRECISION_CONFIGS:
        result = run_one_precision(model_id, cfg, quality_prompts, long_prompt)
        all_results.append(result)
        save_json(out_dir / "metrics.json", {"model": model_id, "tp_size": TP_SIZE, "results": all_results})

    successful = [r for r in all_results if "error" not in r]
    if not successful:
        print("\nEvery precision failed to start -- nothing to plot. Check the errors above (likely a vLLM/model config issue).")
        return

    save_bar_chart(
        out_dir / "01_memory_gb.png",
        [r["label"] for r in successful],
        [r["mem_gb"] for r in successful],
        ylabel=f"GPU memory across {TP_SIZE} GPUs (GiB)",
        title="Capstone: memory footprint by precision",
    )
    ppl_rows = [r for r in successful if r.get("ppl") is not None]
    if ppl_rows:
        save_bar_chart(
            out_dir / "02_quality_ppl.png",
            [r["label"] for r in ppl_rows],
            [r["ppl"] for r in ppl_rows],
            ylabel="approx. WikiText-2 perplexity (vLLM)",
            title="Capstone: quality by precision",
        )

    throughput_series = {}
    for r in successful:
        if r["concurrency_results"]:
            throughput_series[r["label"]] = [cr["tokens_per_s"] for cr in r["concurrency_results"]]
    if throughput_series:
        concurrencies_used = [cr["concurrency"] for cr in successful[0]["concurrency_results"]] if successful[0]["concurrency_results"] else CONCURRENCIES
        save_line_chart(
            out_dir / "03_throughput_vs_concurrency.png",
            concurrencies_used,
            throughput_series,
            xlabel="concurrency (simultaneous requests)",
            ylabel="aggregate tokens/sec",
            title="Capstone: throughput vs concurrency, by precision",
        )

    print(f"\nDone. Plots + metrics.json written to {out_dir}")
    print("\nThis is the end of the curriculum -- you've gone from raw bit-level")
    print("numerics (lab 00) to serving a multi-precision model across multiple")
    print("B200s (this lab). Nice work.")


if __name__ == "__main__":
    main()
