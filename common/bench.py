"""Latency / throughput / GPU-memory benchmarking harness shared by every lab.

Used by labs 01, 04, 05, 06 and 10 so that "how fast/how much memory" numbers
are measured the same way across every precision we compare.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field

import torch


@dataclass
class BenchResult:
    label: str
    batch_size: int
    prompt_tokens: int
    new_tokens: int
    ttft_s: float           # time to first token
    total_s: float          # total generation wall-clock time
    tokens_per_s: float      # decode throughput (new tokens / (total - ttft))
    peak_mem_gb: float
    extra: dict = field(default_factory=dict)

    def as_row(self) -> dict:
        return {
            "label": self.label,
            "batch_size": self.batch_size,
            "prompt_tokens": self.prompt_tokens,
            "new_tokens": self.new_tokens,
            "ttft_s": round(self.ttft_s, 4),
            "total_s": round(self.total_s, 4),
            "tokens_per_s": round(self.tokens_per_s, 2),
            "peak_mem_gb": round(self.peak_mem_gb, 3),
            **self.extra,
        }


@contextmanager
def cuda_timer():
    """Context manager yielding an elapsed-seconds getter, using CUDA events
    when available for accurate GPU timing (falls back to wall clock on CPU).
    """
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    result = {"elapsed": 0.0}
    yield result
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    result["elapsed"] = time.perf_counter() - start


def reset_peak_memory(device: int | str | torch.device = 0) -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)


def peak_memory_gb(device: int | str | torch.device = 0) -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated(device) / (1024**3)


@torch.no_grad()
def benchmark_generate(
    model,
    tokenizer,
    prompt: str,
    label: str,
    batch_size: int = 1,
    max_new_tokens: int = 128,
    device: int | str | torch.device = 0,
) -> BenchResult:
    """Benchmark one decode pass: time-to-first-token + steady-state
    tokens/sec + peak GPU memory, for a given (already loaded) model.
    """
    model_device = next(model.parameters()).device
    inputs = tokenizer([prompt] * batch_size, return_tensors="pt", padding=True).to(model_device)
    prompt_tokens = inputs["input_ids"].shape[1]

    reset_peak_memory(device)

    # Time-to-first-token: a single forward pass over the prompt.
    with cuda_timer() as t_ttft:
        _ = model(**inputs)
    ttft = t_ttft["elapsed"]

    with cuda_timer() as t_total:
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    total = t_total["elapsed"]

    new_tokens = (out.shape[1] - prompt_tokens) * batch_size
    decode_time = max(total - ttft, 1e-6)
    tokens_per_s = new_tokens / decode_time

    return BenchResult(
        label=label,
        batch_size=batch_size,
        prompt_tokens=prompt_tokens,
        new_tokens=new_tokens,
        ttft_s=ttft,
        total_s=total,
        tokens_per_s=tokens_per_s,
        peak_mem_gb=peak_memory_gb(device),
    )


def print_results_table(results: list[BenchResult]) -> None:
    rows = [r.as_row() for r in results]
    if not rows:
        return
    cols = list(rows[0].keys())
    widths = {c: max(len(c), *(len(str(r[c])) for r in rows)) for c in cols}
    header = "  ".join(c.ljust(widths[c]) for c in cols)
    print(header)
    print("-" * len(header))
    for r in rows:
        print("  ".join(str(r[c]).ljust(widths[c]) for c in cols))
