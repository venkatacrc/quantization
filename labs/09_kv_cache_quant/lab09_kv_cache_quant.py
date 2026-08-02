"""Lab 09 - KV-cache quantization.

Run: python labs/09_kv_cache_quant/lab09_kv_cache_quant.py
See README.md in this directory for background + what to look for.
"""

from __future__ import annotations

import gc
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from tqdm import tqdm

from common import quant_ops as qo
from common.data import get_eval_ids
from common.env_check import print_banner
from common.models import MODEL_REGISTRY, load_model, load_tokenizer
from common.plotting import results_dir, save_bar_chart, save_json

MODEL = "large"  # Qwen2.5-7B-Instruct
SEQ_LEN = 512      # window length for the incremental-cache perplexity check
MAX_WINDOWS = 6    # kept small: this eval does one forward-with-cache *per window*, still O(seq_len^2)-ish attention cost


# ---------------------------------------------------------------------------
# Part 1: theoretical KV-cache memory, pure arithmetic.
# ---------------------------------------------------------------------------


def theoretical_kv_cache_gb(num_layers: int, num_kv_heads: int, head_dim: int, seq_len: int, batch: int, bytes_per_elem: float) -> float:
    return 2 * num_layers * num_kv_heads * head_dim * seq_len * batch * bytes_per_elem / (1024**3)


def report_theoretical_memory(model) -> list[dict]:
    cfg = model.config
    num_layers = cfg.num_hidden_layers
    num_kv_heads = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
    head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)

    rows = []
    for seq_len in (4096, 32768, 131072):
        for label, bytes_per_elem in [("bf16", 2.0), ("int8", 1.0), ("fp8", 1.0)]:
            gb = theoretical_kv_cache_gb(num_layers, num_kv_heads, head_dim, seq_len, batch=1, bytes_per_elem=bytes_per_elem)
            rows.append({"seq_len": seq_len, "dtype": label, "gb": gb})
            print(f"    seq_len={seq_len:>7,}  {label:>5s}  {gb:7.2f} GiB  (batch=1)")
    return rows


# ---------------------------------------------------------------------------
# Part 2: a real, per-token dynamic-quantization KV cache.
# ---------------------------------------------------------------------------


def build_quantized_cache_cls():
    from transformers.cache_utils import DynamicCache

    class QuantizedKVCache(DynamicCache):
        """Quantizes every key/value tensor to `fmt` (per-token dynamic
        scale, i.e. one scale per (batch, head, token) computed from that
        token's own head_dim values) the moment it's written, and returns
        the dequantized version for attention to consume -- the rest of the
        model never sees anything but a normal float tensor.
        """

        def __init__(self, fmt: str = "int8", bits: int = 8):
            super().__init__()
            self.fmt = fmt
            self.bits = bits

        def _fake_quant(self, x: torch.Tensor) -> torch.Tensor:
            if self.fmt == "int8":
                scale, zero = qo.compute_qparams(x, bits=self.bits, symmetric=True, dim=-1)
                q = qo.quantize_int(x, scale, zero, self.bits, True)
                return qo.dequantize_int(q, scale, zero).to(x.dtype)
            deq, _ = qo.fp8_quantize_dequantize(x, fmt="e4m3", dim=-1)
            return deq.to(x.dtype)

        def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
            key_states = self._fake_quant(key_states)
            value_states = self._fake_quant(value_states)
            return super().update(key_states, value_states, layer_idx, cache_kwargs)

    return QuantizedKVCache


@torch.no_grad()
def perplexity_with_cache(model, input_ids: torch.Tensor, cache_factory, seq_len: int, max_windows: int) -> float:
    """Same sliding-window perplexity as common/eval.py, but explicitly
    passing a (possibly quantized) KV cache to every window's forward call,
    so self-attention within that window actually reads back
    quantized-and-dequantized keys/values -- see the README for why a
    single one-shot forward call is enough to exercise this.
    """
    device = next(model.parameters()).device
    n_windows = min(input_ids.numel() // seq_len, max_windows)
    nll_sum, token_count = 0.0, 0
    for i in tqdm(range(n_windows), desc="perplexity (with cache)", leave=False):
        chunk = input_ids[i * seq_len : (i + 1) * seq_len].unsqueeze(0).to(device)
        cache = cache_factory()
        outputs = model(chunk, labels=chunk, past_key_values=cache, use_cache=True)
        n_pred = chunk.shape[1] - 1
        nll_sum += outputs.loss.float().item() * n_pred
        token_count += n_pred
    return float(torch.exp(torch.tensor(nll_sum / token_count)))


# ---------------------------------------------------------------------------
# Part 3: production comparison via vLLM's native kv_cache_dtype.
# ---------------------------------------------------------------------------


def run_vllm_config(model_id: str, kv_cache_dtype: str) -> dict | None:
    try:
        from vllm import LLM, SamplingParams
    except ImportError:
        print("  vllm not installed; skipping production KV-cache comparison.")
        return None

    print(f"  starting vLLM with kv_cache_dtype={kv_cache_dtype!r}...")
    try:
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        llm = LLM(model=model_id, dtype="bfloat16", kv_cache_dtype=kv_cache_dtype, gpu_memory_utilization=0.85, max_model_len=8192)
        long_prompt = ("Summarize the following in one sentence. " + "The quick brown fox jumps over the lazy dog. " * 200)
        outputs = llm.generate([long_prompt], SamplingParams(max_tokens=64, temperature=0.0))
        elapsed = time.time() - t0
        peak_mem_gb = torch.cuda.max_memory_allocated() / (1024**3)
        result = {
            "kv_cache_dtype": kv_cache_dtype,
            "elapsed_s": elapsed,
            "peak_mem_gb": peak_mem_gb,
            "sample_output": outputs[0].outputs[0].text[:200],
        }
        del llm
        gc.collect()
        torch.cuda.empty_cache()
        return result
    except Exception as e:  # noqa: BLE001 - vLLM engine lifecycle across configs is best-effort here
        print(f"  vLLM run with kv_cache_dtype={kv_cache_dtype!r} failed: {e}")
        return None


def main() -> None:
    print_banner("Lab 09: KV-cache quantization")
    out_dir = results_dir(__file__)

    tokenizer = load_tokenizer(MODEL)
    model = load_model(MODEL, dtype=torch.bfloat16)

    print("\n[1/3] Theoretical KV-cache memory at a few context lengths...")
    theoretical_rows = report_theoretical_memory(model)

    print("\n[2/3] From-scratch per-token dynamic KV-cache quantization (accuracy check)...")
    eval_ids = get_eval_ids(tokenizer, split="test")
    cache_results = []
    try:
        QuantizedKVCache = build_quantized_cache_cls()

        from transformers.cache_utils import DynamicCache

        configs = [("bf16 (baseline)", lambda: DynamicCache()), ("int8", lambda: QuantizedKVCache("int8")), ("fp8", lambda: QuantizedKVCache("fp8"))]
        for label, factory in configs:
            ppl = perplexity_with_cache(model, eval_ids, factory, seq_len=SEQ_LEN, max_windows=MAX_WINDOWS)
            print(f"    {label:>16s}  ppl={ppl:.3f}")
            cache_results.append({"config": label, "ppl": ppl})
    except Exception as e:  # noqa: BLE001 - Cache internals can shift across transformers versions
        print(f"  custom KV-cache subclass failed on this transformers version: {e}")
        print("  skipping the accuracy check; theoretical-memory and vLLM sections are unaffected.")

    del model
    gc.collect()
    torch.cuda.empty_cache()

    print("\n[3/3] Production comparison: vLLM with kv_cache_dtype='auto' vs 'fp8'...")
    model_id = MODEL_REGISTRY[MODEL]
    vllm_results = []
    for kv_dtype in ("auto", "fp8"):
        r = run_vllm_config(model_id, kv_dtype)
        if r is not None:
            vllm_results.append(r)
            print(f"    kv_cache_dtype={kv_dtype!r}: {r['elapsed_s']:.1f}s, peak_mem={r['peak_mem_gb']:.2f} GiB")

    # --- plots -----------------------------------------------------
    seq_lens = sorted({r["seq_len"] for r in theoretical_rows})
    dtypes = ["bf16", "int8", "fp8"]
    from common.plotting import save_line_chart

    series = {dt: [next(r["gb"] for r in theoretical_rows if r["seq_len"] == s and r["dtype"] == dt) for s in seq_lens] for dt in dtypes}
    save_line_chart(
        out_dir / "01_theoretical_kv_memory.png",
        seq_lens,
        series,
        xlabel="context length (tokens)",
        ylabel="KV-cache memory (GiB, batch=1)",
        title="Theoretical KV-cache memory vs context length",
    )

    if cache_results:
        save_bar_chart(
            out_dir / "02_ppl_with_quantized_cache.png",
            [r["config"] for r in cache_results],
            [r["ppl"] for r in cache_results],
            ylabel="WikiText-2 perplexity (incremental, with cache)",
            title="Perplexity with a quantized KV cache",
        )
    if vllm_results:
        save_bar_chart(
            out_dir / "03_vllm_peak_memory.png",
            [f"kv_cache_dtype={r['kv_cache_dtype']}" for r in vllm_results],
            [r["peak_mem_gb"] for r in vllm_results],
            ylabel="peak GPU memory (GiB)",
            title="vLLM peak memory: auto vs fp8 KV cache",
        )

    save_json(
        out_dir / "metrics.json",
        {"model": MODEL, "theoretical_memory": theoretical_rows, "cache_accuracy": cache_results, "vllm_comparison": vllm_results},
    )
    print(f"\nDone. Plots + metrics.json written to {out_dir}")


if __name__ == "__main__":
    main()
