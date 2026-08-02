# Quantization Learning Labs

A self-contained, hands-on curriculum for understanding **model / LLM quantization**, from
the raw bit-level numerics up to serving a multi-precision, tensor-parallel model across
multiple GPUs. Every lab is a plain Python script you run from the shell -- no notebook
server needed, so it works cleanly over SSH on a remote GPU box.

Built and intended to be run on an **8x NVIDIA B200 (183GB)** node, which is one of the
first GPU generations with native FP8 *and* FP4 tensor cores -- so labs 05/06 in particular
let you see quantization formats that are actually accelerated in hardware, not just
simulated.

## Curriculum

| Lab | Topic | What you'll learn |
|---|---|---|
| [00_fundamentals](labs/00_fundamentals) | Numerics primer | INT8/INT4/FP8/FP4 formats, symmetric vs asymmetric, per-tensor/channel/group scales, quantization error |
| [01_rtn_weight_only](labs/01_rtn_weight_only) | Round-to-nearest PTQ | Simplest weight-only PTQ baseline; granularity vs accuracy/memory trade-off |
| [02_gptq](labs/02_gptq) | GPTQ | Hessian-based layer-wise weight reconstruction, from scratch + production tool comparison |
| [03_awq](labs/03_awq) | AWQ | Activation-aware salient-channel scaling before RTN |
| [04_smoothquant](labs/04_smoothquant) | SmoothQuant | Migrating activation outliers into weights to enable true INT8 W8A8 |
| [05_fp8](labs/05_fp8) | FP8 (E4M3/E5M2) | Native Blackwell FP8 tensor cores, per-tensor vs per-row scaling |
| [06_fp4_nvfp4](labs/06_fp4_nvfp4) | FP4 / NVFP4 / MXFP4 | OCP microscaling format, Blackwell-native FP4 inference |
| [07_qat](labs/07_qat) | QAT / STE | Quantization-aware fine-tuning with a straight-through estimator |
| [08_qlora](labs/08_qlora) | QLoRA | 4-bit NF4 base + LoRA fine-tuning, memory savings |
| [09_kv_cache_quant](labs/09_kv_cache_quant) | KV-cache quantization | Quantizing the KV cache for long-context serving |
| [10_capstone_serving_bench](labs/10_capstone_serving_bench) | Capstone | Multi-GPU vLLM serving benchmark across BF16/FP8/INT4/NVFP4 |

Each lab folder has its own `README.md` with the theory, how to run it, and what to look
for in the output. Labs are numbered in the order they're meant to be done -- later labs
reuse concepts (and sometimes results) from earlier ones.

## Environment

This was designed against the environment already validated on the target node:

- 8x NVIDIA B200, 183GB each, driver 580.126.09, CUDA 13.0, sm_100
- `torch 2.11.0+cu130`, `transformers 5.14.1`, `accelerate 1.14.0`, `peft 0.20.0`,
  `datasets 5.0.1`, `deepspeed 0.19.3`, `vllm 0.26.0`, `flash-attn 2.8.3.post1`
- `HF_HOME=/raid/llm_rl_cache/hf_home` (plenty of disk on `/raid`)

A few extra packages are needed for specific labs (bitsandbytes, GPTQ/AWQ/SmoothQuant
libraries, torchao, etc). Install them with:

```bash
pip install -r requirements-extra.txt
```

You don't have to install everything up front -- each script checks what it needs on
startup (via `common/env_check.py`) and tells you exactly what's missing and how to get it.

Models default to **fully open, ungated** Qwen2.5 checkpoints, so no `HF_TOKEN` or license
click-through is required. Model size scales up across labs (0.5B for fast iteration in
labs 00/01/07, up to 32B tensor-parallel for the lab 10 capstone) -- see
`common/models.py` for the registry. If you'd rather use Llama or another gated family,
that's a one-line change there.

## Running the labs

This repo is developed locally and meant to be synced onto the GPU node, e.g.:

```bash
# on your workstation
git push origin main

# on b200-72
git clone https://github.com/venkatacrc/quantization.git
cd quantization
conda activate <your-env>       # the one with torch/transformers/vllm already installed
pip install -r requirements-extra.txt
```

Then run any lab directly:

```bash
python labs/00_fundamentals/lab00_numerics.py
python labs/01_rtn_weight_only/lab01_rtn.py
python labs/02_gptq/lab02_gptq_from_scratch.py
...
```

Each script prints progress to stdout and writes plots/tables/JSON metrics into that lab's
`results/` directory. Later labs (05+) expect a GPU; labs needing more than one GPU say so
explicitly in their README (e.g. lab 10 launches vLLM with `tensor_parallel_size > 1`).

## Running everything

`run_all.py` runs every lab in order, each as its own subprocess (so one lab's GPU memory /
vLLM processes don't leak into the next), streaming output live and also saving it to
`run_all_logs/`. It keeps going if a lab fails (e.g. a missing optional dependency) and
writes a running summary (`run_all_logs/summary.txt` / `.json`) after every lab, so an
interrupted run still leaves you a useful report.

```bash
python run_all.py                     # everything, in order (expect a few hours -- lab 10 dominates)
python run_all.py --dry-run           # see the planned run order without executing anything
python run_all.py --only 00,01,02,03  # just the from-scratch PTQ labs
python run_all.py --skip 10 --skip-production-comparisons   # skip the capstone + the optional lab02b/03b/05b
python run_all.py --from 06           # resume a run that died partway through
```

Run `python run_all.py --help` for the full set of options.

## Repo layout

```
quantization/
  run_all.py         runs every lab in order (see "Running everything" above)
  common/            shared helpers used by every lab
    quant_ops.py       quantize/dequantize primitives, STE fake-quant, FP4/FP8 simulation
    models.py          model registry + load_model()/load_tokenizer()
    data.py            wikitext2/c4 calibration + eval data loaders
    eval.py            perplexity and accuracy helpers
    bench.py           latency/throughput/GPU-memory benchmarking harness
    plotting.py        matplotlib helpers that save PNGs into a lab's results/
    env_check.py       checks GPU + optional-library availability
  labs/
    00_fundamentals/ .. 10_capstone_serving_bench/
      README.md          theory + how to run
      labXX_*.py         the runnable script(s)
      results/           generated output (gitignored contents)
  requirements-extra.txt
```

## Suggested order

Go through the labs in numeric order -- each README calls out what to compare against
earlier labs. If you only have limited GPU time, the highest-signal path is:
`00 -> 01 -> 02 -> 05 -> 06 -> 10`, and you can come back for QAT/QLoRA/KV-cache (07-09)
and AWQ/SmoothQuant (03-04) afterward.
