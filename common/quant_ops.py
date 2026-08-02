"""Core quantize/dequantize primitives shared by every lab.

This module is the "from scratch" math underneath the whole curriculum:
uniform integer quantization (symmetric/asymmetric, per-tensor/channel/group),
a straight-through-estimator (STE) fake-quant op for QAT, and simulated
FP8 / FP4 (NVFP4-style microscaling) quantization built on top of PyTorch's
native 8-bit float dtypes and a hand-rolled 4-bit float grid.

Nothing here calls out to bitsandbytes/AutoGPTQ/AutoAWQ/torchao -- those are
used later in each lab purely as a *comparison* against this from-scratch
implementation, never as a dependency of it.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

# ---------------------------------------------------------------------------
# Uniform integer quantization: scale/zero-point, symmetric or asymmetric,
# at per-tensor / per-channel / per-group granularity.
# ---------------------------------------------------------------------------


def qrange(bits: int, symmetric: bool) -> tuple[int, int]:
    """Integer grid range for a given bit-width."""
    if symmetric:
        # Symmetric signed range, e.g. bits=4 -> [-7, 7] (leave -8 unused so
        # zero maps exactly to the grid, matching most PTQ implementations).
        qmax = 2 ** (bits - 1) - 1
        return -qmax, qmax
    return 0, 2**bits - 1


def compute_qparams(
    x: torch.Tensor,
    bits: int = 8,
    symmetric: bool = True,
    dim: int | None = None,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute (scale, zero_point) from min/max statistics.

    ``dim=None`` gives one scalar scale for the whole tensor (per-tensor).
    ``dim=0`` gives one scale per row, e.g. per output-channel for a weight
    matrix of shape (out_features, in_features) -- the standard "per-channel"
    granularity. Pass an already-grouped tensor (see ``group_reshape``) with
    ``dim=-1`` to get per-group scales.
    """
    qmin, qmax = qrange(bits, symmetric)
    if dim is None:
        xmin, xmax = x.min(), x.max()
    else:
        xmin = x.amin(dim=dim, keepdim=True)
        xmax = x.amax(dim=dim, keepdim=True)

    if symmetric:
        amax = torch.maximum(xmin.abs(), xmax.abs()).clamp(min=eps)
        scale = amax / qmax
        zero_point = torch.zeros_like(scale)
    else:
        scale = ((xmax - xmin) / (qmax - qmin)).clamp(min=eps)
        zero_point = qmin - xmin / scale

    return scale, zero_point


def quantize_int(
    x: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor, bits: int, symmetric: bool
) -> torch.Tensor:
    """Quantize to the integer grid (returned as a float tensor holding
    integer values, so it composes cleanly with autograd / further math).
    """
    qmin, qmax = qrange(bits, symmetric)
    q = torch.round(x / scale + zero_point)
    return q.clamp(qmin, qmax)


def dequantize_int(q: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor) -> torch.Tensor:
    return (q - zero_point) * scale


def group_reshape(x: torch.Tensor, group_size: int) -> tuple[torch.Tensor, tuple[int, int]]:
    """Reshape a 2D (rows, cols) tensor into (rows, cols // group_size, group_size)
    so ``compute_qparams(..., dim=-1)`` gives one scale per group. Pads the
    last group with zeros if ``cols`` isn't divisible by ``group_size``
    (padding is stripped again by ``group_unreshape``).
    """
    rows, cols = x.shape
    pad = (-cols) % group_size
    if pad:
        x = torch.nn.functional.pad(x, (0, pad))
    grouped = x.reshape(rows, -1, group_size)
    return grouped, (rows, cols)


def group_unreshape(grouped: torch.Tensor, orig_shape: tuple[int, int]) -> torch.Tensor:
    rows, cols = orig_shape
    flat = grouped.reshape(rows, -1)
    return flat[:, :cols]


def fake_quantize_int(
    x: torch.Tensor,
    bits: int = 4,
    symmetric: bool = True,
    granularity: str = "per_tensor",
    group_size: int = 128,
) -> torch.Tensor:
    """Quantize-then-dequantize a 2D tensor in one call ("fake quant"), at
    the requested granularity. This is what RTN (lab 01), the fallback
    reconstruction step in GPTQ (lab 02)/AWQ (lab 03), and QAT (lab 07) all
    call under the hood.
    """
    if granularity == "per_tensor":
        scale, zp = compute_qparams(x, bits, symmetric, dim=None)
        q = quantize_int(x, scale, zp, bits, symmetric)
        return dequantize_int(q, scale, zp)

    if granularity == "per_channel":
        scale, zp = compute_qparams(x, bits, symmetric, dim=1)
        q = quantize_int(x, scale, zp, bits, symmetric)
        return dequantize_int(q, scale, zp)

    if granularity == "per_group":
        grouped, orig_shape = group_reshape(x, group_size)
        scale, zp = compute_qparams(grouped, bits, symmetric, dim=-1)
        q = quantize_int(grouped, scale, zp, bits, symmetric)
        deq = dequantize_int(q, scale, zp)
        return group_unreshape(deq, orig_shape)

    raise ValueError(f"unknown granularity: {granularity!r}")


# ---------------------------------------------------------------------------
# Straight-through estimator (STE) fake-quant, for QAT (lab 07).
# ---------------------------------------------------------------------------


class _STEFakeQuant(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, bits, symmetric, granularity, group_size):
        return fake_quantize_int(x, bits, symmetric, granularity, group_size)

    @staticmethod
    def backward(ctx, grad_output):
        # Straight-through: pretend fake_quantize_int was the identity for
        # gradient purposes, so gradients flow to the *unquantized* weight.
        return grad_output, None, None, None, None


def ste_fake_quantize(
    x: torch.Tensor,
    bits: int = 4,
    symmetric: bool = True,
    granularity: str = "per_channel",
    group_size: int = 128,
) -> torch.Tensor:
    return _STEFakeQuant.apply(x, bits, symmetric, granularity, group_size)


# ---------------------------------------------------------------------------
# FP8 (E4M3 / E5M2) quantization, using PyTorch's native 8-bit float dtypes.
# ---------------------------------------------------------------------------

_FP8_MAX = {
    "e4m3": 448.0,        # torch.float8_e4m3fn max representable magnitude
    "e5m2": 57344.0,      # torch.float8_e5m2 max representable magnitude
}
_FP8_TORCH_DTYPE = {
    "e4m3": torch.float8_e4m3fn,
    "e5m2": torch.float8_e5m2,
}


def fp8_quantize_dequantize(
    x: torch.Tensor,
    fmt: str = "e4m3",
    dim: int | None = None,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Scale ``x`` into the representable range of the chosen FP8 format,
    cast down and back up (round-trip through the real 8-bit float dtype so
    the rounding matches hardware), and return (dequantized, scale).

    ``dim=None`` -> one scale for the whole tensor; ``dim=0``/``1`` -> one
    scale per row/column (matches "per-tensor" vs "per-row" scaling as
    discussed in lab 05).
    """
    torch_dtype = _FP8_TORCH_DTYPE[fmt]
    fmax = _FP8_MAX[fmt]

    if dim is None:
        amax = x.abs().max().clamp(min=eps)
    else:
        amax = x.abs().amax(dim=dim, keepdim=True).clamp(min=eps)

    scale = amax / fmax
    x_scaled = (x / scale).to(torch_dtype)
    x_dq = x_scaled.to(x.dtype) * scale
    return x_dq, scale


# ---------------------------------------------------------------------------
# FP4 microscaling (NVFP4 / MXFP4 style), simulated in pure PyTorch.
#
# 4-bit floating point with 2 exponent bits + 1 mantissa bit (E2M1) has
# exactly 8 non-negative representable magnitudes: {0, 0.5, 1, 1.5, 2, 3, 4, 6}
# (plus sign). Real NVFP4/MXFP4 kernels share one scale factor across a small
# block of elements (16 for NVFP4, 32 for MXFP4) so the format still adapts
# to locally varying magnitudes despite only having 4 bits per value -- this
# is exactly what we simulate here.
# ---------------------------------------------------------------------------

E2M1_GRID = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
E2M1_MAX = float(E2M1_GRID.max())


def _quantize_to_grid(x: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    """Nearest-value quantization of |x| onto a fixed positive grid, sign
    applied afterwards. Grid search is done by brute-force distance (the
    grid only has 8 entries, so this is cheap) -- clearer to read than a
    closed-form E2M1 bit-packing routine, same numerical result.
    """
    grid = grid.to(x.device, x.dtype)
    sign = torch.sign(x)
    mag = x.abs().unsqueeze(-1)                       # (..., 1)
    dist = (mag - grid).abs()                         # (..., grid_size)
    idx = dist.argmin(dim=-1)
    quantized_mag = grid[idx]
    return sign * quantized_mag


def fp4_blockwise_quantize_dequantize(
    x: torch.Tensor,
    block_size: int = 16,
    power_of_two_scale: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Simulated NVFP4 (``block_size=16``, fp32/fp8-precision scale) or
    MXFP4 (``block_size=32``, ``power_of_two_scale=True`` per the OCP MX
    spec) quantization of a 2D tensor along its last dimension.

    Returns (dequantized_tensor, per-block scales).
    """
    orig_shape = x.shape
    x2d = x.reshape(-1, x.shape[-1])
    grouped, gshape = group_reshape(x2d, block_size)   # (rows, n_groups, block_size)

    amax = grouped.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
    scale = amax / E2M1_MAX
    if power_of_two_scale:
        # OCP MX spec: the shared scale is itself quantized to a power of
        # two (storable in 8 bits, no mantissa) -- coarser but even cheaper.
        scale = torch.exp2(torch.ceil(torch.log2(scale)))

    scaled = grouped / scale
    quantized = _quantize_to_grid(scaled, E2M1_GRID.to(scaled.device, scaled.dtype))
    dequantized = quantized * scale

    deq_2d = group_unreshape(dequantized, gshape)
    return deq_2d.reshape(orig_shape), scale


# ---------------------------------------------------------------------------
# NF4 ("NormalFloat4"): the 16-level, non-uniform codebook used by
# bitsandbytes / QLoRA (Dettmers et al. 2023), chosen as quantiles of a
# standard normal distribution so it's information-theoretically well
# matched to (approximately) normally distributed weight tensors -- unlike
# a uniform INT4 grid, or the exponent/mantissa-structured FP4 grid above.
# These are the exact 16 codebook values used by bitsandbytes.
# ---------------------------------------------------------------------------

NF4_GRID = torch.tensor(
    [
        -1.0,
        -0.6961928009986877,
        -0.5250730514526367,
        -0.39491748809814453,
        -0.28444138169288635,
        -0.18477343022823334,
        -0.09105003625154495,
        0.0,
        0.07958029955625534,
        0.16093020141124725,
        0.24611230194568634,
        0.33791524171829224,
        0.44070982933044434,
        0.5626170039176941,
        0.7229568362236023,
        1.0,
    ]
)


def nf4_blockwise_quantize_dequantize(x: torch.Tensor, block_size: int = 64) -> tuple[torch.Tensor, torch.Tensor]:
    """Simulated NF4 quantization of a 2D tensor along its last dimension,
    with one absmax scale per block (bitsandbytes' default block size is
    64). Unlike the FP4 grid above, NF4's codebook isn't derived from a
    sign+exponent+mantissa layout, so we do a brute-force nearest lookup
    over its 16 fixed values instead of the cheaper sign-then-magnitude
    trick used for E2M1.
    """
    orig_shape = x.shape
    x2d = x.reshape(-1, x.shape[-1])
    grouped, gshape = group_reshape(x2d, block_size)   # (rows, n_groups, block_size)

    amax = grouped.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
    scaled = grouped / amax  # normalized into [-1, 1]

    grid = NF4_GRID.to(scaled.device, scaled.dtype)
    dist = (scaled.unsqueeze(-1) - grid).abs()  # (..., 16)
    idx = dist.argmin(dim=-1)
    quantized = grid[idx] * amax

    deq_2d = group_unreshape(quantized, gshape)
    return deq_2d.reshape(orig_shape), amax


# ---------------------------------------------------------------------------
# Shared error metrics, used throughout (esp. lab 00's visualizations).
# ---------------------------------------------------------------------------


@dataclass
class QuantErrorStats:
    mse: float
    max_abs_err: float
    relative_l2_err: float  # ||x - xq|| / ||x||


def estimate_packed_size_bytes(
    numel: int, bits: int, group_size: int | None = None, scale_bits: int = 16
) -> int:
    """Estimate the storage a *real* packed low-bit kernel would use for a
    weight tensor with ``numel`` elements: ``bits`` per weight, plus one
    ``scale_bits``-bit scale per group of ``group_size`` elements (or one
    scale total if ``group_size`` is None, i.e. per-tensor).

    This repo keeps quantized weights "fake quantized" (dequantized back to
    bf16) throughout so the same `nn.Linear` keeps working without a custom
    kernel -- this function is what lets the labs still report the memory a
    *real* packed implementation (bitsandbytes/GPTQ/AWQ kernels) would use.
    """
    weight_bits = numel * bits
    n_groups = 1 if group_size is None else -(-numel // group_size)  # ceil div
    scale_bits_total = n_groups * scale_bits
    return (weight_bits + scale_bits_total) // 8


def quantization_error_stats(x: torch.Tensor, x_dequant: torch.Tensor) -> QuantErrorStats:
    diff = (x - x_dequant).float()
    mse = diff.pow(2).mean().item()
    max_abs_err = diff.abs().max().item()
    rel_l2 = (diff.norm() / x.float().norm().clamp(min=1e-12)).item()
    return QuantErrorStats(mse=mse, max_abs_err=max_abs_err, relative_l2_err=rel_l2)
