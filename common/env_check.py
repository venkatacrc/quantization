"""Lightweight environment checks shared by every lab.

Each lab calls ``require(*names)`` near the top of its ``main()`` so that if
an optional dependency is missing, the user gets a clear one-line
instruction instead of a raw ``ImportError`` halfway through a long run.
"""

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass

# Maps a short logical name (used by lab scripts) -> (pip package to install, import name)
_OPTIONAL_PACKAGES: dict[str, tuple[str, str]] = {
    "bitsandbytes": ("bitsandbytes", "bitsandbytes"),
    "gptqmodel": ("gptqmodel", "gptqmodel"),
    "autoawq": ("autoawq", "awq"),
    "llmcompressor": ("llmcompressor", "llmcompressor"),
    "torchao": ("torchao", "torchao"),
    "modelopt": ("nvidia-modelopt", "modelopt"),
    "matplotlib": ("matplotlib", "matplotlib"),
    "lm_eval": ("lm-eval", "lm_eval"),
    "vllm": ("vllm", "vllm"),
    "transformer_engine": ("transformer_engine", "transformer_engine"),
}


@dataclass
class CheckResult:
    name: str
    available: bool
    detail: str


def check_cuda() -> CheckResult:
    import torch

    if not torch.cuda.is_available():
        return CheckResult("cuda", False, "torch.cuda.is_available() is False -- no GPU visible")
    n = torch.cuda.device_count()
    names = ", ".join(torch.cuda.get_device_name(i) for i in range(n))
    return CheckResult("cuda", True, f"{n} device(s): {names}")


def check_optional(name: str) -> CheckResult:
    if name not in _OPTIONAL_PACKAGES:
        raise KeyError(f"unknown optional package key: {name!r}")
    pip_name, import_name = _OPTIONAL_PACKAGES[name]
    try:
        mod = importlib.import_module(import_name)
        version = getattr(mod, "__version__", "unknown version")
        return CheckResult(name, True, f"{import_name} {version}")
    except ImportError as e:
        return CheckResult(name, False, f"not installed ({e}); pip install {pip_name}")


def require(*names: str, fatal: bool = True) -> list[CheckResult]:
    """Check a list of optional packages by logical name, print a report,
    and (if fatal) exit with a helpful message if any are missing.
    """
    results = [check_optional(n) for n in names]
    missing = [r for r in results if not r.available]
    for r in results:
        status = "OK " if r.available else "MISSING"
        print(f"  [{status}] {r.name}: {r.detail}")
    if missing and fatal:
        pip_names = ", ".join(_OPTIONAL_PACKAGES[r.name][0] for r in missing)
        print(
            f"\nThis lab needs packages that aren't installed yet.\n"
            f"Run:  pip install {pip_names}\n"
            f"(or `pip install -r requirements-extra.txt` to grab everything at once)",
            file=sys.stderr,
        )
        sys.exit(1)
    return results


def print_banner(lab_name: str) -> None:
    print("=" * 78)
    print(f"  {lab_name}")
    print("=" * 78)
    cuda = check_cuda()
    print(f"  GPU: {'OK ' if cuda.available else 'MISSING'} -- {cuda.detail}")
    print("-" * 78)


if __name__ == "__main__":
    print_banner("Environment check")
    for key in _OPTIONAL_PACKAGES:
        check_optional(key)
    require(*_OPTIONAL_PACKAGES.keys(), fatal=False)
