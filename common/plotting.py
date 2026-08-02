"""Small matplotlib helpers so every lab saves plots the same way, into that
lab's own ``results/`` directory, without needing a display (Agg backend --
works fine over SSH with no X server).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def results_dir(lab_file: str) -> Path:
    """Given a lab script's ``__file__``, return (and create) its sibling
    ``results/`` directory.
    """
    d = Path(lab_file).resolve().parent / "results"
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_bar_chart(
    out_path: Path,
    labels: list[str],
    values: list[float],
    ylabel: str,
    title: str,
    groups: list[str] | None = None,
) -> None:
    """Simple grouped or ungrouped bar chart. If ``groups`` is given, it must
    be the same length as ``labels``/``values`` and bars are colored by group.
    """
    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 0.9), 4.5))
    x = range(len(labels))
    ax.bar(x, values)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    for i, v in enumerate(values):
        ax.text(i, v, f"{v:.3g}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_histogram(
    out_path: Path,
    data,
    bins: int,
    xlabel: str,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.hist(data, bins=bins)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_line_chart(
    out_path: Path,
    x,
    series: dict[str, list[float]],
    xlabel: str,
    ylabel: str,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for label, ys in series.items():
        ax.plot(x, ys, marker="o", label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_json(out_path: Path, obj) -> None:
    with open(out_path, "w") as f:
        json.dump(obj, f, indent=2, default=str)
    print(f"  wrote {out_path}")
