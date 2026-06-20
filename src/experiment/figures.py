"""Figures for the integrity & repair study (reads ``results/results.parquet``).

Run with ``python -m src.experiment.figures`` after the experiment runner.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

RESULTS = Path("results")


def _mean_ci(values) -> Tuple[float, float, float]:
    """Mean and 95% normal-approx CI of the mean (NaNs dropped)."""
    a = np.asarray(values, dtype=float)
    a = a[~np.isnan(a)]
    if a.size == 0:
        return float("nan"), float("nan"), float("nan")
    m = a.mean()
    sem = a.std(ddof=1) / np.sqrt(a.size) if a.size > 1 else 0.0
    return m, m - 1.96 * sem, m + 1.96 * sem


def fig1_coverage(df: pd.DataFrame, path: Path) -> None:
    datasets = list(df.dataset.unique())
    arms = list(df.arm.unique())
    series = [
        ("cov_corrupt", "dual, no repair", "#d62728"),
        ("cov_repair", "dual + repair", "#2ca02c"),
        ("cov_single", "single-node, no repair", "#1f77b4"),
    ]
    fig, axes = plt.subplots(len(datasets), len(arms),
                             figsize=(4 * len(arms), 3.2 * len(datasets)),
                             squeeze=False, sharex=True, sharey=True)
    for r, ds in enumerate(datasets):
        for c, arm in enumerate(arms):
            ax = axes[r][c]
            sub = df[(df.dataset == ds) & (df.arm == arm)]
            rhos = sorted(sub.rho.unique())
            for col, label, color in series:
                stats = [_mean_ci(sub[sub.rho == x][col]) for x in rhos]
                means = [s[0] for s in stats]
                lo = [s[1] for s in stats]
                hi = [s[2] for s in stats]
                ax.plot(rhos, means, marker="o", ms=3, label=label, color=color)
                ax.fill_between(rhos, lo, hi, alpha=0.15, color=color)
            if r == 0:
                ax.set_title(arm)
            if c == 0:
                ax.set_ylabel(f"{ds}\ncoverage")
            if r == len(datasets) - 1:
                ax.set_xlabel("corruption rate ρ")
            ax.set_ylim(-0.02, 1.05)
            ax.grid(alpha=0.3)
    axes[0][0].legend(fontsize=7, loc="lower left")
    fig.suptitle("Coverage degradation & recovery vs corruption rate", y=1.0)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def fig5_detection(df: pd.DataFrame, path: Path, rho: float = 0.3) -> None:
    datasets = list(df.dataset.unique())
    arms = list(df.arm.unique())
    fig, axes = plt.subplots(1, len(datasets), figsize=(5 * len(datasets), 3.4),
                             squeeze=False, sharey=True)
    for c, ds in enumerate(datasets):
        ax = axes[0][c]
        sub = df[(df.dataset == ds) & (np.isclose(df.rho, rho))]
        fn = [sub[sub.arm == a].single_fn.mean() for a in arms]
        fp = [sub[sub.arm == a].single_fp.mean() for a in arms]
        x = np.arange(len(arms))
        w = 0.38
        ax.bar(x - w / 2, fn, w, label="single-node FN (missed real break)", color="#d62728")
        ax.bar(x + w / 2, fp, w, label="single-node FP (false alarm)", color="#ff7f0e")
        ax.set_xticks(x)
        ax.set_xticklabels(arms)
        ax.set_title(ds)
        ax.set_ylabel("pairs (mean per trial)")
        ax.grid(alpha=0.3, axis="y")
    axes[0][0].legend(fontsize=8)
    fig.suptitle(f"Single-node detection errors vs the dual validator (ρ={rho})\n"
                 f"dual validator is exact by construction; single-node is blind to hasExtension breaks")
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    df = pd.read_parquet(RESULTS / "results.parquet")
    figs = RESULTS / "figures"
    figs.mkdir(exist_ok=True, parents=True)
    fig1_coverage(df, figs / "fig1_coverage.png")
    fig5_detection(df, figs / "fig5_detection.png")
    print(f"Wrote {figs/'fig1_coverage.png'} and {figs/'fig5_detection.png'}")


if __name__ == "__main__":
    main()
