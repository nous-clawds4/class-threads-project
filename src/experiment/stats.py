"""Publication-grade statistics for the integrity & repair study.

Three reusable primitives — a percentile **bootstrap CI** (10k resamples, seeded
so it is reproducible), the **paired Wilcoxon** signed-rank test, and
**Holm–Bonferroni** multiplicity control — plus a money-grid summary that puts
error bars and significance on the headline corroboration result.

Design notes (``docs/experiment-design.md`` §9):

* Percentile bootstrap (not SEM) so intervals never spill past [0, 1] near the
  precision/recall boundaries.
* The corroboration-vs-per-type comparison is **paired by trial** (same seed,
  same rewire, same corrupted graph), so the natural test is paired Wilcoxon on
  per-trial average precision, and the natural interval is a bootstrap of the
  per-trial AP *difference*.
* Holm–Bonferroni across the dataset family controls the family-wise error rate.

Run with ``.venv/bin/python -m src.experiment.stats`` (after the money grid).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .money import TIER_B_REWIRES, average_precision
from .util import make_rng

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------
def bootstrap_ci(
    values: Sequence[float],
    *,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    statistic: Callable[[np.ndarray], float] = np.mean,
    seed: object = "boot",
) -> Tuple[float, float, float]:
    """Percentile bootstrap ``(point, lo, hi)`` for ``statistic`` over ``values``.

    Deterministic: the resampling RNG is seeded from ``seed`` via ``make_rng``.
    Returns ``(nan, nan, nan)`` for an empty input. With a single value the CI
    collapses to that value (degenerate but well-defined).
    """
    a = np.asarray([v for v in values if v == v], dtype=float)  # drop NaNs
    if a.size == 0:
        return float("nan"), float("nan"), float("nan")
    point = float(statistic(a))
    if a.size == 1:
        return point, point, point
    rng = make_rng("bootstrap", seed, a.size, n_boot)
    idx = rng.integers(0, a.size, size=(n_boot, a.size))
    boot = np.array([statistic(a[row]) for row in idx])
    lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return point, float(lo), float(hi)


def paired_wilcoxon(a: Sequence[float], b: Sequence[float]) -> Tuple[float, float]:
    """Paired Wilcoxon signed-rank test ``(statistic, p_value)`` for ``a`` vs ``b``.

    Guards the degenerate all-zero-difference case (scipy raises): identical
    paired samples mean "no effect" -> ``p = 1.0``.
    """
    from scipy.stats import wilcoxon

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("paired_wilcoxon requires equal-length paired samples")
    if np.allclose(a, b):
        return 0.0, 1.0
    res = wilcoxon(a, b)
    return float(res.statistic), float(res.pvalue)


def holm_bonferroni(
    pvalues: Sequence[float], alpha: float = 0.05
) -> Tuple[List[bool], List[float]]:
    """Holm–Bonferroni step-down. Returns ``(rejected, adjusted_pvalues)`` aligned
    to the INPUT order."""
    p = np.asarray(pvalues, dtype=float)
    m = p.size
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(running, 1.0)
    rejected = [bool(adj[i] <= alpha) for i in range(m)]
    return rejected, [float(x) for x in adj]


# ---------------------------------------------------------------------------
# Per-trial average precision (the paired unit of analysis)
# ---------------------------------------------------------------------------
def _trial_ap(trial: pd.DataFrame, prec_col: str = "edge_precision") -> float:
    """Average precision of ONE trial's θ-sweep (a single corrupted graph)."""
    g = trial[["edge_recall", prec_col]].dropna()
    if g.empty:
        return 0.0
    return average_precision(g["edge_recall"].to_numpy(), g[prec_col].to_numpy())


def paired_trial_aps(
    df: pd.DataFrame, dataset: str, prec_col: str = "edge_precision",
    rewires: Sequence[float] = TIER_B_REWIRES,
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-trial AP for corroboration vs per-type, paired by (rewire, seed)."""
    sub = df[(df.dataset == dataset) & (df.rewire.isin(rewires))]
    corr, base = [], []
    for (rw, seed), trial in sub.groupby(["rewire", "seed"]):
        c = trial[trial.confidence == "corroboration"]
        b = trial[trial.confidence == "per_type"]
        if c.empty or b.empty:
            continue
        corr.append(_trial_ap(c, prec_col))
        base.append(_trial_ap(b, prec_col))
    return np.asarray(corr), np.asarray(base)


# ---------------------------------------------------------------------------
# Money-grid summary
# ---------------------------------------------------------------------------
def money_advantage_stats(df: pd.DataFrame, prec_col: str = "edge_precision") -> pd.DataFrame:
    """Per-dataset AP-advantage with bootstrap CI, paired Wilcoxon, Holm correction,
    and the small-n caveat (edges at the precision-maximizing corroboration point)."""
    datasets = [d for d in df.dataset.unique()]
    rows, pvals = [], []
    for ds in datasets:
        corr, base = paired_trial_aps(df, ds, prec_col)
        diff = corr - base
        point, lo, hi = bootstrap_ci(diff, seed=("adv", ds, prec_col))
        _, p = paired_wilcoxon(corr, base)
        # small-n caveat: mean edges at corroboration's highest-precision operating point
        c = df[(df.dataset == ds) & (df.confidence == "corroboration")
               & (df.rewire.isin(TIER_B_REWIRES))]
        gp = c.groupby("theta").agg(p=("edge_precision", "mean"), n=("n_added", "mean")).dropna()
        gp = gp[gp.n >= 0.5]                       # operating points that add >=1 edge
        n_at_best = float(gp.loc[gp.p.idxmax(), "n"]) if not gp.empty else 0.0
        rows.append(dict(dataset=ds, redundancy=float(df[df.dataset == ds].redundancy.iloc[0]),
                         n_trials=int(diff.size), ap_adv=point, ci_lo=lo, ci_hi=hi,
                         wilcoxon_p=p, n_edges_at_peak_prec=n_at_best))
        pvals.append(p)
    rejected, adj = holm_bonferroni(pvals)
    out = pd.DataFrame(rows)
    out["holm_p"] = adj
    out["significant"] = rejected
    return out.sort_values("redundancy").reset_index(drop=True)


def fig6_with_ci(df: pd.DataFrame, stats: pd.DataFrame, path: Path) -> None:
    """fig6, now with bootstrap-CI error bars and Holm-significance markers."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sdag = stats[stats.dataset.str.startswith("sdag:")].sort_values("redundancy")
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    yerr = np.vstack([sdag.ap_adv - sdag.ci_lo, sdag.ci_hi - sdag.ap_adv])
    ax.errorbar(sdag.redundancy, sdag.ap_adv, yerr=yerr, marker="o", color="#1b7837",
                capsize=3, label="synthetic DAG sweep (95% bootstrap CI)")
    for _, r in stats[~stats.dataset.str.startswith("sdag:")].iterrows():
        mk = "*" if r.significant else "X"
        ax.errorbar([r.redundancy], [r.ap_adv],
                    yerr=[[r.ap_adv - r.ci_lo], [r.ci_hi - r.ap_adv]],
                    marker=mk, ms=12, capsize=3, label=r.dataset, zorder=5)
    ax.axhline(0, ls=":", color="grey")
    ax.set_xlabel("graph redundancy (fraction of ancestor pairs with >1 route)")
    ax.set_ylabel("average-precision advantage\n(corroboration − per-type, Tier-B)")
    ax.set_title("Corroboration's precision gain scales with graph redundancy\n"
                 "bootstrap 95% CIs; advantage is indistinguishable from 0 on trees, "
                 "significant where redundancy exists")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    logging.basicConfig(level=logging.ERROR)
    df = pd.read_parquet("results/money.parquet")
    stats = money_advantage_stats(df, "edge_precision")
    stats_clo = money_advantage_stats(df, "edge_precision_closure")[
        ["dataset", "ap_adv", "ci_lo", "ci_hi", "holm_p", "significant"]
    ].rename(columns={c: f"{c}_closure" for c in
                      ["ap_adv", "ci_lo", "ci_hi", "holm_p", "significant"]})

    out = Path("results")
    merged = stats.merge(stats_clo, on="dataset")
    merged.to_csv(out / "stats_money.csv", index=False)
    fig6_with_ci(df, stats, out / "figures" / "fig6_redundancy_scaling.png")

    pd.set_option("display.width", 200, "display.max_columns", 30)
    print("Money-figure AP advantage (corroboration − per-type), Tier-B, paired by trial:\n")
    show = stats[["dataset", "redundancy", "n_trials", "ap_adv", "ci_lo", "ci_hi",
                  "wilcoxon_p", "holm_p", "significant", "n_edges_at_peak_prec"]]
    print(show.round(4).to_string(index=False))
    print("\nWrote results/stats_money.csv and refreshed fig6 with CI error bars.")


if __name__ == "__main__":
    main()
