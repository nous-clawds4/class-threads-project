"""Tests for the statistics primitives and the money-grid AP-advantage summary."""
import numpy as np
import pandas as pd
import pytest

from src.experiment import stats as S


# --- bootstrap CI ----------------------------------------------------------
def test_bootstrap_ci_deterministic_and_brackets_point():
    rng = np.random.default_rng(0)
    vals = rng.normal(0.5, 0.1, 200)
    a = S.bootstrap_ci(vals, seed="x")
    b = S.bootstrap_ci(vals, seed="x")
    assert a == b                                  # seeded -> reproducible
    point, lo, hi = a
    assert lo <= point <= hi
    assert point == pytest.approx(float(vals.mean()))
    assert lo >= vals.min() and hi <= vals.max()   # percentile CI stays in range


def test_bootstrap_ci_edge_cases():
    assert all(np.isnan(x) for x in S.bootstrap_ci([]))
    assert S.bootstrap_ci([0.7]) == (0.7, 0.7, 0.7)  # degenerate single value


def test_bootstrap_ci_never_exceeds_unit_interval_near_one():
    # values pinned near 1.0 -> a percentile CI must not spill above 1 (unlike SEM)
    point, lo, hi = S.bootstrap_ci([1.0, 1.0, 0.98, 1.0, 0.99, 1.0], seed="hi")
    assert hi <= 1.0 and lo >= 0.0


# --- paired Wilcoxon -------------------------------------------------------
def test_paired_wilcoxon_identical_is_p_one():
    a = [0.1, 0.2, 0.3]
    assert S.paired_wilcoxon(a, a) == (0.0, 1.0)


def test_paired_wilcoxon_detects_consistent_shift():
    a = [0.5, 0.6, 0.7, 0.8, 0.9, 0.4, 0.55, 0.65]
    b = [x - 0.1 for x in a]            # a strictly dominates b
    _, p = S.paired_wilcoxon(a, b)
    assert p < 0.05


# --- Holm–Bonferroni -------------------------------------------------------
def test_holm_bonferroni_matches_manual():
    rejected, adj = S.holm_bonferroni([0.01, 0.04, 0.03], alpha=0.05)
    assert np.allclose(adj, [0.03, 0.06, 0.06])    # step-down, monotone
    assert rejected == [True, False, False]


def test_holm_preserves_input_order_and_caps_at_one():
    rejected, adj = S.holm_bonferroni([0.9, 0.001])
    assert adj[1] < adj[0]                          # small p maps back to index 1
    assert all(x <= 1.0 for x in adj)


# --- money-grid integration ------------------------------------------------
def _trial_rows(dataset, confidence, rewire, seed, points):
    """Build θ-sweep rows for one trial from explicit (theta, recall, precision)."""
    rows = []
    for theta, r, p in points:
        rows.append(dict(dataset=dataset, confidence=confidence, redundancy=0.2,
                         rewire=rewire, seed=seed, theta=theta, n_added=(1 if p == p else 0),
                         edge_recall=r, edge_precision=p, edge_precision_closure=p))
    return rows


def _toy_money_df(corr_extra: bool):
    rows = []
    for seed in range(8):
        # per-type: a single operating point (then nothing)
        rows += _trial_rows("d", "per_type", 0.2, seed,
                            [(0.0, 0.6, 0.3), (1.0, 0.0, float("nan"))])
        corr_pts = [(0.0, 0.6, 0.3)]
        if corr_extra:                           # corroboration adds a high-precision point
            corr_pts.append((0.5, 0.2, 0.9))
        corr_pts.append((1.0, 0.0, float("nan")))
        rows += _trial_rows("d", "corroboration", 0.2, seed, corr_pts)
    return pd.DataFrame(rows)


def test_money_advantage_positive_and_significant_when_frontier_extends():
    out = S.money_advantage_stats(_toy_money_df(corr_extra=True))
    row = out.iloc[0]
    assert row.ap_adv > 0 and row.ci_lo > 0       # advantage CI excludes 0
    assert row.significant and row.wilcoxon_p < 0.05


def test_money_advantage_zero_when_modes_identical():
    out = S.money_advantage_stats(_toy_money_df(corr_extra=False))
    row = out.iloc[0]
    assert row.ap_adv == pytest.approx(0.0)
    assert row.wilcoxon_p == 1.0 and not row.significant
