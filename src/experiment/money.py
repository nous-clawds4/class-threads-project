"""The "money figure": repair precision–recall vs the confidence threshold under
adversarial Tier-B guidance corruption, contrasting two confidence scorers.

Key design point (forced by the circularity analysis): repair runs
**autonomously** — it is NOT handed the pristine oracle, so it must infer its
target from the *corrupted* graph. Under subClassOf rewiring its inferred target
is wrong, so it can (and does) fabricate edges. All edits are scored against the
held-out pristine graph (``PRISTINE_EDGES`` / frozen ``EXPECTED``).

Two scorers are compared on the SAME trials:

* ``per_type`` — the per-edge-*type* prior (baseline / ablation). θ can only switch
  repair on or off: good and fabricated ``supersetOf`` proposals are tied at 0.9.
* ``corroboration`` — the evidence-based per-proposal confidence
  (``confidence.py``). It ranks a proposal by how much INDEPENDENT surviving
  structure entails it, so sweeping θ peels fabrications off first — a graded
  precision–recall knob, *to the degree the graph is redundant*.

The synthetic-DAG datasets (``sdag:…``) sweep that redundancy as a controlled
variable (``mi_rate = 0`` is a tree, the null); the Wikidata slice is the real-DAG
credibility point.

Run with ``python -m src.experiment.money``.
"""
from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .. import graph_utils as gu  # noqa: E402
from .. import process2_thread_enforce as p2  # noqa: E402
from . import metrics_ext as M  # noqa: E402
from . import corruption as C  # noqa: E402
from . import oracle as O  # noqa: E402
from .confidence import corroboration_confidence  # noqa: E402
from .experiment import build_dataset  # noqa: E402
from .util import make_rng  # noqa: E402

logger = logging.getLogger(__name__)

# Real datasets (credibility + continuity with fig2/fig3) plus a synthetic-DAG
# redundancy sweep (the controlled independent variable; mi_rate=0 is the tree null).
REAL_DATASETS = ["synthetic", "wordnet:vehicle.n.01:2", "wikidata:Q42889:3"]
SDAG_SWEEP = [f"sdag:5:3:{mi}:0" for mi in (0.0, 0.15, 0.3, 0.5)]
DATASETS = REAL_DATASETS + SDAG_SWEEP

# Confidence scorers under test. ``None`` == the per-type baseline.
CONFIDENCE_MODES: Dict[str, Optional[Any]] = {
    "per_type": None,
    "corroboration": corroboration_confidence(floor=0.0),
}

THETAS = sorted(set([round(x, 3) for x in np.linspace(0.0, 1.0, 21)] + [0.55, 0.75, 0.9])) + [1.01]
REWIRE_RATES = [0.0, 0.1, 0.2, 0.4]
TIER_B_REWIRES = [0.2, 0.4]   # the adversarial regime where fabrication is possible
ARM = "E-SO"                  # thread-layer damage that creates genuinely broken pairs
RHO_THREAD = 0.2
SEEDS = 12


def _nn(x) -> float:
    """None -> NaN (so columns stay float and aggregate cleanly)."""
    return float("nan") if x is None else float(x)


def _cfg_theta(config, theta: float) -> Dict[str, Any]:
    c = copy.deepcopy(config) if config else {}
    c.setdefault("process2", {}).setdefault("repair", {})
    c["process2"]["repair"]["enabled"] = True
    c["process2"]["repair"]["min_confidence"] = theta
    c["process2"]["repair"]["max_new_edges"] = 10_000_000  # non-binding
    return c


def dataset_redundancy(g0: nx.DiGraph, config) -> float:
    """Fraction of (descendant, ancestor) taxonomy pairs reachable by >1 route —
    the multiple-inheritance redundancy corroboration depends on. 0 on a tree."""
    sub = gu.get_relations(config)["sub_class_of"]
    from ..process1_dual_node import concept_of
    tax = nx.DiGraph()
    for u, v, d in g0.edges(data=True):
        if d.get("relation") == sub:
            tax.add_edge(concept_of(u) or u, concept_of(v) or v)
    desc: Dict[str, set] = {}

    def D(n):
        if n not in desc:
            desc[n] = nx.descendants(tax, n) if tax.has_node(n) else set()
        return desc[n]

    tot = red = 0
    for c in tax.nodes():
        for a in D(c):
            tot += 1
            if sum(1 for nb in tax.successors(c) if nb == a or a in D(nb)) > 1:
                red += 1
    return red / tot if tot else 0.0


def run_money_grid(datasets, thetas, rewire_rates, seeds, config) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for ds in datasets:
        flat, g0 = build_dataset(ds, config)
        orc = O.freeze_oracle(g0, config)
        exp = set(orc.expected)
        pe = orc.pristine_edges
        sem = M.build_semantic_oracle(pe, exp, config)  # closure-aware (semantic) oracle
        redund = dataset_redundancy(g0, config)
        logger.info("Dataset %s: redundancy=%.3f, %d expected pairs", ds, redund, len(exp))
        for rewire in rewire_rates:
            for seed in range(seeds):
                rng = make_rng("money", ds, rewire, seed)
                g_dmg, removed, _ = C.corrupt(
                    g0, ARM, RHO_THREAD, rng, config=config, guidance_rewire_rate=rewire)
                surviving = set(g_dmg.nodes())
                v0 = M.valid_pairs(g_dmg, exp, config)
                broken0 = len(exp) - len(v0)
                for mode, builder in CONFIDENCE_MODES.items():
                    for theta in thetas:
                        g = copy.deepcopy(g_dmg)
                        res = p2.repair_threads(  # AUTONOMOUS (no oracle handed in)
                            g, config=_cfg_theta(config, theta), confidence=builder)
                        added = M.added_edges(res)
                        v1 = M.valid_pairs(g, exp, config)
                        rows.append(dict(
                            dataset=ds, confidence=mode, redundancy=redund,
                            rewire=rewire, theta=theta, seed=seed,
                            n_added=len(added),
                            edge_precision=_nn(M.edge_precision(added, pe)),
                            edge_precision_closure=_nn(M.closure_precision(added, sem)),
                            edge_recall=_nn(M.edge_recall(added, removed, pe, surviving)),
                            hallucination=_nn(M.hallucination_rate(added, pe)),
                            hallucination_closure=_nn(M.closure_hallucination_rate(added, sem)),
                            pair_recall=(len(v1 - v0) / broken0 if broken0 else float("nan")),
                            cov_after=len(v1) / len(exp),
                        ))
    return rows


# ---------------------------------------------------------------------------
# Precision–recall aggregation helpers
# ---------------------------------------------------------------------------
_TRAPZ = getattr(np, "trapezoid", None) or np.trapz  # NumPy 2.x renamed trapz


def _pr_points(sub: pd.DataFrame, prec_col: str = "edge_precision"):
    """(recall, precision) points, one per θ, averaged over the slice's seeds/rewires.
    ``prec_col`` selects exact (``edge_precision``) or closure-aware
    (``edge_precision_closure``) precision."""
    g = sub.groupby("theta").agg(r=("edge_recall", "mean"), p=(prec_col, "mean"))
    g = g.dropna()
    pts = sorted(set(zip(g.r.round(4), g.p.round(4))))
    return np.array([x for x, _ in pts]), np.array([y for _, y in pts])


def average_precision(recall, precision) -> float:
    """Area under the precision–recall *frontier* (best precision achievable at
    recall ≥ r, integrated over r). A single, ranking-sensitive summary: the
    per-type scorer (one uncontrolled point) scores low; a graded frontier that
    sustains high precision at lower recall scores high."""
    if len(recall) == 0:
        return 0.0
    grid = np.linspace(0.0, float(recall.max()), 101)
    env = [precision[recall >= r - 1e-9].max() if (recall >= r - 1e-9).any() else 0.0
           for r in grid]
    return float(_TRAPZ(env, grid))


def precision_advantage(df: pd.DataFrame, ds: str,
                        prec_col: str = "edge_precision") -> float:
    """Average-precision advantage of corroboration over per-type in the Tier-B
    (rewired) regime — i.e. how much the corroboration frontier extends beyond the
    baseline's lone operating point. ~0 when the graph has no redundancy.
    ``prec_col`` picks exact or closure-aware precision."""
    def ap(mode):
        sub = df[(df.dataset == ds) & (df.confidence == mode)
                 & (df.rewire.isin(TIER_B_REWIRES))]
        return average_precision(*_pr_points(sub, prec_col))
    return ap("corroboration") - ap("per_type")


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def fig_fidelity(df: pd.DataFrame, path: Path, theta: float = 0.75) -> None:
    """The honest headline (per-type scorer): repair fidelity is governed by
    guidance integrity, not by the confidence threshold. Sweep the rewiring rate."""
    sub_all = df[df.confidence == "per_type"]
    datasets = REAL_DATASETS
    metrics = [("edge_precision", "edge precision", "#2ca02c", "o"),
               ("edge_recall", "edge recall", "#1f77b4", "s"),
               ("hallucination", "hallucination rate", "#d62728", "^")]
    fig, axes = plt.subplots(1, len(datasets), figsize=(5.4 * len(datasets), 4.0),
                             squeeze=False, sharey=True)
    for c, ds in enumerate(datasets):
        ax = axes[0][c]
        sub = sub_all[(sub_all.dataset == ds) & (np.isclose(sub_all.theta, theta))]
        rewires = sorted(sub.rewire.unique())
        for col, label, color, mk in metrics:
            means = [sub[sub.rewire == rw][col].mean() for rw in rewires]
            ax.plot([r * 100 for r in rewires], means, marker=mk, color=color, label=label)
        ax.set_xlabel("guidance rewiring rate (%)")
        ax.set_title(ds)
        ax.set_ylim(-0.02, 1.05)
        ax.grid(alpha=0.3)
        if c == 0:
            ax.set_ylabel(f"value (θ={theta})")
    axes[0][0].legend(fontsize=8, loc="center left")
    fig.suptitle("Repair fidelity is governed by guidance integrity\n"
                 "clean guidance → precision 1.0; precision falls and hallucination rises "
                 "as the taxonomy is rewired")
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def fig3_hallucination(df: pd.DataFrame, path: Path) -> None:
    """Per-type scorer: the gate is on/off, not graded."""
    sub_all = df[df.confidence == "per_type"]
    datasets = REAL_DATASETS
    rewires = sorted(sub_all.rewire.unique())
    colors = plt.cm.viridis(np.linspace(0, 0.85, len(rewires)))
    fig, axes = plt.subplots(1, len(datasets), figsize=(5.4 * len(datasets), 3.7),
                             squeeze=False, sharey=True)
    for c, ds in enumerate(datasets):
        ax = axes[0][c]
        for rw, color in zip(rewires, colors):
            g = sub_all[(sub_all.dataset == ds) & (sub_all.rewire == rw)].groupby("theta")["hallucination"].mean()
            ax.plot(g.index, g.values, marker="o", ms=4, color=color, label=f"rewire={rw:.0%}")
        ax.axvline(0.75, ls="--", color="grey", alpha=0.6)
        ax.set_xlabel("confidence threshold θ")
        ax.set_title(ds)
        ax.set_ylabel("hallucination rate")
        ax.grid(alpha=0.3)
    axes[0][0].legend(fontsize=8)
    fig.suptitle("The fixed per-type confidence gate is on/off, not graded\n"
                 "hallucination is flat in θ until repair shuts off entirely (θ>0.9): "
                 "θ cannot selectively filter fabrications")
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def _pareto_front(recall, precision):
    """Non-dominated (recall, precision) points — the achievable PR frontier,
    sorted by recall. Drops degenerate/dominated points (e.g. all-fabrication)."""
    pts = sorted(zip(recall, precision))
    front = []
    best_p = -1.0
    for r, p in reversed(pts):              # high recall -> low: keep precision records
        if p > best_p + 1e-9:
            front.append((r, p))
            best_p = p
    return sorted(front)


def fig4_pr_curve(df: pd.DataFrame, path: Path, datasets: List[str]) -> None:
    """THE money figure: the precision–recall frontier traced by θ, per-type vs
    corroboration, in the Tier-B (rewired) regime. Per-type is a SINGLE
    uncontrolled operating point; corroboration grades into a frontier that bows
    above it — but only where the graph is redundant (collapses onto the point on
    a tree)."""
    fig, axes = plt.subplots(1, len(datasets), figsize=(4.7 * len(datasets), 4.3),
                             squeeze=False, sharex=True, sharey=True)
    for c, ds in enumerate(datasets):
        ax = axes[0][c]
        redund = df[df.dataset == ds].redundancy.iloc[0]
        corr = df[(df.dataset == ds) & (df.confidence == "corroboration")
                  & (df.rewire.isin(TIER_B_REWIRES))]
        # corroboration frontier, exact-edge precision (conservative headline)
        front = _pareto_front(*_pr_points(corr, "edge_precision"))
        ax.plot([r for r, _ in front], [p for _, p in front], "-s", ms=5,
                color="#1b7837", label="corroboration (exact precision)", zorder=4)
        # corroboration frontier, closure-aware (semantic) precision — the honest
        # ceiling; true transitive shortcuts the exact metric mislabels are credited
        cfront = _pareto_front(*_pr_points(corr, "edge_precision_closure"))
        ax.plot([r for r, _ in cfront], [p for _, p in cfront], "--^", ms=5,
                color="#66bd63", label="corroboration (closure precision)", zorder=3)
        # per-type: a single operating point (bold marker), exact precision
        ptf = _pareto_front(*_pr_points(df[(df.dataset == ds) & (df.confidence == "per_type")
                                           & (df.rewire.isin(TIER_B_REWIRES))], "edge_precision"))
        ax.scatter([r for r, _ in ptf], [p for _, p in ptf], marker="*", s=240,
                   color="#b30000", edgecolor="k", linewidth=0.5, zorder=6,
                   label="per-type (single point)")
        ax.set_xlabel("edge recall")
        ax.set_title(f"{ds}\n(redundancy={redund:.2f})", fontsize=9)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.05)
        ax.grid(alpha=0.3)
        if c == 0:
            ax.set_ylabel("edge precision")
    axes[0][0].legend(fontsize=8, loc="upper right")
    fig.suptitle("Money figure — evidence-based confidence turns θ into a graded "
                 "precision–recall knob\n(Tier-B rewiring; per-type is one uncontrolled "
                 "point, corroboration trades recall for precision where the graph is redundant)")
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def fig6_redundancy_scaling(df: pd.DataFrame, path: Path) -> None:
    """The mechanism, isolated: corroboration's average-precision advantage over
    per-type vs the graph's redundancy. ~0 on a tree (the null), rising with the
    synthetic multiple-inheritance rate; the Wikidata slice is the real-DAG point."""
    present = list(df.dataset.unique())
    sdag = [d for d in present if d.startswith("sdag:")]
    pts = sorted(((df[df.dataset == d].redundancy.iloc[0], precision_advantage(df, d), d)
                  for d in sdag), key=lambda t: t[0])
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    xs = [x for x, _, _ in pts]
    ys = [y for _, y, _ in pts]
    ax.plot(xs, ys, marker="o", color="#1b7837", label="synthetic DAG sweep (mi_rate)")
    for x, y, d in pts:
        mi = d.split(":")[3]
        ax.annotate(f"mi={mi}", (x, y), fontsize=7, xytext=(3, 4),
                    textcoords="offset points")
    for ds in REAL_DATASETS:
        if ds in df.dataset.unique():
            rx = df[df.dataset == ds].redundancy.iloc[0]
            ry = precision_advantage(df, ds)
            ax.scatter([rx], [ry], marker="*", s=160, zorder=5,
                       label=ds, edgecolor="k", linewidth=0.4)
    ax.axhline(0, ls=":", color="grey")
    ax.set_xlabel("graph redundancy (fraction of ancestor pairs with >1 route)")
    ax.set_ylabel("average-precision advantage\n(corroboration − per-type, Tier-B)")
    ax.set_title("Corroboration's precision gain scales with graph redundancy\n"
                 "≈0 on a tree (the null); the power of the knob is a measurable "
                 "property of the graph")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    logging.basicConfig(level=logging.ERROR)
    cfg = gu.load_config("config.yaml")
    rows = run_money_grid(DATASETS, THETAS, REWIRE_RATES, SEEDS, cfg)
    df = pd.DataFrame(rows)

    out = Path("results")
    out.mkdir(exist_ok=True)
    df.to_parquet(out / "money.parquet")
    df.to_csv(out / "money.csv", index=False)

    figs = out / "figures"
    figs.mkdir(exist_ok=True, parents=True)
    fig_fidelity(df, figs / "fig2_fidelity_vs_corruption.png")
    fig3_hallucination(df, figs / "fig3_theta_gate.png")
    pr_datasets = ["sdag:5:3:0.0:0", "sdag:5:3:0.5:0", "wikidata:Q42889:3"]
    fig4_pr_curve(df, figs / "fig4_pr_curve.png", [d for d in pr_datasets if d in df.dataset.unique()])
    fig6_redundancy_scaling(df, figs / "fig6_redundancy_scaling.png")
    print(f"Wrote {len(df)} rows; figures fig2, fig3, fig4_pr_curve, fig6_redundancy_scaling")

    # Honest quantitative summary: average-precision advantage per dataset, under
    # exact (conservative) and closure-aware (semantic) precision.
    print("\nAverage-precision advantage (corroboration − per-type), Tier-B:")
    print(f"  {'dataset':24s} {'redundancy':>10} {'AP_adv(exact)':>14} {'AP_adv(closure)':>16}")
    for ds in DATASETS:
        if ds in df.dataset.unique():
            redund = df[df.dataset == ds].redundancy.iloc[0]
            adv = precision_advantage(df, ds, "edge_precision")
            advc = precision_advantage(df, ds, "edge_precision_closure")
            print(f"  {ds:24s} {redund:>10.3f} {adv:>+14.3f} {advc:>+16.3f}")


if __name__ == "__main__":
    main()
