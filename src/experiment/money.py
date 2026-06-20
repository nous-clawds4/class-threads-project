"""The "money figure": repair precision–recall vs the confidence threshold under
adversarial Tier-B guidance corruption.

Key design point (forced by the circularity analysis): repair runs
**autonomously** — it is NOT handed the pristine oracle, so it must infer its
target from the *corrupted* graph. Under subClassOf rewiring its inferred target
is wrong, so it can (and does) fabricate edges. All edits are scored against the
held-out pristine graph (``PRISTINE_EDGES`` / frozen ``EXPECTED``).

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
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .. import graph_utils as gu  # noqa: E402
from .. import process2_thread_enforce as p2  # noqa: E402
from . import corruption as C  # noqa: E402
from . import metrics_ext as M  # noqa: E402
from . import oracle as O  # noqa: E402
from .experiment import build_dataset  # noqa: E402
from .util import make_rng  # noqa: E402

logger = logging.getLogger(__name__)

DATASETS = ["synthetic", "wordnet:vehicle.n.01:2", "wikidata:Q42889:3"]
THETAS = [0.0, 0.50, 0.55, 0.60, 0.75, 0.85, 0.90, 0.95, 1.00, 1.01]
REWIRE_RATES = [0.0, 0.1, 0.2, 0.4]
ARM = "E-SO"          # thread-layer damage that creates genuinely broken pairs
RHO_THREAD = 0.2
SEEDS = 15


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


def run_money_grid(datasets, thetas, rewire_rates, seeds, config) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for ds in datasets:
        flat, g0 = build_dataset(ds, config)
        orc = O.freeze_oracle(g0, config)
        exp = set(orc.expected)
        pe = orc.pristine_edges
        for rewire in rewire_rates:
            for seed in range(seeds):
                rng = make_rng("money", ds, rewire, seed)
                g_dmg, removed, _ = C.corrupt(
                    g0, ARM, RHO_THREAD, rng, config=config, guidance_rewire_rate=rewire)
                surviving = set(g_dmg.nodes())
                v0 = M.valid_pairs(g_dmg, exp, config)
                broken0 = len(exp) - len(v0)
                for theta in thetas:
                    g = copy.deepcopy(g_dmg)
                    res = p2.repair_threads(g, config=_cfg_theta(config, theta))  # AUTONOMOUS
                    added = M.added_edges(res)
                    v1 = M.valid_pairs(g, exp, config)
                    rows.append(dict(
                        dataset=ds, rewire=rewire, theta=theta, seed=seed,
                        n_added=len(added),
                        edge_precision=_nn(M.edge_precision(added, pe)),
                        edge_recall=_nn(M.edge_recall(added, removed, pe, surviving)),
                        hallucination=_nn(M.hallucination_rate(added, pe)),
                        pair_recall=(len(v1 - v0) / broken0 if broken0 else float("nan")),
                        cov_after=len(v1) / len(exp),
                    ))
    return rows


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def fig_fidelity(df: pd.DataFrame, path: Path, theta: float = 0.75) -> None:
    """The honest headline: repair fidelity is governed by guidance integrity,
    not by the confidence threshold. At a fixed theta, sweep the rewiring rate."""
    datasets = list(df.dataset.unique())
    metrics = [("edge_precision", "edge precision", "#2ca02c", "o"),
               ("edge_recall", "edge recall", "#1f77b4", "s"),
               ("hallucination", "hallucination rate", "#d62728", "^")]
    fig, axes = plt.subplots(1, len(datasets), figsize=(5.4 * len(datasets), 4.0),
                             squeeze=False, sharey=True)
    for c, ds in enumerate(datasets):
        ax = axes[0][c]
        sub = df[(df.dataset == ds) & (np.isclose(df.theta, theta))]
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
    datasets = list(df.dataset.unique())
    rewires = sorted(df.rewire.unique())
    colors = plt.cm.viridis(np.linspace(0, 0.85, len(rewires)))
    fig, axes = plt.subplots(1, len(datasets), figsize=(5.4 * len(datasets), 3.7),
                             squeeze=False, sharey=True)
    for c, ds in enumerate(datasets):
        ax = axes[0][c]
        for rw, color in zip(rewires, colors):
            g = df[(df.dataset == ds) & (df.rewire == rw)].groupby("theta")["hallucination"].mean()
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
    print(f"Wrote {len(df)} rows; figures fig2_fidelity_vs_corruption.png, fig3_theta_gate.png")

    summ = (df.groupby(["dataset", "rewire", "theta"])
              .agg(prec=("edge_precision", "mean"), rec=("edge_recall", "mean"),
                   halluc=("hallucination", "mean"), added=("n_added", "mean")).round(3))
    print(summ.to_string())


if __name__ == "__main__":
    main()
