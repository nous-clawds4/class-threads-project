"""Factor-grid runner for the integrity & repair study.

Increment 1: Tier-A degradation/recovery + the dual-vs-single ablation on
synthetic + WordNet. Emits one tidy row per (dataset, arm, rho, seed) trial to
``results/results.{parquet,csv}``. Run with ``python -m src.experiment.experiment``.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from .. import graph_utils as gu
from .. import process1_dual_node as p1
from .. import process2_thread_enforce as p2
from . import ablation as A
from . import corruption as C
from . import metrics_ext as M
from . import oracle as O
from .util import make_rng

logger = logging.getLogger(__name__)

DEFAULT_DATASETS = ["synthetic", "wordnet:vehicle.n.01:2", "wikidata:Q42889:3"]
DEFAULT_ARMS = ["E-SO", "E-HE", "E-HX"]
DEFAULT_RHOS = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
DEFAULT_SEEDS = 20


def build_dataset(name: str, config):
    """Return ``(flat, dual)`` for a dataset spec.

    ``"synthetic"`` or ``"wordnet:<root>:<depth>"`` (e.g. ``wordnet:vehicle.n.01:2``).
    """
    if name == "synthetic":
        flat = gu.build_synthetic_graph(config=config)
    elif name.startswith("wordnet:"):
        _, root, depth = name.split(":")
        flat = gu.build_wordnet_graph(roots=[root], max_depth=int(depth), config=config)
    elif name.startswith("wikidata:"):
        from .wikidata import build_wikidata_graph
        _, qid, depth = name.split(":")
        flat = build_wikidata_graph(root=qid, max_depth=int(depth), config=config)
    else:
        raise ValueError(f"Unknown dataset {name!r}")
    return flat, p1.expand_to_dual_nodes(flat, config=config)


def run_grid(datasets, arms, rhos, seeds, config) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for ds in datasets:
        flat, g0 = build_dataset(ds, config)
        orc = O.freeze_oracle(g0, config)
        exp = set(orc.expected)
        logger.info("Dataset %s: %d dual nodes, %d expected pairs",
                    ds, g0.number_of_nodes(), len(exp))
        for arm in arms:
            for rho in rhos:
                for seed in range(seeds):
                    rng = make_rng(ds, arm, rho, seed)
                    g_dmg, removed, _ = C.corrupt(g0, arm, rho, rng, config=config)

                    # --- dual model: degradation then repair-recovery ---
                    dual_valid = M.valid_pairs(g_dmg, exp, config)
                    cov_corrupt = len(dual_valid) / len(exp)
                    report = p2.validate_threads(g_dmg, config=config, expected=exp)
                    result = p2.repair_threads(g_dmg, config=config, expected=exp, report=report)
                    cov_repair = M.coverage(g_dmg, exp, config)
                    added = M.added_edges(result)
                    eprec = M.edge_precision(added, orc.pristine_edges)

                    # --- single-node comparator under homologous loss ---
                    flat_dmg = A.apply_flat_removed(flat, A.homologous_flat_removed(removed, config))
                    single_valid = A.single_node_valid_pairs(flat_dmg, exp, config)
                    cov_single = len(single_valid) / len(exp)

                    rows.append(dict(
                        dataset=ds, arm=arm, rho=rho, seed=seed,
                        n_expected=len(exp), removed=len(removed),
                        cov_corrupt=cov_corrupt, cov_repair=cov_repair, cov_single=cov_single,
                        edge_precision=(float("nan") if eprec is None else eprec),
                        # single-node detection errors vs the (exact) dual validator
                        single_fn=len(single_valid - dual_valid),  # missed a real break
                        single_fp=len(dual_valid - single_valid),  # false alarm
                    ))
    return rows


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    import pandas as pd

    cfg = gu.load_config("config.yaml")
    rows = run_grid(DEFAULT_DATASETS, DEFAULT_ARMS, DEFAULT_RHOS, DEFAULT_SEEDS, cfg)
    df = pd.DataFrame(rows)

    out = Path("results")
    out.mkdir(exist_ok=True)
    df.to_parquet(out / "results.parquet")
    df.to_csv(out / "results.csv", index=False)
    print(f"Wrote {len(df)} rows to results/results.parquet")

    summary = (df.groupby(["dataset", "arm", "rho"])
                 [["cov_corrupt", "cov_repair", "cov_single", "edge_precision", "single_fn"]]
                 .mean().round(3))
    print(summary.to_string())


if __name__ == "__main__":
    main()
