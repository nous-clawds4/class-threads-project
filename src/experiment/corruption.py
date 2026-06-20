"""Controlled corruption of a (dual-node) Class Thread graph.

Two tiers (see ``docs/experiment-design.md``):

* **Tier A** — remove only thread-bearing edges (``hasExtension`` /
  ``supersetOf`` / ``hasElement``) or extension nodes. The ``subClassOf``
  guidance stays intact, so repair re-derives structurally-implied edges; this
  tier measures graceful DEGRADATION and DETECTION, not repair precision.
* **Tier B** — additionally remove ``subClassOf`` guidance edges and inject
  spurious DISTRACTOR ``subClassOf`` edges, so repair can be MISLED into
  proposing fabricated edges. Repair precision / hallucination become real
  measurements only here (otherwise they are 1.0 by construction).
"""
from __future__ import annotations

import copy
import logging
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx

from ..graph_utils import KIND_ABSTRACT, KIND_EXTENSION, get_relations

logger = logging.getLogger(__name__)

Edge = Tuple[str, str, str]

# Thread-bearing corruption arms -> the config relation-key each removes.
THREAD_ARMS = {"E-HX": "has_extension", "E-SO": "superset_of", "E-HE": "has_element"}
ARMS: List[str] = list(THREAD_ARMS) + ["E-MIX", "N-REM"]


def _edges_with_relation(g: nx.DiGraph, rel: str) -> List[Tuple[str, str]]:
    return [(u, v) for u, v, d in g.edges(data=True) if d.get("relation") == rel]


def _sample(items: List, k: int, rng) -> List:
    if k <= 0 or not items:
        return []
    k = min(k, len(items))
    idx = rng.choice(len(items), size=k, replace=False)
    return [items[int(i)] for i in idx]


def _targeted_superset(g: nx.DiGraph, rel_sup: str, k: int) -> List[Tuple[str, str]]:
    """Top-k ``supersetOf`` edges by betweenness on the supersetOf subgraph — the
    "spine" edges whose loss breaks the most threads. Deterministic."""
    sup_edges = _edges_with_relation(g, rel_sup)
    if k <= 0 or not sup_edges:
        return []
    h = nx.DiGraph()
    h.add_edges_from(sup_edges)
    bc = nx.edge_betweenness_centrality(h)
    ranked = sorted(bc, key=lambda e: (-bc[e], e))
    return ranked[:min(k, len(ranked))]


def corrupt(
    graph: nx.DiGraph,
    arm: str,
    rho: float,
    rng,
    *,
    policy: str = "UNIF",
    guidance_removal_rho: float = 0.0,
    guidance_distractor_rate: float = 0.0,
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[nx.DiGraph, Set[Edge], Set[Edge]]:
    """Return ``(damaged_graph, removed, injected)`` without mutating ``graph``.

    ``removed`` / ``injected`` are sets of typed ``(u, v, relation)`` triples.
    Setting either ``guidance_*`` argument > 0 makes this a Tier B condition.

    Arms: ``E-HX`` / ``E-SO`` / ``E-HE`` (single thread relation), ``E-MIX``
    (proportional across all three), ``N-REM`` (remove extension nodes — the
    repair ceiling, since repair never invents nodes).
    Policies: ``UNIF`` (uniform random) or ``TARG`` (targeted, E-SO only).
    """
    g = copy.deepcopy(graph)
    rels = get_relations(config)
    removed: Set[Edge] = set()
    injected: Set[Edge] = set()

    # --- Primary (thread-layer) corruption ---------------------------------
    if arm in THREAD_ARMS:
        rel = rels[THREAD_ARMS[arm]]
        elig = _edges_with_relation(g, rel)
        k = round(rho * len(elig))
        if policy == "TARG":
            if arm != "E-SO":
                raise ValueError("TARG policy is only defined for the E-SO arm.")
            chosen = _targeted_superset(g, rel, k)
        elif policy == "UNIF":
            chosen = _sample(elig, k, rng)
        else:
            raise ValueError(f"Unknown policy {policy!r}.")
        for u, v in chosen:
            removed.add((u, v, rel))
            g.remove_edge(u, v)
    elif arm == "E-MIX":
        for rel in (rels["has_extension"], rels["superset_of"], rels["has_element"]):
            elig = _edges_with_relation(g, rel)
            for u, v in _sample(elig, round(rho * len(elig)), rng):
                removed.add((u, v, rel))
                g.remove_edge(u, v)
    elif arm == "N-REM":
        ext_nodes = [n for n, d in g.nodes(data=True) if d.get("kind") == KIND_EXTENSION]
        for n in _sample(ext_nodes, round(rho * len(ext_nodes)), rng):
            for u, v, d in list(g.in_edges(n, data=True)) + list(g.out_edges(n, data=True)):
                removed.add((u, v, d.get("relation")))
            g.remove_node(n)
    else:
        raise ValueError(f"Unknown corruption arm {arm!r}. Known arms: {ARMS}.")

    # --- Optional guidance corruption (turns this into Tier B) -------------
    sub = rels["sub_class_of"]
    if guidance_removal_rho > 0:
        elig_sub = _edges_with_relation(g, sub)
        for u, v in _sample(elig_sub, round(guidance_removal_rho * len(elig_sub)), rng):
            removed.add((u, v, sub))
            g.remove_edge(u, v)
    if guidance_distractor_rate > 0:
        abst = [n for n, d in g.nodes(data=True) if d.get("kind") == KIND_ABSTRACT]
        n_sub = sum(1 for _, _, d in g.edges(data=True) if d.get("relation") == sub)
        n_inject = round(guidance_distractor_rate * max(n_sub, 1))
        attempts, cap = 0, 100 * max(n_inject, 1)
        while len(injected) < n_inject and len(abst) > 1 and attempts < cap:
            attempts += 1
            i, j = (int(x) for x in rng.choice(len(abst), size=2, replace=False))
            a, b = abst[i], abst[j]
            if a != b and not g.has_edge(a, b):
                g.add_edge(a, b, relation=sub, layer="abstract", distractor=True)
                injected.add((a, b, sub))

    tier = "B" if (guidance_removal_rho or guidance_distractor_rate) else "A"
    logger.info("Corrupt[%s rho=%.2f policy=%s]: removed=%d injected=%d (Tier %s)",
                arm, rho, policy, len(removed), len(injected), tier)
    return g, removed, injected
