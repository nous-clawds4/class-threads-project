"""Extended metrics for the integrity & repair study.

Every metric is referenced to an INDEPENDENT oracle: the frozen ``expected``
pairs and ``pristine_edges`` captured from the complete graph BEFORE corruption
(see ``oracle.py``). Edge-level precision/recall are the primary, hard-to-game
signals; pair-level coverage/recovery are secondary.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Set, Tuple

import networkx as nx

from ..graph_utils import get_relations
from ..process2_thread_enforce import concepts_via_threads, has_thread

logger = logging.getLogger(__name__)

Pair = Tuple[str, str]
Edge = Tuple[str, str, str]


# ---------------------------------------------------------------------------
# Edge-level metrics (scored against pristine_edges)
# ---------------------------------------------------------------------------
def added_edges(repair_result) -> Set[Edge]:
    """Typed ``(u, v, relation)`` triples a RepairResult actually added."""
    return {(r["u"], r["v"], r["relation"]) for r in repair_result.added}


def edge_precision(added: Set[Edge], pristine_edges) -> Optional[float]:
    """Fraction of added edges that were in the pristine graph. ``None`` if no
    edges were added (precision undefined)."""
    if not added:
        return None
    return len(added & set(pristine_edges)) / len(added)


def hallucination_rate(added: Set[Edge], pristine_edges) -> Optional[float]:
    """Fraction of added edges NEVER in the pristine graph (fabricated).
    Equals ``1 - edge_precision``; ``None`` if no edges were added."""
    if not added:
        return None
    return len(added - set(pristine_edges)) / len(added)


def _default_repairable_relations() -> Set[str]:
    """Relation names repair is able to add (the three thread-bearing edges)."""
    r = get_relations(None)
    return {r["has_extension"], r["superset_of"], r["has_element"]}


def recoverable_removed(removed: Set[Edge], pristine_edges, surviving_nodes,
                        repairable_relations: Optional[Set[str]] = None) -> Set[Edge]:
    """Removed pristine edges that repair could in principle re-add: present in
    the pristine graph, both endpoints still exist, and of a relation type repair
    actually proposes (hasExtension / supersetOf / hasElement). Excludes the
    node-removal ceiling and the guidance (subClassOf) edges repair never targets."""
    pe, sn = set(pristine_edges), set(surviving_nodes)
    rel_ok = (repairable_relations if repairable_relations is not None
              else _default_repairable_relations())
    return {(u, v, r) for (u, v, r) in removed
            if (u, v, r) in pe and u in sn and v in sn and r in rel_ok}


def edge_recall(added: Set[Edge], removed: Set[Edge], pristine_edges, surviving_nodes,
                repairable_relations: Optional[Set[str]] = None) -> Optional[float]:
    """Fraction of *recoverable* removed edges that repair re-added. ``None`` if
    there were no recoverable removed edges."""
    rec = recoverable_removed(removed, pristine_edges, surviving_nodes, repairable_relations)
    if not rec:
        return None
    return len(added & rec) / len(rec)


# ---------------------------------------------------------------------------
# Pair-level metrics (scored against frozen `expected`)
# ---------------------------------------------------------------------------
def valid_pairs(graph: nx.DiGraph, expected, config=None) -> Set[Pair]:
    """Expected pairs that currently have at least one valid thread."""
    return {(i, c) for (i, c) in expected if has_thread(graph, c, i, config)}


def coverage(graph: nx.DiGraph, expected, config=None) -> float:
    """Fraction of frozen expected pairs with >=1 valid thread."""
    exp = set(expected)
    if not exp:
        return 1.0
    return len(valid_pairs(graph, exp, config)) / len(exp)


def recovery_fraction(cov_corrupt: float, cov_repair: float) -> Optional[float]:
    """Fraction of the *lost* coverage that repair restored. ``None`` if nothing
    was lost (``cov_corrupt == 1.0``)."""
    if cov_corrupt >= 1.0:
        return None
    return (cov_repair - cov_corrupt) / (1.0 - cov_corrupt)


def pair_false_positives(graph: nx.DiGraph, expected, config=None) -> Set[Pair]:
    """``(instance, concept)`` pairs that thread but are NOT in ``expected`` —
    i.e. false memberships introduced (e.g. by a fabricated repair edge). Must
    be empty in the conservative regime; non-empty is a real safety finding."""
    expected = set(expected)
    instances = {i for (i, _) in expected}
    fps: Set[Pair] = set()
    for inst in instances:
        threaded = concepts_via_threads(graph, inst, config)
        allowed = {c for (i, c) in expected if i == inst}
        fps.update((inst, c) for c in (threaded - allowed))
    return fps


def newly_threading(before_valid: Set[Pair], after_valid: Set[Pair]) -> Set[Pair]:
    """Expected pairs threading after repair but not before."""
    return after_valid - before_valid
