"""Extended metrics for the integrity & repair study.

Every metric is referenced to an INDEPENDENT oracle: the frozen ``expected``
pairs and ``pristine_edges`` captured from the complete graph BEFORE corruption
(see ``oracle.py``). Edge-level precision/recall are the primary, hard-to-game
signals; pair-level coverage/recovery are secondary.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, Optional, Set, Tuple

import networkx as nx

from ..graph_utils import get_relations
from ..process1_dual_node import concept_of
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


# ---------------------------------------------------------------------------
# Closure-aware (semantic) precision
# ---------------------------------------------------------------------------
# Exact-typed-edge precision (above) marks an added edge "correct" only if that
# *exact* triple was in the pristine graph. That penalizes a repaired edge which
# asserts a *true* fact via a transitive shortcut never materialized in pristine
# — e.g. supersetOf(ext:P->ext:C) where C ⊑ P holds in the pristine taxonomic
# CLOSURE but only as P -> ... -> C. The semantic metric credits such edges, so
# it is an *upper-companion* to exact precision (closure precision >= exact),
# and the gap is exactly the "true transitive shortcut" mislabeling. We report
# BOTH: exact is the conservative, hard-to-game headline; closure is the honest
# semantic ceiling.
@dataclass(frozen=True)
class SemanticOracle:
    """Pristine-closure ground truth for judging an added edge's *meaning*."""

    ancestors: Dict[str, FrozenSet[str]]   # concept -> its pristine (proper) ancestors
    expected: FrozenSet[Pair]              # frozen (instance, concept) membership closure
    superset_of: str
    has_element: str
    has_extension: str


def _pristine_taxonomy(pristine_edges, rels: Dict[str, str]) -> nx.DiGraph:
    """Concept-level child->parent taxonomy from the pristine typed edges, using
    BOTH the abstract subClassOf mirror and the extension supersetOf layer (they
    encode the same pristine subsumptions; the union is robust to expansion
    strategy)."""
    sub, sup = rels["sub_class_of"], rels["superset_of"]
    tax = nx.DiGraph()
    for (u, v, r) in pristine_edges:
        if r == sub:        # abstract:child -> abstract:parent
            tax.add_edge(concept_of(u) or u, concept_of(v) or v)
        elif r == sup:      # extension:parent -> extension:child  => child -> parent
            tax.add_edge(concept_of(v) or v, concept_of(u) or u)
    return tax


def build_semantic_oracle(pristine_edges, expected, config=None) -> SemanticOracle:
    """Snapshot the pristine-closure oracle (ancestor sets + membership closure)."""
    rels = get_relations(config)
    tax = _pristine_taxonomy(pristine_edges, rels)
    ancestors = {c: frozenset(nx.descendants(tax, c)) for c in tax.nodes()}
    return SemanticOracle(
        ancestors=ancestors, expected=frozenset(expected),
        superset_of=rels["superset_of"], has_element=rels["has_element"],
        has_extension=rels["has_extension"],
    )


def is_semantically_correct(edge: Edge, oracle: SemanticOracle) -> bool:
    """Does the added ``edge`` assert something TRUE in the pristine closure?

    * ``supersetOf(ext:P->ext:C)`` asserts ``C ⊑ P`` -> true iff P is a pristine
      ancestor of C (or P == C).
    * ``hasElement(ext:L->i)`` asserts membership -> true iff ``(i, L)`` is in the
      frozen expected (membership-closure) set.
    * ``hasExtension(abstract:C->ext:C)`` -> true iff it links one concept's two
      halves.
    """
    u, v, r = edge
    if r == oracle.superset_of:
        child, parent = (concept_of(v) or v), (concept_of(u) or u)
        return parent == child or parent in oracle.ancestors.get(child, frozenset())
    if r == oracle.has_element:
        return (v, concept_of(u) or u) in oracle.expected
    if r == oracle.has_extension:
        return (concept_of(u) or u) == (concept_of(v) or v)
    return False


def closure_precision(added: Set[Edge], oracle: SemanticOracle) -> Optional[float]:
    """Fraction of added edges that are semantically correct in the pristine
    closure. ``None`` if no edges were added. Always >= exact ``edge_precision``."""
    if not added:
        return None
    return sum(1 for e in added if is_semantically_correct(e, oracle)) / len(added)


def closure_hallucination_rate(added: Set[Edge], oracle: SemanticOracle) -> Optional[float]:
    """Fraction of added edges that are semantically FALSE (``1 - closure_precision``)."""
    p = closure_precision(added, oracle)
    return None if p is None else 1.0 - p


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
