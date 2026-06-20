"""Metrics & evaluation for the Class Thread model.

Consumes :class:`~src.process2_thread_enforce.ThreadReport` objects and graphs
to produce the headline numbers:

* **Thread coverage** — % of expected ``(instance, concept)`` pairs with >=1 valid thread
* **Broken threads** before vs. after enforcement
* **Average valid threads per instance**
* **Flat-vs-threaded query examples** that show why the dual-node model helps

The query comparison is the qualitative payoff: in a *flat* graph, "give me all
animals" either under-answers (naive 1-hop ``instanceOf``) or needs a bespoke
recursive walk mixing ``subClassOf`` + ``instanceOf`` in the right directions.
In the *threaded* graph it is one uniform path pattern
``hasExtension -> supersetOf* -> hasElement`` — and it returns the full answer.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set

import networkx as nx

from .graph_utils import KIND_INSTANCE, get_relations
from .process2_thread_enforce import (
    ThreadReport,
    instances_via_threads,
    validate_threads,
)

logger = logging.getLogger(__name__)


# ===========================================================================
# Scalar metrics derived from a ThreadReport
# ===========================================================================
def thread_coverage(report: ThreadReport) -> float:
    """% (as a 0..1 fraction) of expected pairs that have >=1 valid thread."""
    return report.coverage


def broken_threads(report: ThreadReport) -> int:
    """Number of expected pairs with no valid thread."""
    return len(report.broken)


def avg_valid_threads_per_instance(report: ThreadReport) -> float:
    """Mean number of concepts each instance validly threads to."""
    counts = report.per_instance_valid
    return (sum(counts.values()) / len(counts)) if counts else 0.0


def compute_metrics(
    graph: nx.DiGraph,
    config: Optional[Dict[str, Any]] = None,
    expected: Optional[Set] = None,
    reference: Optional[nx.DiGraph] = None,
    report: Optional[ThreadReport] = None,
) -> Dict[str, Any]:
    """Validate ``graph`` (unless ``report`` is supplied) and return a metrics dict."""
    if report is None:
        report = validate_threads(graph, config=config, expected=expected,
                                  reference=reference)
    return {
        "expected_pairs": len(report.expected),
        "valid_pairs": len(report.valid),
        "broken_pairs": len(report.broken),
        "coverage": round(report.coverage, 4),
        "num_instances": len(report.per_instance_valid),
        "avg_valid_threads_per_instance": round(
            avg_valid_threads_per_instance(report), 4
        ),
        "threads_per_instance": dict(sorted(report.per_instance_valid.items())),
    }


# ===========================================================================
# Before / after enforcement comparison
# ===========================================================================
def enforcement_delta(before: ThreadReport, after: ThreadReport) -> Dict[str, Any]:
    """Compare validation reports from before and after Process 2 repair."""
    newly_valid = sorted(after.valid - before.valid)
    return {
        "coverage_before": round(before.coverage, 4),
        "coverage_after": round(after.coverage, 4),
        "broken_before": len(before.broken),
        "broken_after": len(after.broken),
        "threads_repaired": len(before.broken) - len(after.broken),
        "newly_valid_pairs": [f"{i}~{c}" for i, c in newly_valid],
    }


# ===========================================================================
# Flat-vs-threaded query examples (the value demonstration)
# ===========================================================================
def flat_instances_direct(
    flat: nx.DiGraph, concept: str, config: Optional[Dict[str, Any]] = None
) -> Set[str]:
    """NAIVE flat query: instances asserted *directly* ``instanceOf`` ``concept``.

    This is the obvious 1-hop query on a flat graph — and it *misses* instances
    that belong only through a subclass (e.g. a ``dog`` is not returned for
    ``animal``).
    """
    rels = get_relations(config)
    inst_rel = rels["instance_of"]
    return {
        u for u, v, d in flat.in_edges(concept, data=True)
        if d.get("relation") == inst_rel and v == concept
    }


def flat_instances_transitive(
    flat: nx.DiGraph, concept: str, config: Optional[Dict[str, Any]] = None
) -> Set[str]:
    """CORRECT flat query, the hard way: full closure over subClassOf+instanceOf.

    Requires a bespoke recursive walk (collect subclasses, then their direct
    instances) with careful edge-direction handling.
    """
    rels = get_relations(config)
    sub, inst_rel = rels["sub_class_of"], rels["instance_of"]

    # Build child->parent taxonomy, then subclasses(concept) = ancestors in it.
    tax = nx.DiGraph()
    for u, v, d in flat.edges(data=True):
        if d.get("relation") == sub:  # flat subClassOf: child -> parent
            tax.add_edge(u, v)
    concepts = {concept}
    if tax.has_node(concept):
        concepts |= set(nx.ancestors(tax, concept))  # all subclasses of concept

    result: Set[str] = set()
    for u, v, d in flat.edges(data=True):
        if d.get("relation") == inst_rel and v in concepts:
            result.add(u)
    return result


def threaded_instances(
    graph: nx.DiGraph, concept: str, config: Optional[Dict[str, Any]] = None
) -> Set[str]:
    """THREADED query: one uniform pattern ``hasExtension -> supersetOf* -> hasElement``."""
    return instances_via_threads(graph, concept, config)


def query_comparison(
    flat: nx.DiGraph,
    dual: nx.DiGraph,
    concept: str,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Compare the three query strategies for "all instances of ``concept``"."""
    direct = flat_instances_direct(flat, concept, config)
    transitive = flat_instances_transitive(flat, concept, config)
    threaded = threaded_instances(dual, concept, config)
    return {
        "concept": concept,
        "flat_direct": sorted(direct),               # naive — typically incomplete
        "flat_transitive": sorted(transitive),       # correct but bespoke
        "threaded": sorted(threaded),                # correct via one pattern
        "flat_direct_misses": sorted(transitive - direct),
        "threaded_matches_transitive": threaded == transitive,
    }


# ===========================================================================
# Pretty-printing helpers (for the notebook / README)
# ===========================================================================
def format_metrics(metrics: Dict[str, Any]) -> str:
    """Render a metrics dict (from :func:`compute_metrics`) as an aligned block."""
    lines = [
        f"  coverage                       : {metrics['coverage'] * 100:.1f}%"
        f"  ({metrics['valid_pairs']}/{metrics['expected_pairs']} pairs)",
        f"  broken threads                 : {metrics['broken_pairs']}",
        f"  instances                      : {metrics['num_instances']}",
        f"  avg valid threads / instance   : {metrics['avg_valid_threads_per_instance']}",
    ]
    return "\n".join(lines)


def format_delta(delta: Dict[str, Any]) -> str:
    """Render an :func:`enforcement_delta` dict as a before->after block."""
    lines = [
        f"  coverage   : {delta['coverage_before'] * 100:.1f}%"
        f"  ->  {delta['coverage_after'] * 100:.1f}%",
        f"  broken     : {delta['broken_before']}  ->  {delta['broken_after']}"
        f"   ({delta['threads_repaired']} repaired)",
    ]
    if delta["newly_valid_pairs"]:
        lines.append("  restored   : " + ", ".join(delta["newly_valid_pairs"]))
    return "\n".join(lines)
