"""Frozen ground-truth oracle for the integrity & repair experiments.

The whole study hinges on measuring against structure that does NOT move when
the graph is damaged. We snapshot, from the pristine (complete) graph:

* ``expected`` — the ``(instance, concept)`` pairs that must each have at least
  one valid Class Thread (derived once from the stable abstract-taxonomy +
  direct memberships), then passed explicitly to every downstream validation so
  the goalposts never shift; and
* ``pristine_edges`` — the exact typed edge set, used as an INDEPENDENT oracle
  for repair correctness (an added edge is "correct" iff it was here). This is
  what keeps repair precision from being 1.0 by construction.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, Optional, Set, Tuple

import networkx as nx

from ..process2_thread_enforce import expected_pairs, validate_threads

logger = logging.getLogger(__name__)

Pair = Tuple[str, str]        # (instance, concept)
Edge = Tuple[str, str, str]   # (u, v, relation)


def typed_edges(graph: nx.DiGraph) -> Set[Edge]:
    """The graph's edges as ``(u, v, relation)`` triples."""
    return {(u, v, d.get("relation")) for u, v, d in graph.edges(data=True)}


@dataclass(frozen=True)
class Oracle:
    """Immutable ground truth snapshotted from a pristine graph."""

    expected: FrozenSet[Pair]
    pristine_edges: FrozenSet[Edge]
    pristine_nodes: FrozenSet[str]

    def expected_for(self, instance: str) -> Set[str]:
        """Concepts the given instance is expected to thread to."""
        return {c for (i, c) in self.expected if i == instance}


def freeze_oracle(
    g_pristine: nx.DiGraph, config: Optional[Dict[str, Any]] = None
) -> Oracle:
    """Snapshot the oracle from a pristine graph; assert it is actually complete.

    Raises ``AssertionError`` unless pristine coverage == 1.0 (the experiment is
    only meaningful when we start from a fully-threaded graph).
    """
    expected = frozenset(expected_pairs(g_pristine, config=config))
    report = validate_threads(g_pristine, config=config, expected=set(expected))
    if report.coverage != 1.0:
        raise AssertionError(
            f"Pristine graph is not complete: coverage={report.coverage:.4f}, "
            f"{len(report.broken)} broken pair(s). freeze_oracle requires 1.0."
        )
    oracle = Oracle(
        expected=expected,
        pristine_edges=frozenset(typed_edges(g_pristine)),
        pristine_nodes=frozenset(g_pristine.nodes()),
    )
    logger.info(
        "Froze oracle: %d expected pairs, %d edges, %d nodes.",
        len(oracle.expected), len(oracle.pristine_edges), len(oracle.pristine_nodes),
    )
    return oracle


def holdout_split(
    pairs: Set[Pair], h: float, rng
) -> Tuple[Set[Pair], Set[Pair]]:
    """Split gold ``(instance, concept)`` pairs into ``(train, held_out)``.

    Held-out pairs are meant to be withheld BEFORE the graph/taxonomy is built,
    so any later recovery of them is a non-circular recall signal — repair never
    saw them.
    """
    ordered = sorted(pairs)
    k = round(h * len(ordered))
    if k <= 0:
        return set(ordered), set()
    idx = set(int(i) for i in rng.choice(len(ordered), size=k, replace=False))
    held = {ordered[i] for i in idx}
    return set(ordered) - held, held
