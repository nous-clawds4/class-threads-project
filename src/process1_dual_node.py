"""Process 1 — Dual-Node Expansion.

Transforms a *flat* graph (one node per concept, one per instance) into a
*dual-node* graph in which every selected concept is split into:

    abstract:{C}   — the intension (the idea of the concept)
    extension:{C}  — the set of all instances of the concept

linked by a ``hasExtension`` edge.

Edge rewiring (the "Downhill forward path" topology)
----------------------------------------------------
A **Class Thread** must be a literal forward directed path::

    abstract:C --hasExtension--> extension:C --supersetOf--> ... --hasElement--> instance

To make that hold, edges are rewired as follows. Given a flat ``subClassOf``
edge ``child -> parent`` and a flat ``instanceOf`` edge ``instance -> concept``:

    hasExtension : abstract:C        -> extension:C            (per selected concept)
    supersetOf   : extension:parent  -> extension:child        (EXTENSION layer, general->specific, thread-bearing)
    subClassOf   : abstract:child    -> abstract:parent        (ABSTRACT layer, conventional, fidelity)
    hasElement   : extension:concept -> instance               (extension -> instance, thread-bearing)

The extension-layer ``supersetOf`` edges are the ones threads traverse; the
abstract-layer ``subClassOf`` edges mirror the classic taxonomy for
readability/visualization and are never on a thread (after the first
``hasExtension`` a thread stays in the extension/instance layers).

Concepts *not* selected for expansion keep their original single node, which
plays both the abstract and extension role, so mixed selections stay connected.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import networkx as nx

from .graph_utils import (
    KIND_ABSTRACT,
    KIND_CONCEPT,
    KIND_EXTENSION,
    KIND_INSTANCE,
    get_relations,
)

logger = logging.getLogger(__name__)

# Node-id prefixes for the two halves of a concept.
ABSTRACT_PREFIX = "abstract:"
EXTENSION_PREFIX = "extension:"


def abstract_id(concept: str) -> str:
    """Node id for a concept's abstract (intension) node."""
    return f"{ABSTRACT_PREFIX}{concept}"


def extension_id(concept: str) -> str:
    """Node id for a concept's extension (set-of-instances) node."""
    return f"{EXTENSION_PREFIX}{concept}"


# ===========================================================================
# Concept selection (configurable: list or heuristic)
# ===========================================================================
def select_concepts(
    flat: nx.DiGraph,
    config: Optional[Dict[str, Any]] = None,
    selected: Optional[Iterable[str]] = None,
) -> Set[str]:
    """Decide which concept nodes get dual (abstract+extension) representation.

    Precedence: an explicit ``selected`` argument wins; otherwise the
    ``process1`` block of ``config`` is consulted:

        strategy: "all"   -> every concept node
        strategy: "list"  -> only the names listed under ``concepts``
    """
    all_concepts = {
        n for n, d in flat.nodes(data=True) if d.get("kind") == KIND_CONCEPT
    }
    if selected is not None:
        chosen = set(selected) & all_concepts
        _warn_unknown(set(selected) - all_concepts)
        return chosen

    p1 = (config or {}).get("process1", {}) or {}
    strategy = p1.get("strategy", "all")
    if strategy == "list":
        requested = set(p1.get("concepts", []) or [])
        _warn_unknown(requested - all_concepts)
        return requested & all_concepts
    if strategy != "all":
        logger.warning("Unknown process1.strategy %r; defaulting to 'all'.", strategy)
    return all_concepts


def _warn_unknown(unknown: Set[str]) -> None:
    if unknown:
        logger.warning("Ignoring %d unknown concept(s): %s", len(unknown),
                       ", ".join(sorted(unknown)))


# ===========================================================================
# The expansion itself
# ===========================================================================
def expand_to_dual_nodes(
    flat: nx.DiGraph,
    config: Optional[Dict[str, Any]] = None,
    selected: Optional[Iterable[str]] = None,
) -> nx.DiGraph:
    """Return a new dual-node graph built from a flat graph.

    The input graph is not mutated.
    """
    rels = get_relations(config)
    rel_has_ext = rels["has_extension"]
    rel_sub = rels["sub_class_of"]        # flat input + abstract-layer mirror
    rel_inst = rels["instance_of"]        # flat input
    rel_superset = rels["superset_of"]    # extension-layer, thread-bearing
    rel_has_element = rels["has_element"]  # extension -> instance, thread-bearing

    chosen = select_concepts(flat, config=config, selected=selected)
    logger.info("Process 1: expanding %d/%d concept(s).", len(chosen),
                sum(1 for _, d in flat.nodes(data=True) if d.get("kind") == KIND_CONCEPT))

    g = nx.DiGraph()

    # --- 1. Create nodes -----------------------------------------------------
    # For each concept, create dual nodes (if selected) or keep the single node.
    for n, d in flat.nodes(data=True):
        kind = d.get("kind")
        if kind == KIND_CONCEPT:
            if n in chosen:
                g.add_node(abstract_id(n), kind=KIND_ABSTRACT, concept=n,
                           label=f"abstract:{d.get('label', n)}")
                g.add_node(extension_id(n), kind=KIND_EXTENSION, concept=n,
                           label=f"extension:{d.get('label', n)}")
                g.add_edge(abstract_id(n), extension_id(n), relation=rel_has_ext)
            else:
                # Unexpanded concept keeps its single node, playing both roles.
                g.add_node(n, **d)
        elif kind == KIND_INSTANCE:
            g.add_node(n, **d)
        else:
            # Pass through anything unexpected unchanged.
            g.add_node(n, **d)

    # Resolver: which node plays the abstract / extension role for a concept.
    def role(concept: str, prefix_fn) -> str:
        return prefix_fn(concept) if concept in chosen else concept

    abstract_of = lambda c: role(c, abstract_id)      # noqa: E731
    extension_of = lambda c: role(c, extension_id)     # noqa: E731

    # --- 2. Rewire subClassOf edges -----------------------------------------
    for u, v, d in flat.edges(data=True):
        if d.get("relation") != rel_sub:
            continue
        child, parent = u, v  # flat subClassOf is child -> parent
        # Extension layer: general -> specific (the thread-bearing edge).
        g.add_edge(extension_of(parent), extension_of(child),
                   relation=rel_superset, layer="extension")
        # Abstract layer: conventional child -> parent, only when both expanded.
        if child in chosen and parent in chosen:
            g.add_edge(abstract_id(child), abstract_id(parent),
                       relation=rel_sub, layer="abstract")

    # --- 3. Rewire instanceOf edges -----------------------------------------
    for u, v, d in flat.edges(data=True):
        if d.get("relation") != rel_inst:
            continue
        instance, concept = u, v  # flat instanceOf is instance -> concept
        g.add_edge(extension_of(concept), instance, relation=rel_has_element)

    logger.info(
        "Process 1 done: %d nodes, %d edges (%d abstract, %d extension, %d instance).",
        g.number_of_nodes(), g.number_of_edges(),
        sum(1 for _, dd in g.nodes(data=True) if dd.get("kind") == KIND_ABSTRACT),
        sum(1 for _, dd in g.nodes(data=True) if dd.get("kind") == KIND_EXTENSION),
        sum(1 for _, dd in g.nodes(data=True) if dd.get("kind") == KIND_INSTANCE),
    )
    return g


# ===========================================================================
# Helpers used by Process 2 / metrics / tests
# ===========================================================================
def concept_of(node_id: str) -> Optional[str]:
    """Recover the concept name from an ``abstract:`` / ``extension:`` node id.

    Returns ``None`` for plain (unexpanded or instance) nodes.
    """
    if node_id.startswith(ABSTRACT_PREFIX):
        return node_id[len(ABSTRACT_PREFIX):]
    if node_id.startswith(EXTENSION_PREFIX):
        return node_id[len(EXTENSION_PREFIX):]
    return None


def list_dual_pairs(graph: nx.DiGraph) -> List[Tuple[str, str, str]]:
    """Return ``(concept, abstract_id, extension_id)`` for each expanded concept."""
    pairs = []
    for n, d in graph.nodes(data=True):
        if d.get("kind") == KIND_ABSTRACT:
            c = d.get("concept") or concept_of(n)
            pairs.append((c, n, extension_id(c)))
    return pairs
