"""Single-node (flat) comparator for the dual-vs-single ablation.

The single-node model is the original FLAT graph (one node per concept), with
membership detected by taxonomic closure. To compare fairly under identical
information loss, we map each removed *dual* edge to its flat pre-image:

    supersetOf(extension:parent -> extension:child)  <->  subClassOf(child -> parent)
    hasElement(extension:C -> instance)              <->  instanceOf(instance -> C)
    subClassOf(abstract:child -> abstract:parent)    <->  subClassOf(child -> parent)
    hasExtension(abstract:C -> extension:C)          <->  (no flat pre-image)

``hasExtension`` having no pre-image is the structural variable under test: a
break there is representable (and repairable) only in the dual model.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Set, Tuple

import networkx as nx

from ..graph_utils import get_relations
from ..process1_dual_node import concept_of

logger = logging.getLogger(__name__)

Pair = Tuple[str, str]
Edge = Tuple[str, str, str]


def homologous_flat_removed(
    removed_dual: Set[Edge], config: Optional[Dict[str, Any]] = None
) -> Set[Edge]:
    """Map removed dual edges to the flat edges that carry the same information."""
    rels = get_relations(config)
    sup, he, hx = rels["superset_of"], rels["has_element"], rels["has_extension"]
    sub, inst = rels["sub_class_of"], rels["instance_of"]
    out: Set[Edge] = set()
    for (u, v, r) in removed_dual:
        if r == sup:        # extension:parent -> extension:child  =>  child -subClassOf-> parent
            out.add((concept_of(v) or v, concept_of(u) or u, sub))
        elif r == he:       # extension:C -> instance  =>  instance -instanceOf-> C
            out.add((v, concept_of(u) or u, inst))
        elif r == sub:      # abstract:child -> abstract:parent  =>  child -subClassOf-> parent
            out.add((concept_of(u) or u, concept_of(v) or v, sub))
        elif r == hx:       # hasExtension has no flat pre-image
            continue
    return out


def apply_flat_removed(flat: nx.DiGraph, removed_flat: Set[Edge]) -> nx.DiGraph:
    """A copy of ``flat`` with the given typed edges removed (input untouched)."""
    g = flat.copy()
    for (u, v, r) in removed_flat:
        if g.has_edge(u, v) and g.edges[u, v].get("relation") == r:
            g.remove_edge(u, v)
    return g


def single_node_valid_pairs(
    flat: nx.DiGraph, expected, config: Optional[Dict[str, Any]] = None
) -> Set[Pair]:
    """Expected ``(instance, concept)`` pairs derivable by flat taxonomic closure:
    ``c`` is a direct concept of ``i`` or one of its ``subClassOf`` ancestors."""
    rels = get_relations(config)
    sub, inst = rels["sub_class_of"], rels["instance_of"]
    tax = nx.DiGraph()
    direct: Dict[str, Set[str]] = {}
    for u, v, d in flat.edges(data=True):
        rel = d.get("relation")
        if rel == sub:            # child -> parent
            tax.add_edge(u, v)
        elif rel == inst:         # instance -> concept
            direct.setdefault(u, set()).add(v)
    valid: Set[Pair] = set()
    for (i, c) in expected:
        concepts = direct.get(i, set())
        if c in concepts:
            valid.add((i, c))
            continue
        for c0 in concepts:
            if tax.has_node(c0) and c in nx.descendants(tax, c0):  # child->parent reach = ancestors
                valid.add((i, c))
                break
    return valid
