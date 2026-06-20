"""Process 2 — Class Thread Enforcement.

A **Class Thread** is the forward directed path::

    abstract:C --hasExtension--> extension:C --supersetOf--> ... --hasElement--> instance

This module provides:

* **Validation** — for every expected ``(instance, concept)`` pair, check that
  at least one valid thread exists; report coverage and the broken pairs.
* **Repair** — conservatively add the *minimal* missing edges that restore a
  thread, gated by a per-edge confidence heuristic and a hard edge cap.

Ground truth ("expected" pairs)
-------------------------------
An instance ``i`` is *expected* to thread to concept ``C`` iff ``C`` is ``i``'s
directly-asserted concept or one of its taxonomic ancestors. We derive this
from **stable** structure — the abstract-layer ``subClassOf`` taxonomy plus each
instance's direct ``hasElement`` membership — which is independent of the
thread-bearing ``hasExtension`` / ``supersetOf`` edges. So damaging threads does
not move the goalposts. Expected pairs may instead be supplied explicitly
(recommended when a demo damages the graph: snapshot them first) or derived from
a reference flat graph.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import networkx as nx

from .graph_utils import KIND_INSTANCE, get_relations
from .process1_dual_node import abstract_id, concept_of, extension_id

logger = logging.getLogger(__name__)

# (instance, concept) ground-truth pair
Pair = Tuple[str, str]

# Per-edge *type* confidence — the prior used by repair. The gate
# (config.process2.repair.min_confidence, default 0.75) decides which are
# allowed. These are constants per edge TYPE, so good and fabricated proposals of
# the same type are indistinguishable; this is exactly why θ behaves as an on/off
# switch rather than a graded precision–recall knob under the default scorer. A
# pluggable ``confidence`` builder (see ``src/experiment/confidence.py``) can
# instead score each proposal by its evidential support.
CONF_HAS_EXTENSION = 1.0   # structurally implied: abstract:C <-> extension:C
CONF_SUPERSET = 0.9        # taxonomy-attested by the subClassOf hierarchy
CONF_HAS_ELEMENT = 0.55    # asserting a NEW membership — risky, off by default


# ===========================================================================
# Reports
# ===========================================================================
@dataclass
class ThreadReport:
    """Result of validating Class Threads over a graph."""

    expected: Set[Pair]
    valid: Set[Pair]
    broken: Set[Pair]
    per_instance_valid: Dict[str, int]

    @property
    def coverage(self) -> float:
        """Fraction of expected pairs that have at least one valid thread."""
        return len(self.valid) / len(self.expected) if self.expected else 1.0

    def summary(self) -> Dict[str, Any]:
        return {
            "expected_pairs": len(self.expected),
            "valid_pairs": len(self.valid),
            "broken_pairs": len(self.broken),
            "coverage": round(self.coverage, 4),
        }


@dataclass
class RepairResult:
    """Result of a repair pass: edges added and edges considered-but-skipped."""

    added: List[Dict[str, Any]] = field(default_factory=list)
    skipped: List[Dict[str, Any]] = field(default_factory=list)

    def summary(self) -> Dict[str, Any]:
        return {"edges_added": len(self.added), "edges_skipped": len(self.skipped)}


@dataclass(frozen=True)
class EdgeProposal:
    """A single candidate repair edge plus the semantic roles a confidence
    function needs to score it.

    ``base_confidence`` is the per-edge-*type* prior (the default scorer simply
    returns it). The role fields say what the edge would assert, so an
    evidence-based scorer can judge *this* proposal rather than its type:

    * ``supersetOf`` (``extension:P -> extension:C``) asserts ``C ⊑ P`` →
      ``parent_concept=P`` (the superset), ``child_concept=C`` (the subset).
    * ``hasExtension`` (``abstract:C -> extension:C``) → ``concept=C``.
    * ``hasElement`` (``extension:L -> instance``) asserts membership →
      ``leaf_concept=L``, ``instance=i``.
    """

    u: str
    v: str
    relation: str
    base_confidence: float
    reason: str
    parent_concept: Optional[str] = None
    child_concept: Optional[str] = None
    concept: Optional[str] = None
    leaf_concept: Optional[str] = None
    instance: Optional[str] = None


# A confidence function scores one proposal in [0, 1]; a builder specializes it
# to a particular (damaged) graph so structural signals can be precomputed once.
ConfidenceFn = Callable[[EdgeProposal], float]
ConfidenceBuilder = Callable[[nx.DiGraph, Optional[Dict[str, Any]]], ConfidenceFn]


def _per_type_scorer(graph: nx.DiGraph, config: Optional[Dict[str, Any]]) -> ConfidenceFn:
    """Default builder: confidence is the per-edge-type prior (the on/off baseline)."""
    return lambda proposal: proposal.base_confidence


# ===========================================================================
# Thread search (the core validator primitive)
# ===========================================================================
def find_thread(
    graph: nx.DiGraph,
    concept: str,
    instance: str,
    config: Optional[Dict[str, Any]] = None,
) -> Optional[List[str]]:
    """Return a valid thread (node path) from ``concept`` to ``instance``, or None.

    A valid thread is ``abstract:C -hasExtension-> extension:C -supersetOf*->
    extension:L -hasElement-> instance``.

    A concept that was *not* expanded into dual nodes (Process 1 with
    ``strategy="list"``) keeps a single node playing both the abstract and
    extension role, so it has no ``hasExtension`` edge; the thread is allowed to
    originate at that node directly.
    """
    rels = get_relations(config)
    rel_has_ext, rel_sup, rel_he = (
        rels["has_extension"], rels["superset_of"], rels["has_element"],
    )

    start = abstract_id(concept) if graph.has_node(abstract_id(concept)) else concept
    if not graph.has_node(start) or not graph.has_node(instance):
        return None

    # Step 1: leave the abstract node via hasExtension into the extension layer.
    # Each entry is ``(entry_node, initial_path)``.
    entries: List[Tuple[str, List[str]]] = [
        (v, [start, v])
        for v in graph.successors(start)
        if graph.edges[start, v].get("relation") == rel_has_ext
    ]
    # An *unexpanded* concept (start is the plain concept node, no abstract node
    # exists) has no outgoing hasExtension edge but plays the extension role
    # itself, so let the thread originate there. Note: an expanded-but-damaged
    # concept resolves ``start`` to its abstract node (start != concept), so a
    # removed hasExtension edge is still correctly reported as broken.
    if not entries and start == concept:
        entries = [(start, [start])]

    # Step 2: DFS down supersetOf edges, looking for a hasElement to the instance.
    for entry, init_path in entries:
        visited: Set[str] = set()
        stack: List[Tuple[str, List[str]]] = [(entry, init_path)]
        while stack:
            node, path = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            edge = graph.get_edge_data(node, instance)
            if edge and edge.get("relation") == rel_he:
                return path + [instance]
            for nxt in graph.successors(node):
                if graph.edges[node, nxt].get("relation") == rel_sup:
                    stack.append((nxt, path + [nxt]))
    return None


def has_thread(graph, concept, instance, config=None) -> bool:
    """True iff at least one valid Class Thread connects ``concept`` -> ``instance``."""
    return find_thread(graph, concept, instance, config) is not None


# ===========================================================================
# Taxonomy + membership extraction (for ground truth & repair guidance)
# ===========================================================================
def _concept_taxonomy(graph: nx.DiGraph, rels: Dict[str, str]) -> nx.DiGraph:
    """Concept-level taxonomy as a DiGraph of ``child -> parent`` edges.

    Built from the stable abstract-layer ``subClassOf`` mirror; falls back to
    the ``supersetOf`` hierarchy (reversed) if no abstract mirror is present.
    """
    sub, sup = rels["sub_class_of"], rels["superset_of"]
    tax = nx.DiGraph()
    for u, v, d in graph.edges(data=True):
        if d.get("relation") == sub:  # abstract:child -> abstract:parent
            tax.add_edge(concept_of(u) or u, concept_of(v) or v)
    if tax.number_of_edges() == 0:  # fallback: supersetOf is parent -> child
        for u, v, d in graph.edges(data=True):
            if d.get("relation") == sup:
                tax.add_edge(concept_of(v) or v, concept_of(u) or u)
    return tax


def _direct_concepts(graph: nx.DiGraph, instance: str, rels: Dict[str, str]) -> Set[str]:
    """Concepts that hold ``instance`` directly via a ``hasElement`` edge."""
    he = rels["has_element"]
    out: Set[str] = set()
    for p in graph.predecessors(instance):
        if graph.edges[p, instance].get("relation") == he:
            out.add(concept_of(p) or p)
    return out


def _instances(graph: nx.DiGraph) -> List[str]:
    return [n for n, d in graph.nodes(data=True) if d.get("kind") == KIND_INSTANCE]


# ===========================================================================
# Expected pairs (ground truth)
# ===========================================================================
def expected_pairs(
    graph: nx.DiGraph,
    config: Optional[Dict[str, Any]] = None,
    expected: Optional[Set[Pair]] = None,
    reference: Optional[nx.DiGraph] = None,
) -> Set[Pair]:
    """Compute the set of expected ``(instance, concept)`` pairs (ground truth).

    Precedence: explicit ``expected`` > ``reference`` flat graph > derive from
    ``graph`` itself (abstract taxonomy + direct memberships).
    """
    if expected is not None:
        return set(expected)
    rels = get_relations(config)

    if reference is not None:
        return _expected_from_flat(reference, rels)

    tax = _concept_taxonomy(graph, rels)
    pairs: Set[Pair] = set()
    for inst in _instances(graph):
        for c0 in _direct_concepts(graph, inst, rels):
            pairs.add((inst, c0))
            if tax.has_node(c0):
                for anc in nx.descendants(tax, c0):  # child->parent reach = ancestors
                    pairs.add((inst, anc))
    return pairs


def _expected_from_flat(flat: nx.DiGraph, rels: Dict[str, str]) -> Set[Pair]:
    """Derive expected pairs from a flat graph's instanceOf + subClassOf closure."""
    inst_rel, sub = rels["instance_of"], rels["sub_class_of"]
    tax = nx.DiGraph()
    for u, v, d in flat.edges(data=True):
        if d.get("relation") == sub:  # flat: child -> parent
            tax.add_edge(u, v)
    pairs: Set[Pair] = set()
    for u, v, d in flat.edges(data=True):
        if d.get("relation") == inst_rel:  # flat: instance -> concept
            inst, c0 = u, v
            pairs.add((inst, c0))
            if tax.has_node(c0):
                for anc in nx.descendants(tax, c0):
                    pairs.add((inst, anc))
    return pairs


# ===========================================================================
# Validation
# ===========================================================================
def validate_threads(
    graph: nx.DiGraph,
    config: Optional[Dict[str, Any]] = None,
    expected: Optional[Set[Pair]] = None,
    reference: Optional[nx.DiGraph] = None,
) -> ThreadReport:
    """Validate that every expected pair has at least one valid Class Thread."""
    exp = expected_pairs(graph, config=config, expected=expected, reference=reference)
    valid: Set[Pair] = set()
    per_instance: Dict[str, int] = {inst: 0 for inst in _instances(graph)}
    for inst, concept in exp:
        if has_thread(graph, concept, inst, config):
            valid.add((inst, concept))
            per_instance[inst] = per_instance.get(inst, 0) + 1
    broken = exp - valid
    report = ThreadReport(expected=exp, valid=valid, broken=broken,
                          per_instance_valid=per_instance)
    logger.info("Validation: %s", report.summary())
    if broken:
        logger.info("Broken threads (%d): %s", len(broken),
                    ", ".join(f"{i}~{c}" for i, c in sorted(broken)))
    return report


# ===========================================================================
# Repair (conservative, confidence-gated)
# ===========================================================================
def repair_threads(
    graph: nx.DiGraph,
    config: Optional[Dict[str, Any]] = None,
    expected: Optional[Set[Pair]] = None,
    reference: Optional[nx.DiGraph] = None,
    report: Optional[ThreadReport] = None,
    confidence: Optional[ConfidenceBuilder] = None,
) -> RepairResult:
    """Add the minimal missing edges to restore broken threads (in place).

    Conservative by construction:
      * Only the canonical edges implied by the taxonomy are *proposed*.
      * Each proposal carries a confidence; only proposals at or above
        ``config.process2.repair.min_confidence`` are applied.
      * At most ``max_new_edges`` edges are added.

    ``confidence`` is a builder ``(graph, config) -> (EdgeProposal -> float)``.
    It is invoked once on the *damaged* ``graph`` (before any edge is added) so
    structural signals can be precomputed, then scores every proposal. ``None``
    falls back to the per-edge-type prior (the on/off baseline); see
    ``src/experiment/confidence.py`` for an evidence-based corroboration scorer
    that turns θ into a graded precision–recall knob.
    """
    rels = get_relations(config)
    rcfg = ((config or {}).get("process2", {}) or {}).get("repair", {}) or {}
    enabled = rcfg.get("enabled", True)
    min_conf = float(rcfg.get("min_confidence", 0.75))
    max_edges = int(rcfg.get("max_new_edges", 100))
    result = RepairResult()

    if not enabled:
        logger.info("Repair disabled (config.process2.repair.enabled=false).")
        return result

    if report is None:
        report = validate_threads(graph, config=config, expected=expected,
                                  reference=reference)
    if not report.broken:
        logger.info("Repair: nothing to do (no broken threads).")
        return result

    rel_has_ext, rel_sup, rel_he = (
        rels["has_extension"], rels["superset_of"], rels["has_element"],
    )
    tax = _concept_taxonomy(graph, rels)

    # Map each instance to its expected concepts, so repair can recover an
    # instance's leaf even when its direct hasElement membership was lost.
    expected_by_instance: Dict[str, Set[str]] = {}
    for inst, concept in report.expected:
        expected_by_instance.setdefault(inst, set()).add(concept)

    def leaf_concepts(inst: str) -> Set[str]:
        """The instance's most-specific concept(s): direct memberships if any,
        else the minimal (most specific) elements of its expected concepts."""
        direct = _direct_concepts(graph, inst, rels)
        if direct:
            return direct
        e_i = expected_by_instance.get(inst, set())
        # C is most-specific if none of its subclasses are also expected.
        return {c for c in e_i
                if tax.has_node(c) and not (set(nx.ancestors(tax, c)) & e_i)}

    def ext_node(c: str) -> str:
        return extension_id(c) if graph.has_node(extension_id(c)) else c

    def abs_node(c: str) -> str:
        return abstract_id(c) if graph.has_node(abstract_id(c)) else c

    # Collect deduped edge proposals. The per-type base confidence and the
    # semantic roles are a pure function of the (u, v, relation) key, so the
    # first proposal for a key wins (dedup is order-independent).
    proposals: Dict[Tuple[str, str, str], EdgeProposal] = {}

    def propose(u: str, v: str, relation: str, conf: float, reason: str,
                **roles: Optional[str]) -> None:
        if not (graph.has_node(u) and graph.has_node(v)):
            return  # never invent nodes; only edges between existing ones
        if graph.has_edge(u, v) and graph.edges[u, v].get("relation") == relation:
            return  # already present
        key = (u, v, relation)
        if key not in proposals:
            proposals[key] = EdgeProposal(u, v, relation, conf, reason, **roles)

    for inst, concept in sorted(report.broken):
        leaves = leaf_concepts(inst)
        for leaf in leaves:
            # Taxonomy path leaf -> ... -> concept (child .. parent), if `concept`
            # really is an ancestor of the leaf. Reverse => general -> specific.
            if not (tax.has_node(leaf) and tax.has_node(concept)):
                continue
            try:
                up = nx.shortest_path(tax, leaf, concept)
            except nx.NetworkXNoPath:
                continue
            chain = list(reversed(up))  # [concept, ..., leaf]

            # (a) hasExtension at the head concept.
            propose(abs_node(concept), ext_node(concept), rel_has_ext,
                    CONF_HAS_EXTENSION,
                    f"hasExtension structurally implied for '{concept}'",
                    concept=concept)
            # (b) supersetOf for each parent->child step along the chain.
            for parent, child in zip(chain, chain[1:]):
                propose(ext_node(parent), ext_node(child), rel_sup, CONF_SUPERSET,
                        f"supersetOf attested by taxonomy: {parent} -> {child}",
                        parent_concept=parent, child_concept=child)
            # (c) hasElement for the leaf membership (risky; usually gated out).
            propose(ext_node(leaf), inst, rel_he, CONF_HAS_ELEMENT,
                    f"hasElement would assert NEW membership: {leaf} -> {inst}",
                    leaf_concept=leaf, instance=inst)

    # Score every proposal once, against the damaged graph (no edge added yet).
    score: ConfidenceFn = (confidence or _per_type_scorer)(graph, config)
    scored = [(score(p), p) for p in proposals.values()]

    # Apply proposals: highest confidence first, honoring threshold and cap.
    # Secondary key (u, v, relation) makes capped/tied selection deterministic
    # and seed-order-independent (required for reproducible experiments).
    for conf, p in sorted(scored, key=lambda t: (-t[0], t[1].u, t[1].v, t[1].relation)):
        record = {"u": p.u, "v": p.v, "relation": p.relation,
                  "confidence": conf, "base_confidence": p.base_confidence,
                  "reason": p.reason}
        if conf < min_conf:
            record["skip_reason"] = f"confidence {conf} < min_confidence {min_conf}"
            result.skipped.append(record)
            continue
        if len(result.added) >= max_edges:
            record["skip_reason"] = f"max_new_edges cap ({max_edges}) reached"
            result.skipped.append(record)
            continue
        graph.add_edge(p.u, p.v, relation=p.relation, repaired=True)
        result.added.append(record)
        logger.info("Repair +edge %s -[%s]-> %s (conf=%.2f): %s",
                    p.u, p.relation, p.v, conf, p.reason)

    logger.info("Repair: %s", result.summary())
    return result


# ===========================================================================
# Query helpers (used by metrics / notebook for flat-vs-threaded examples)
# ===========================================================================
def instances_via_threads(
    graph: nx.DiGraph, concept: str, config: Optional[Dict[str, Any]] = None
) -> Set[str]:
    """All instances reachable from ``concept`` by a valid Class Thread."""
    return {
        inst for inst in _instances(graph)
        if has_thread(graph, concept, inst, config)
    }


def concepts_via_threads(
    graph: nx.DiGraph, instance: str, config: Optional[Dict[str, Any]] = None
) -> Set[str]:
    """All concepts that thread to ``instance`` (the instance's full type set)."""
    rels = get_relations(config)
    tax = _concept_taxonomy(graph, rels)
    candidates = set(tax.nodes()) | _direct_concepts(graph, instance, rels)
    return {c for c in candidates if has_thread(graph, c, instance, config)}
