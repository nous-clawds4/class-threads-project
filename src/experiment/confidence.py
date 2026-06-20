"""Evidence-based, per-proposal confidence for repair.

The default repair scorer assigns confidence per edge *type* (1.0 / 0.9 / 0.55),
so a good ``supersetOf`` proposal and a fabricated one (from a rewired/distractor
``subClassOf`` edge) carry identical confidence and cannot be separated. The
threshold θ can therefore only switch repair on or off — it cannot *selectively*
filter fabrications (see ``docs/experiment-design.md`` §11, Fig 3).

This module provides a builder that scores *each proposed edge* by its structural
**corroboration**: how much INDEPENDENT surviving structure entails the link it
would assert. The intuition, and its limits:

* On a **tree**, every taxonomy edge is a bridge — removing it severs the only
  route between its endpoints, so no surviving structure can vouch for (or refute)
  a proposal to re-add it. Corroboration is ~0 for *everything*, good or bad, and
  the knob cannot discriminate.
* On a **DAG with multiple inheritance**, a child reaches an ancestor by more than
  one route. A genuinely-missing edge is then redundantly entailed by the
  surviving alternative routes (high corroboration); a lone rewired edge to a
  random wrong parent has no alternative route (≈0 corroboration). The knob can
  now rank real proposals above fabricated ones.

So the precision–recall power of the knob scales with the graph's redundancy — a
*measurable* property — which is the headline of this study, not an artifact.

Two independent layers supply corroboration, and we use both:

1. ``subClassOf`` redundancy — alternative *taxonomy* routes ``child ↝ parent``
   that do not use the (rewired) direct edge.
2. ``supersetOf`` redundancy — the surviving *extension* layer (≈80% intact under
   the E-SO arm, and corrupted independently of ``subClassOf`` rewiring) routes
   ``parent ↝ child``. This is the cleanest "independent witness" because the
   layer being queried is not the layer that produced the proposal.

The two are summed into a count ``k`` (capped), mapped through a saturating
``r = 1 − 1/(1+k) ∈ {0, .5, .667, .75, …}``, and blended with the per-type prior::

    confidence = base_prior × (floor + (1 − floor) · r)

``floor`` interpolates to the baseline: ``floor = 1`` reproduces the per-type
scorer exactly (the ablation), ``floor = 0`` is pure corroboration. Only the
``supersetOf`` layer is modulated — that is where rewiring induces fabrications;
``hasExtension`` (no flat pre-image, structurally implied) and ``hasElement``
(membership, separately conservative) keep their per-type prior.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Set

import networkx as nx

from ..graph_utils import get_relations
from ..process1_dual_node import concept_of
from ..process2_thread_enforce import ConfidenceBuilder, ConfidenceFn, EdgeProposal


def per_type_confidence() -> ConfidenceBuilder:
    """Builder for the baseline scorer: confidence is the per-edge-type prior.

    Equivalent to passing ``confidence=None`` to ``repair_threads``; provided so
    experiments can name the ablation baseline explicitly.
    """
    def build(graph: nx.DiGraph, config: Optional[Dict[str, Any]] = None) -> ConfidenceFn:
        return lambda proposal: proposal.base_confidence
    return build


def corroboration_confidence(
    *,
    floor: float = 0.0,
    cap: int = 3,
    use_subclass: bool = True,
    use_superset: bool = True,
) -> ConfidenceBuilder:
    """Builder for the evidence-based scorer.

    Parameters
    ----------
    floor:
        Weight on the per-type prior for a *wholly uncorroborated* edge, in
        ``[0, 1]``. ``floor = 1`` recovers the per-type baseline; ``floor = 0``
        is pure corroboration (uncorroborated structural edges score 0).
    cap:
        Saturation cap on the corroboration count ``k`` (diminishing returns).
    use_subclass / use_superset:
        Toggle each corroboration layer (for the layer ablation).
    """
    if not 0.0 <= floor <= 1.0:
        raise ValueError(f"floor must be in [0, 1], got {floor}")

    def build(graph: nx.DiGraph, config: Optional[Dict[str, Any]] = None) -> ConfidenceFn:
        rels = get_relations(config)
        sub, sup = rels["sub_class_of"], rels["superset_of"]

        # Concept-name taxonomy from surviving subClassOf edges (child -> parent).
        tax = nx.DiGraph()
        for u, v, d in graph.edges(data=True):
            if d.get("relation") == sub:
                tax.add_edge(concept_of(u) or u, concept_of(v) or v)
        # Concept-name DAG from surviving supersetOf edges (parent -> child); the
        # extension layer, corrupted independently of subClassOf rewiring.
        supg = nx.DiGraph()
        for u, v, d in graph.edges(data=True):
            if d.get("relation") == sup:
                supg.add_edge(concept_of(u) or u, concept_of(v) or v)

        _tax_desc: Dict[str, Set[str]] = {}
        _sup_desc: Dict[str, Set[str]] = {}

        def tax_desc(n: str) -> Set[str]:
            if n not in _tax_desc:
                _tax_desc[n] = nx.descendants(tax, n) if tax.has_node(n) else set()
            return _tax_desc[n]

        def sup_desc(n: str) -> Set[str]:
            if n not in _sup_desc:
                _sup_desc[n] = nx.descendants(supg, n) if supg.has_node(n) else set()
            return _sup_desc[n]

        def corroboration(child: str, parent: str) -> int:
            """# independent surviving routes that entail ``child ⊑ parent``,
            always EXCLUDING the direct edge under proposal (an alternative route
            has length ≥ 2). Both layers are scanned the same way: count the
            neighbours through which the far endpoint is still reachable."""
            k = 0
            if use_subclass and tax.has_node(child) and tax.has_node(parent):
                # Each OTHER parent of `child` from which `parent` is reachable is
                # an alternative route child -> nb -> ... -> parent (multiple
                # inheritance). ``nb != parent`` excludes the direct edge.
                for nb in tax.successors(child):
                    if nb != parent and parent in tax_desc(nb):
                        k += 1
            if use_superset and supg.has_node(parent) and supg.has_node(child):
                # The surviving extension layer (corrupted independently of the
                # subClassOf rewiring) routes parent -> s -> ... -> child; the
                # direct parent->child edge (``s != child``) is excluded.
                for s in supg.successors(parent):
                    if s != child and child in sup_desc(s):
                        k += 1
            return k

        def score(proposal: EdgeProposal) -> float:
            if (proposal.relation == sup
                    and proposal.parent_concept is not None
                    and proposal.child_concept is not None):
                k = min(corroboration(proposal.child_concept, proposal.parent_concept), cap)
                r = 1.0 - 1.0 / (1.0 + k)
                return proposal.base_confidence * (floor + (1.0 - floor) * r)
            # Other edge types keep their per-type prior (no rewire-fabrication path).
            return proposal.base_confidence

        return score

    return build
