"""Tests for Class Thread logic: expansion, validation, repair, and metrics.

The Class Thread pattern under test is the forward directed path::

    abstract:C -hasExtension-> extension:C -supersetOf*-> extension:L -hasElement-> instance
"""

import networkx as nx
import pytest

from src import graph_utils as gu
from src import metrics as m
from src import process1_dual_node as p1
from src import process2_thread_enforce as p2


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def cfg():
    return gu.load_config("config.yaml")


@pytest.fixture
def flat(cfg):
    return gu.build_synthetic_graph(config=cfg)


@pytest.fixture
def dual(flat, cfg):
    return p1.expand_to_dual_nodes(flat, config=cfg)


@pytest.fixture
def expected(dual, cfg):
    """Ground-truth (instance, concept) pairs from the pristine dual graph."""
    return set(p2.validate_threads(dual, config=cfg).expected)


# ---------------------------------------------------------------------------
# Process 1 — dual-node expansion
# ---------------------------------------------------------------------------
def test_id_helpers_roundtrip():
    assert p1.abstract_id("dog") == "abstract:dog"
    assert p1.extension_id("dog") == "extension:dog"
    assert p1.concept_of("abstract:dog") == "dog"
    assert p1.concept_of("extension:dog") == "dog"
    assert p1.concept_of("rex") is None


def test_expansion_creates_dual_nodes(flat, dual):
    n_concepts = sum(1 for _, d in flat.nodes(data=True) if d["kind"] == "concept")
    abstracts = [n for n, d in dual.nodes(data=True) if d["kind"] == "abstract"]
    extensions = [n for n, d in dual.nodes(data=True) if d["kind"] == "extension"]
    assert len(abstracts) == n_concepts
    assert len(extensions) == n_concepts
    # every abstract has exactly one hasExtension edge to its extension
    for a in abstracts:
        c = p1.concept_of(a)
        assert dual.has_edge(a, p1.extension_id(c))
        assert dual.edges[a, p1.extension_id(c)]["relation"] == "hasExtension"


def test_expansion_does_not_mutate_input(flat):
    before_nodes, before_edges = flat.number_of_nodes(), flat.number_of_edges()
    _ = p1.expand_to_dual_nodes(flat)
    assert flat.number_of_nodes() == before_nodes
    assert flat.number_of_edges() == before_edges


def test_superset_edges_point_general_to_specific(dual):
    # animal is more general than mammal -> edge must run animal -> mammal
    assert dual.has_edge("extension:animal", "extension:mammal")
    assert dual.edges["extension:animal", "extension:mammal"]["relation"] == "supersetOf"
    assert not dual.has_edge("extension:mammal", "extension:animal")


def test_haselement_points_extension_to_instance(dual):
    assert dual.has_edge("extension:dog", "rex")
    assert dual.edges["extension:dog", "rex"]["relation"] == "hasElement"


def test_abstract_layer_mirror_uses_subclassof(dual):
    # abstract mirror keeps the conventional child -> parent subClassOf direction
    assert dual.has_edge("abstract:dog", "abstract:mammal")
    assert dual.edges["abstract:dog", "abstract:mammal"]["relation"] == "subClassOf"


def test_select_concepts_strategies(flat, cfg):
    all_sel = p1.select_concepts(flat, config={"process1": {"strategy": "all"}})
    assert "dog" in all_sel and "vehicle" in all_sel

    list_sel = p1.select_concepts(
        flat, config={"process1": {"strategy": "list", "concepts": ["dog", "nope"]}}
    )
    assert list_sel == {"dog"}  # unknown 'nope' dropped

    explicit = p1.select_concepts(flat, selected=["cat"])
    assert explicit == {"cat"}


def test_partial_expansion_stays_connected(flat):
    # Expanding only 'dog' should still leave a traceable structure.
    dual = p1.expand_to_dual_nodes(flat, selected=["dog"])
    assert dual.has_node("abstract:dog") and dual.has_node("extension:dog")
    assert dual.has_node("mammal")  # unexpanded concept keeps single node
    # mammal (general) -> extension:dog (specific) via supersetOf
    assert dual.has_edge("mammal", "extension:dog")


def test_partial_expansion_threads_from_unexpanded_concepts(flat, cfg):
    """Regression: under partial (strategy='list') expansion, a valid thread must
    be able to ORIGINATE at an unexpanded concept, which plays both roles and so
    has no ``hasExtension`` edge. Previously coverage collapsed because
    ``find_thread`` always required a ``hasExtension`` first hop.
    """
    # Only 'animal' and 'dog' are expanded; mammal/cat/truck/... stay single nodes.
    dual = p1.expand_to_dual_nodes(flat, selected=["animal", "dog"])
    # Snapshot ground truth from the pristine partial graph (undamaged).
    expected = set(p2.validate_threads(dual, config=cfg).expected)

    report = p2.validate_threads(dual, config=cfg, expected=expected)
    # Every expected (instance, concept) pair threads despite partial expansion.
    assert report.coverage == 1.0
    assert report.broken == set()

    # Threads that ORIGINATE at an unexpanded concept node (no abstract:/hasExtension).
    assert p2.find_thread(dual, "mammal", "rex", cfg) is not None
    assert p2.find_thread(dual, "cat", "whiskers", cfg) is not None
    assert p2.find_thread(dual, "truck", "big_rig", cfg) is not None
    # An expanded concept still threads through unexpanded relays.
    assert p2.find_thread(dual, "animal", "rex", cfg) is not None
    # No false positives: a genuine non-membership still returns None.
    assert p2.find_thread(dual, "cat", "rex", cfg) is None


# ---------------------------------------------------------------------------
# Thread search / validation
# ---------------------------------------------------------------------------
def test_find_thread_matches_pattern(dual, cfg):
    path = p2.find_thread(dual, "animal", "rex", cfg)
    assert path == [
        "abstract:animal", "extension:animal", "extension:mammal",
        "extension:dog", "rex",
    ]
    labels = [dual.edges[path[i], path[i + 1]]["relation"] for i in range(len(path) - 1)]
    assert labels[0] == "hasExtension"
    assert labels[-1] == "hasElement"
    assert all(lbl == "supersetOf" for lbl in labels[1:-1])


def test_zero_superset_thread(dual, cfg):
    # Direct membership: abstract:dog -hasExtension-> extension:dog -hasElement-> rex
    path = p2.find_thread(dual, "dog", "rex", cfg)
    assert path == ["abstract:dog", "extension:dog", "rex"]


def test_has_thread_negative(dual, cfg):
    # rex is not a vehicle, and not a cat
    assert not p2.has_thread(dual, "vehicle", "rex", cfg)
    assert not p2.has_thread(dual, "cat", "rex", cfg)


def test_pristine_full_coverage(dual, cfg):
    report = p2.validate_threads(dual, config=cfg)
    assert report.coverage == 1.0
    assert report.broken == set()


def test_expected_pairs_dual_matches_reference(flat, dual, cfg):
    from_dual = p2.expected_pairs(dual, config=cfg)
    from_flat = p2.expected_pairs(dual, config=cfg, reference=flat)
    assert from_dual == from_flat


def test_expected_pairs_include_ancestors(dual, cfg):
    exp = p2.expected_pairs(dual, config=cfg)
    # rex (a dog) should be expected to thread to dog, mammal, and animal
    assert {("rex", "dog"), ("rex", "mammal"), ("rex", "animal")} <= exp
    # ...but not to unrelated concepts
    assert ("rex", "vehicle") not in exp


# ---------------------------------------------------------------------------
# Damage detection + repair
# ---------------------------------------------------------------------------
def test_broken_thread_detected(dual, cfg, expected):
    dual.remove_edge("extension:animal", "extension:mammal")
    report = p2.validate_threads(dual, config=cfg, expected=expected)
    assert report.coverage < 1.0
    assert ("rex", "animal") in report.broken
    # direct dog membership is unaffected
    assert ("rex", "dog") in report.valid


def test_repair_restores_structural_edges(dual, cfg, expected):
    dual.remove_edge("extension:animal", "extension:mammal")
    dual.remove_edge("abstract:car", "extension:car")
    before = p2.validate_threads(dual, config=cfg, expected=expected)
    assert before.broken

    result = p2.repair_threads(dual, config=cfg, report=before)
    after = p2.validate_threads(dual, config=cfg, expected=expected)
    assert after.coverage == 1.0
    # the two structural edges were re-added
    added = {(r["u"], r["v"], r["relation"]) for r in result.added}
    assert ("extension:animal", "extension:mammal", "supersetOf") in added
    assert ("abstract:car", "extension:car", "hasExtension") in added
    # repaired edges are tagged
    assert dual.edges["abstract:car", "extension:car"].get("repaired") is True


def test_repair_is_conservative_about_membership(dual, cfg, expected):
    # Removing an instance's only hasElement = lost membership.
    dual.remove_edge("extension:dog", "rex")

    # Default threshold (0.75) must NOT invent the membership.
    result = p2.repair_threads(dual, config=cfg, expected=expected)
    assert result.added == []
    assert any(r["relation"] == "hasElement" for r in result.skipped)
    assert p2.validate_threads(dual, config=cfg, expected=expected).broken

    # Lowering the threshold allows asserting the membership.
    cfg_low = gu.load_config("config.yaml")
    cfg_low["process2"]["repair"]["min_confidence"] = 0.5
    result2 = p2.repair_threads(dual, config=cfg_low, expected=expected)
    assert any(r["relation"] == "hasElement" for r in result2.added)
    assert p2.validate_threads(dual, config=cfg_low, expected=expected).coverage == 1.0


def test_repair_respects_max_new_edges_cap(dual, cfg, expected):
    dual.remove_edge("extension:animal", "extension:mammal")
    dual.remove_edge("extension:vehicle", "extension:car")
    cfg_cap = gu.load_config("config.yaml")
    cfg_cap["process2"]["repair"]["max_new_edges"] = 1

    result = p2.repair_threads(dual, config=cfg_cap, expected=expected)
    assert len(result.added) == 1
    assert any("max_new_edges" in r.get("skip_reason", "") for r in result.skipped)


def test_repair_disabled(dual, cfg, expected):
    dual.remove_edge("extension:animal", "extension:mammal")
    cfg_off = gu.load_config("config.yaml")
    cfg_off["process2"]["repair"]["enabled"] = False
    result = p2.repair_threads(dual, config=cfg_off, expected=expected)
    assert result.added == [] and result.skipped == []


# ---------------------------------------------------------------------------
# Metrics + flat-vs-threaded queries
# ---------------------------------------------------------------------------
def test_compute_metrics_shape(dual, cfg):
    met = m.compute_metrics(dual, config=cfg)
    assert met["coverage"] == 1.0
    assert met["num_instances"] == 8
    assert met["avg_valid_threads_per_instance"] > 1.0  # ancestors add threads


def test_enforcement_delta(dual, cfg, expected):
    dual.remove_edge("extension:animal", "extension:mammal")
    before = p2.validate_threads(dual, config=cfg, expected=expected)
    p2.repair_threads(dual, config=cfg, report=before)
    after = p2.validate_threads(dual, config=cfg, expected=expected)
    delta = m.enforcement_delta(before, after)
    assert delta["broken_before"] == 3
    assert delta["broken_after"] == 0
    assert delta["threads_repaired"] == 3


def test_flat_query_misses_subclass_instances(flat, dual, cfg):
    qc = m.query_comparison(flat, dual, "animal", cfg)
    # naive 1-hop finds no direct 'animal' instances...
    assert qc["flat_direct"] == []
    # ...but the threaded query finds them all, matching the correct closure.
    assert qc["threaded"] == qc["flat_transitive"]
    assert set(qc["threaded"]) == {"fido", "freckles", "rex", "sammy", "whiskers"}


def test_concepts_via_threads(dual, cfg):
    assert p2.concepts_via_threads(dual, "rex", cfg) == {"dog", "mammal", "animal"}
    assert p2.concepts_via_threads(dual, "my_tesla", cfg) == {"car", "vehicle"}
