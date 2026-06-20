"""Tests for the integrity & repair experiment core (oracle, corruption, metrics).

These exercise the measurement harness directly and do NOT depend on repair
ever hallucinating (the metrics are validated on hand-made inputs), so they are
robust to dataset shape.
"""
import copy

import pytest

from src import graph_utils as gu
from src import process1_dual_node as p1
from src import process2_thread_enforce as p2
from src.experiment import corruption as C
from src.experiment import metrics_ext as M
from src.experiment import oracle as O
from src.experiment import util


@pytest.fixture
def cfg():
    return gu.load_config("config.yaml")


@pytest.fixture
def G0(cfg):
    return p1.expand_to_dual_nodes(gu.build_synthetic_graph(config=cfg), config=cfg)


@pytest.fixture
def orc(G0, cfg):
    return O.freeze_oracle(G0, cfg)


# --- oracle ----------------------------------------------------------------
def test_freeze_requires_complete_graph(G0, cfg):
    O.freeze_oracle(G0, cfg)  # pristine graph: ok
    dmg = copy.deepcopy(G0)
    # Remove a THREAD-bearing edge (supersetOf), not a membership-defining one:
    # this breaks a thread while leaving the expected set intact, so the gate
    # fires. (Removing hasElement would also shrink `expected` -> goalposts move.)
    u, v = next((a, b) for a, b, d in dmg.edges(data=True) if d["relation"] == "supersetOf")
    dmg.remove_edge(u, v)
    with pytest.raises(AssertionError):
        O.freeze_oracle(dmg, cfg)


def test_expected_invariant_under_corruption(G0, orc, cfg):
    assert len(orc.expected) == 20
    rng = util.make_rng("t", 1)
    Gd, _, _ = C.corrupt(G0, "E-MIX", 0.5, rng, config=cfg)
    rep = p2.validate_threads(Gd, config=cfg, expected=set(orc.expected))
    # ground truth never moves, even though the damaged graph would re-derive less
    assert rep.expected == set(orc.expected)


def test_holdout_split_partitions(orc):
    rng = util.make_rng("h", 1)
    train, held = O.holdout_split(set(orc.expected), 0.15, rng)
    assert train.isdisjoint(held)
    assert train | held == set(orc.expected)
    assert len(held) == round(0.15 * len(orc.expected))


# --- corruption ------------------------------------------------------------
def test_corruption_removes_and_is_nonmutating(G0, cfg):
    n_before = G0.number_of_edges()
    n_sup = sum(1 for _, _, d in G0.edges(data=True) if d["relation"] == "supersetOf")
    rng = util.make_rng("t", 2)
    Gd, removed, injected = C.corrupt(G0, "E-SO", 0.5, rng, config=cfg)
    assert injected == set()
    assert all(r == "supersetOf" for (_, _, r) in removed)
    assert len(removed) == round(0.5 * n_sup)
    for (u, v, _) in removed:
        assert not Gd.has_edge(u, v)
    assert G0.number_of_edges() == n_before  # input untouched (Tier A)


def test_tierB_injects_distractor_subclassof(G0, cfg, orc):
    rng = util.make_rng("t", 3)
    Gd, removed, injected = C.corrupt(
        G0, "E-SO", 0.2, rng, config=cfg, guidance_distractor_rate=0.6
    )
    assert injected  # distractors were added
    for (u, v, r) in injected:
        assert r == "subClassOf"
        assert Gd.edges[u, v].get("distractor") is True
        assert (u, v, r) not in orc.pristine_edges  # genuinely fabricated


def test_node_removal_records_incident_edges(G0, cfg):
    rng = util.make_rng("t", 4)
    Gd, removed, _ = C.corrupt(G0, "N-REM", 0.2, rng, config=cfg)
    # removed extension nodes take their incident edges with them
    assert removed
    for (u, v, _) in removed:
        assert not (Gd.has_node(u) and Gd.has_node(v) and Gd.has_edge(u, v))


# --- metrics ---------------------------------------------------------------
def test_edge_precision_and_hallucination():
    pristine = {("a", "b", "supersetOf"), ("c", "d", "supersetOf")}
    added = {("a", "b", "supersetOf"), ("x", "y", "supersetOf")}  # 1 real, 1 fake
    assert M.edge_precision(added, pristine) == 0.5
    assert M.hallucination_rate(added, pristine) == 0.5
    assert M.edge_precision(set(), pristine) is None  # undefined with no adds


def test_recoverable_excludes_non_repairable_relations():
    pristine = {("a", "b", "supersetOf"), ("e", "f", "subClassOf")}
    removed = {("a", "b", "supersetOf"), ("e", "f", "subClassOf")}
    surviving = {"a", "b", "e", "f"}
    # subClassOf is guidance, never re-added by repair -> excluded from recall denom
    assert M.recoverable_removed(removed, pristine, surviving) == {("a", "b", "supersetOf")}


def test_recovery_fraction():
    assert M.recovery_fraction(0.8, 1.0) == 1.0
    assert M.recovery_fraction(1.0, 1.0) is None
    assert abs(M.recovery_fraction(0.5, 0.75) - 0.5) < 1e-9


def test_pair_false_positive_detects_false_membership(G0, orc, cfg):
    g = copy.deepcopy(G0)
    # fabricate a wrong superset: vehicle's extension now contains dogs
    g.add_edge("extension:vehicle", "extension:dog", relation="supersetOf")
    fps = M.pair_false_positives(g, orc.expected, cfg)
    assert any(c == "vehicle" for (_, c) in fps)  # rex now wrongly threads to vehicle


# --- end-to-end / determinism ---------------------------------------------
def test_tierA_repair_is_exact(G0, orc, cfg):
    rng = util.make_rng("e", 1)
    Gd, removed, _ = C.corrupt(G0, "E-SO", 0.3, rng, config=cfg)
    rep = p2.validate_threads(Gd, config=cfg, expected=set(orc.expected))
    assert rep.coverage < 1.0  # damage detected
    res = p2.repair_threads(Gd, config=cfg, expected=set(orc.expected), report=rep)
    added = M.added_edges(res)
    assert M.coverage(Gd, orc.expected, cfg) == 1.0  # fully recovered
    assert M.edge_precision(added, orc.pristine_edges) == 1.0  # exact, no fabrication
    assert M.hallucination_rate(added, orc.pristine_edges) == 0.0


def test_repair_determinism(G0, orc, cfg):
    a = []
    for _ in range(2):
        rng = util.make_rng("d", 9)
        Gd, _, _ = C.corrupt(G0, "E-MIX", 0.4, rng, config=cfg)
        a.append(M.added_edges(p2.repair_threads(Gd, config=cfg, expected=set(orc.expected))))
    assert a[0] == a[1]  # requires the deterministic secondary sort key
