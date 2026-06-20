"""Tests for the evidence-based per-proposal confidence.

Two things must hold for the corroboration scorer to be a sound precision–recall
knob: (1) it must *reduce* to the per-type baseline at ``floor=1`` (so the
ablation is honest), and (2) it must score a structurally-corroborated proposal
above a lone / fabricated one — but only when redundancy actually exists (a tree
gives no corroboration to anything).
"""
import copy

import networkx as nx
import pytest

from src import graph_utils as gu
from src import process1_dual_node as p1
from src import process2_thread_enforce as p2
from src.experiment import corruption as C
from src.experiment import metrics_ext as M
from src.experiment import oracle as O
from src.experiment import util
from src.experiment.confidence import corroboration_confidence, per_type_confidence
from src.process2_thread_enforce import EdgeProposal


@pytest.fixture
def cfg():
    return gu.load_config("config.yaml")


def _cfg_theta(cfg, theta):
    c = copy.deepcopy(cfg)
    c["process2"]["repair"]["min_confidence"] = theta
    c["process2"]["repair"]["max_new_edges"] = 1_000_000
    return c


# --- builder hygiene -------------------------------------------------------
def test_floor_out_of_range_rejected():
    with pytest.raises(ValueError):
        corroboration_confidence(floor=1.5)
    with pytest.raises(ValueError):
        corroboration_confidence(floor=-0.1)


def test_per_type_builder_returns_prior():
    g = nx.DiGraph()
    score = per_type_confidence()(g, None)
    prop = EdgeProposal("a", "b", "supersetOf", 0.9, "r",
                        parent_concept="a", child_concept="b")
    assert score(prop) == 0.9


# --- the scorer separates corroborated from uncorroborated ------------------
def _mini_dual_graph():
    """C ⊑ M ⊑ P with a REDUNDANT direct C ⊑ P, plus an isolated W.

    subClassOf (abstract layer): C->M, M->P, C->P (redundant).
    supersetOf (extension layer): P->M, M->C  (P reaches C another way).
    So a proposed supersetOf(ext:P -> ext:C) is corroborated by BOTH layers;
    a proposed supersetOf(ext:W -> ext:C) is corroborated by neither.
    """
    g = nx.DiGraph()
    for c in ("P", "M", "C", "W"):
        g.add_node(f"abstract:{c}", kind="abstract", concept=c)
        g.add_node(f"extension:{c}", kind="extension", concept=c)
    g.add_edge("abstract:C", "abstract:M", relation="subClassOf")
    g.add_edge("abstract:M", "abstract:P", relation="subClassOf")
    g.add_edge("abstract:C", "abstract:P", relation="subClassOf")  # redundant
    g.add_edge("extension:P", "extension:M", relation="supersetOf")
    g.add_edge("extension:M", "extension:C", relation="supersetOf")
    return g


def test_corroborated_scores_above_uncorroborated(cfg):
    g = _mini_dual_graph()
    score = corroboration_confidence(floor=0.0)(g, cfg)
    good = EdgeProposal("extension:P", "extension:C", "supersetOf", 0.9, "r",
                        parent_concept="P", child_concept="C")
    lone = EdgeProposal("extension:W", "extension:C", "supersetOf", 0.9, "r",
                        parent_concept="W", child_concept="C")
    s_good, s_lone = score(good), score(lone)
    assert s_lone == 0.0                       # no surviving route entails W ⊒ C
    assert s_good > s_lone                      # redundancy lifts the real proposal
    # k = 2 (one subClassOf alt route + one supersetOf route) -> 0.9*(1-1/3)
    assert s_good == pytest.approx(0.9 * (1 - 1 / 3))


def test_layer_toggles(cfg):
    g = _mini_dual_graph()
    good = EdgeProposal("extension:P", "extension:C", "supersetOf", 0.9, "r",
                        parent_concept="P", child_concept="C")
    only_sub = corroboration_confidence(floor=0.0, use_superset=False)(g, cfg)(good)
    only_sup = corroboration_confidence(floor=0.0, use_subclass=False)(g, cfg)(good)
    # each layer alone contributes k=1 -> 0.9*(1-1/2)=0.45
    assert only_sub == pytest.approx(0.45)
    assert only_sup == pytest.approx(0.45)


def test_non_superset_keeps_prior(cfg):
    g = _mini_dual_graph()
    score = corroboration_confidence(floor=0.0)(g, cfg)
    hx = EdgeProposal("abstract:C", "extension:C", "hasExtension", 1.0, "r", concept="C")
    he = EdgeProposal("extension:C", "i1", "hasElement", 0.55, "r",
                      leaf_concept="C", instance="i1")
    assert score(hx) == 1.0    # structurally implied; not modulated
    assert score(he) == 0.55   # membership; separately conservative


# --- tree null: corroboration cannot help when there is no redundancy -------
def test_tree_gives_no_corroboration(cfg):
    flat = gu.build_synthetic_dag(levels=4, branching=3, mi_rate=0.0, seed=1, config=cfg)
    g0 = p1.expand_to_dual_nodes(flat, config=cfg)
    score = corroboration_confidence(floor=0.0)(g0, cfg)
    # every supersetOf proposal on a tree scores 0 (no alternative route exists)
    sup_props = [
        EdgeProposal(u, v, "supersetOf", 0.9, "r",
                     parent_concept=p1.concept_of(u), child_concept=p1.concept_of(v))
        for u, v, d in g0.edges(data=True) if d.get("relation") == "supersetOf"
    ]
    assert sup_props
    assert all(score(p) == 0.0 for p in sup_props)


# --- integration: floor=1 reproduces the baseline exactly -------------------
def test_floor_one_reproduces_baseline(cfg):
    flat = gu.build_synthetic_dag(levels=4, branching=3, mi_rate=0.4, seed=2, config=cfg)
    g0 = p1.expand_to_dual_nodes(flat, config=cfg)
    O.freeze_oracle(g0, cfg)  # sanity: pristine graph is complete
    rng = util.make_rng("conf", 1)
    g_dmg, _, _ = C.corrupt(g0, "E-SO", 0.2, rng, config=cfg, guidance_rewire_rate=0.3)
    for theta in (0.0, 0.5, 0.75, 0.9):
        base = M.added_edges(p2.repair_threads(copy.deepcopy(g_dmg), config=_cfg_theta(cfg, theta)))
        floored = M.added_edges(p2.repair_threads(
            copy.deepcopy(g_dmg), config=_cfg_theta(cfg, theta),
            confidence=corroboration_confidence(floor=1.0)))
        assert base == floored


# --- integration: graded spread + fabrications scored low -------------------
def test_corroboration_grades_and_demotes_fabrications(cfg):
    flat = gu.build_synthetic_dag(levels=5, branching=3, mi_rate=0.4, seed=3, config=cfg)
    g0 = p1.expand_to_dual_nodes(flat, config=cfg)
    orc = O.freeze_oracle(g0, cfg)
    pe = set(orc.pristine_edges)
    rng = util.make_rng("conf", 7)
    g_dmg, _, _ = C.corrupt(g0, "E-SO", 0.2, rng, config=cfg, guidance_rewire_rate=0.4)

    res = p2.repair_threads(copy.deepcopy(g_dmg), config=_cfg_theta(cfg, 0.0),
                            confidence=corroboration_confidence(floor=0.0))
    sup = [r for r in res.added if r["relation"] == "supersetOf"]
    # graded, not on/off: more than one distinct supersetOf confidence appears
    assert len({round(r["confidence"], 3) for r in sup}) > 1
    real = [r["confidence"] for r in sup if (r["u"], r["v"], r["relation"]) in pe]
    fab = [r["confidence"] for r in sup if (r["u"], r["v"], r["relation"]) not in pe]
    assert real and fab
    # fabricated (rewiring-induced) edges are demoted relative to real ones
    assert max(fab) <= max(real)
    assert sum(fab) / len(fab) < sum(real) / len(real)


def test_corroboration_is_deterministic(cfg):
    flat = gu.build_synthetic_dag(levels=4, branching=3, mi_rate=0.4, seed=4, config=cfg)
    g0 = p1.expand_to_dual_nodes(flat, config=cfg)
    runs = []
    for _ in range(2):
        rng = util.make_rng("conf", 9)
        g_dmg, _, _ = C.corrupt(g0, "E-SO", 0.3, rng, config=cfg, guidance_rewire_rate=0.3)
        runs.append(M.added_edges(p2.repair_threads(
            g_dmg, config=_cfg_theta(cfg, 0.3),
            confidence=corroboration_confidence(floor=0.2))))
    assert runs[0] == runs[1]
