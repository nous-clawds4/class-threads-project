"""Tests for the closure-aware (semantic) precision metric.

Exact-typed-edge precision penalizes a repaired edge that asserts a TRUE fact via
a transitive shortcut never materialized in the pristine graph. The closure-aware
metric credits such edges, so it must (a) credit true transitive shortcuts, (b)
still reject genuinely-false edges, and (c) dominate exact precision everywhere.
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
from src.experiment.confidence import corroboration_confidence


@pytest.fixture
def cfg():
    return gu.load_config("config.yaml")


# Pristine: C ⊑ M ⊑ P (so ancestors(C) = {M, P}); instance i is a member of C.
PRISTINE = {
    ("abstract:C", "abstract:M", "subClassOf"),
    ("abstract:M", "abstract:P", "subClassOf"),
    ("extension:P", "extension:M", "supersetOf"),
    ("extension:M", "extension:C", "supersetOf"),
    ("extension:C", "i", "hasElement"),
}
EXPECTED = {("i", "C"), ("i", "M"), ("i", "P")}


def test_true_transitive_shortcut_credited(cfg):
    oracle = M.build_semantic_oracle(PRISTINE, EXPECTED, cfg)
    shortcut = ("extension:P", "extension:C", "supersetOf")  # asserts C ⊑ P (true), not exact
    assert M.is_semantically_correct(shortcut, oracle)        # closure: correct
    assert shortcut not in PRISTINE                           # exact: would be "hallucination"
    assert M.edge_precision({shortcut}, PRISTINE) == 0.0
    assert M.closure_precision({shortcut}, oracle) == 1.0


def test_false_edge_rejected_by_both(cfg):
    oracle = M.build_semantic_oracle(PRISTINE, EXPECTED, cfg)
    bogus = ("extension:X", "extension:C", "supersetOf")  # X is not an ancestor of C
    assert not M.is_semantically_correct(bogus, oracle)
    assert M.closure_precision({bogus}, oracle) == 0.0


def test_haselement_and_hasextension_semantics(cfg):
    oracle = M.build_semantic_oracle(PRISTINE, EXPECTED, cfg)
    he_true = ("extension:M", "i", "hasElement")            # (i, M) in expected closure
    he_false = ("extension:P", "x", "hasElement")           # x is no member of anything
    hx = ("abstract:C", "extension:C", "hasExtension")      # links C's two halves
    assert M.is_semantically_correct(he_true, oracle)
    assert not M.is_semantically_correct(he_false, oracle)
    assert M.is_semantically_correct(hx, oracle)


def test_closure_precision_mixed_set(cfg):
    oracle = M.build_semantic_oracle(PRISTINE, EXPECTED, cfg)
    added = {("extension:P", "extension:C", "supersetOf"),   # true shortcut
             ("extension:X", "extension:C", "supersetOf")}   # false
    assert M.closure_precision(added, oracle) == 0.5
    assert M.closure_hallucination_rate(added, oracle) == 0.5
    assert M.closure_precision(set(), oracle) is None


def test_closure_dominates_exact_on_real_repair(cfg):
    """On an actual corrupted-then-repaired DAG, closure precision >= exact."""
    flat = gu.build_synthetic_dag(levels=5, branching=3, mi_rate=0.4, seed=5, config=cfg)
    g0 = p1.expand_to_dual_nodes(flat, config=cfg)
    orc = O.freeze_oracle(g0, cfg)
    sem = M.build_semantic_oracle(orc.pristine_edges, orc.expected, cfg)
    seen_gap = False
    for seed in range(6):
        rng = util.make_rng("clo", seed)
        g_dmg, _, _ = C.corrupt(g0, "E-SO", 0.2, rng, config=cfg, guidance_rewire_rate=0.3)
        cc = copy.deepcopy(cfg)
        cc["process2"]["repair"]["min_confidence"] = 0.0
        cc["process2"]["repair"]["max_new_edges"] = 1_000_000
        res = p2.repair_threads(g_dmg, config=cc, confidence=corroboration_confidence(floor=0.0))
        added = M.added_edges(res)
        exact = M.edge_precision(added, orc.pristine_edges)
        clo = M.closure_precision(added, sem)
        if exact is not None:
            assert clo >= exact - 1e-9          # semantic never worse than exact
            if clo > exact + 1e-9:
                seen_gap = True
    assert seen_gap, "expected at least one true transitive shortcut credited by closure"
