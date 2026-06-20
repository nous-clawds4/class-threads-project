"""Tests for the money-figure mechanism: autonomous repair under adversarial
Tier-B guidance corruption must be ABLE to fabricate (precision < 1), and the
confidence gate must be able to suppress all repair."""
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


def _cfg_theta(cfg, theta):
    c = copy.deepcopy(cfg)
    c["process2"]["repair"]["min_confidence"] = theta
    c["process2"]["repair"]["max_new_edges"] = 1_000_000
    return c


def test_rewiring_can_induce_hallucination(G0, orc, cfg):
    """Under subClassOf rewiring, autonomous repair (no oracle handed in) must be
    able to fabricate at least one edge not in the pristine graph."""
    found = False
    for s in range(5):
        rng = util.make_rng("m", s)
        g_dmg, _, injected = C.corrupt(G0, "E-SO", 0.2, rng, config=cfg, guidance_rewire_rate=0.5)
        assert injected  # rewired (wrong-parent) edges were added
        g = copy.deepcopy(g_dmg)
        res = p2.repair_threads(g, config=_cfg_theta(cfg, 0.75))  # AUTONOMOUS (no expected=)
        halluc = M.hallucination_rate(M.added_edges(res), orc.pristine_edges)
        if halluc and halluc > 0.0:
            found = True
            break
    assert found, "adversarial rewiring should induce at least one fabricated edge"


def test_clean_guidance_repairs_without_fabrication(G0, orc, cfg):
    """With guidance intact (Tier A), autonomous repair fabricates nothing."""
    rng = util.make_rng("m", 7)
    g_dmg, _, _ = C.corrupt(G0, "E-SO", 0.3, rng, config=cfg)  # no rewiring
    g = copy.deepcopy(g_dmg)
    res = p2.repair_threads(g, config=_cfg_theta(cfg, 0.75))
    halluc = M.hallucination_rate(M.added_edges(res), orc.pristine_edges)
    assert halluc in (None, 0.0)  # nothing fabricated


def test_high_threshold_blocks_all_repair(G0, cfg):
    rng = util.make_rng("m", 2)
    g_dmg, _, _ = C.corrupt(G0, "E-SO", 0.3, rng, config=cfg, guidance_rewire_rate=0.3)
    g = copy.deepcopy(g_dmg)
    res = p2.repair_threads(g, config=_cfg_theta(cfg, 1.01))  # above max confidence (1.0)
    assert res.added == []
