"""Tests for the single-node ablation comparator."""
import pytest

from src import graph_utils as gu
from src import process1_dual_node as p1
from src.experiment import ablation as A
from src.experiment import oracle as O


@pytest.fixture
def cfg():
    return gu.load_config("config.yaml")


@pytest.fixture
def flat(cfg):
    return gu.build_synthetic_graph(config=cfg)


@pytest.fixture
def G0(flat, cfg):
    return p1.expand_to_dual_nodes(flat, config=cfg)


def test_homologous_mapping(cfg):
    removed = {
        ("extension:animal", "extension:mammal", "supersetOf"),
        ("extension:dog", "rex", "hasElement"),
        ("abstract:dog", "extension:dog", "hasExtension"),
    }
    flat_removed = A.homologous_flat_removed(removed, cfg)
    assert ("mammal", "animal", "subClassOf") in flat_removed   # supersetOf reverses to child->parent
    assert ("rex", "dog", "instanceOf") in flat_removed         # hasElement -> instanceOf
    assert all(r != "hasExtension" for (_, _, r) in flat_removed)  # hasExtension has no pre-image
    assert len(flat_removed) == 2


def test_single_node_pristine_full_coverage(flat, G0, cfg):
    exp = set(O.freeze_oracle(G0, cfg).expected)
    assert A.single_node_valid_pairs(flat, exp, cfg) == exp  # closure covers all expected pairs


def test_single_node_drops_pairs_when_subclass_removed(flat, G0, cfg):
    exp = set(O.freeze_oracle(G0, cfg).expected)
    fd = flat.copy()
    fd.remove_edge("mammal", "animal")  # flat subClassOf is child -> parent
    valid = A.single_node_valid_pairs(fd, exp, cfg)
    assert ("rex", "animal") not in valid   # rex loses its path to animal
    assert ("rex", "mammal") in valid        # but keeps the closer ancestor


def test_hasExtension_arm_has_no_flat_preimage(cfg):
    # An all-hasExtension removal maps to an empty flat removal: the single-node
    # model cannot represent the loss (the core asymmetry under test).
    removed = {("abstract:dog", "extension:dog", "hasExtension"),
               ("abstract:cat", "extension:cat", "hasExtension")}
    assert A.homologous_flat_removed(removed, cfg) == set()
