"""Test the Wikidata slice loads from cache, is a consistent DAG with real
multiple inheritance, and expands to full pristine coverage.

Network-free: skips if the cached snapshot is absent (it is committed, so this
runs in practice without hitting WDQS)."""
from pathlib import Path

import networkx as nx
import pytest

from src import graph_utils as gu
from src import process1_dual_node as p1
from src.experiment import oracle as O

CACHE = Path("data/raw/wikidata_Q42889_d3.json")


@pytest.mark.skipif(not CACHE.exists(), reason="wikidata cache snapshot not present")
def test_wikidata_slice_is_consistent_dag():
    from src.experiment.wikidata import build_wikidata_graph

    cfg = gu.load_config("config.yaml")
    flat = build_wikidata_graph(root="Q42889", max_depth=3, config=cfg)  # loads the cache

    # real DAG with multiple inheritance (>=1 concept with >1 subClassOf parent)
    sub = flat.edge_subgraph(
        [(u, v) for u, v, e in flat.edges(data=True) if e["relation"] == "subClassOf"])
    assert nx.is_directed_acyclic_graph(sub)
    multi_parent = [n for n, d in flat.nodes(data=True)
                    if d.get("kind") == "concept"
                    and sum(1 for _, _, e in flat.out_edges(n, data=True)
                            if e["relation"] == "subClassOf") > 1]
    assert len(multi_parent) >= 1

    # expands to a fully-threaded graph (freeze_oracle asserts coverage == 1.0)
    dual = p1.expand_to_dual_nodes(flat, config=cfg)
    orc = O.freeze_oracle(dual, cfg)
    assert len(orc.expected) > 0
