"""Wikidata slice loader — the credibility dataset.

Fetches a bounded subclass (``P279``) slice under a root QID plus instances
(``P31``), cleans it into a consistent *acyclic* flat graph (concept/instance
nodes; ``subClassOf`` child->parent + ``instanceOf`` instance->concept edges),
and caches the result to ``data/raw`` so runs are reproducible and offline.

Unlike the near-tree WordNet subset, a Wikidata slice is a genuine DAG with real
multiple inheritance (a class with >1 superclass) — the structural redundancy the
integrity/repair study needs to test evidence-based confidence.
"""
from __future__ import annotations

import json
import logging
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx

from ..graph_utils import KIND_CONCEPT, KIND_INSTANCE, get_relations

logger = logging.getLogger(__name__)

WDQS = "https://query.wikidata.org/sparql"
USER_AGENT = "ClassThreadResearch/0.1 (research; pretty.good.freedom.tech@gmail.com)"


# ---------------------------------------------------------------------------
# SPARQL plumbing
# ---------------------------------------------------------------------------
def _query(sparql: str, max_retries: int = 3) -> List[Dict[str, Any]]:
    body = urllib.parse.urlencode({"query": sparql, "format": "json"}).encode()
    req = urllib.request.Request(
        WDQS, data=body,
        headers={"User-Agent": USER_AGENT, "Accept": "application/sparql-results+json"})
    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(req, timeout=90) as resp:
                return json.loads(resp.read())["results"]["bindings"]
        except Exception as exc:  # noqa: BLE001
            if attempt == max_retries - 1:
                raise
            logger.warning("WDQS retry %d after error: %s", attempt + 1, exc)
            time.sleep(2 * (attempt + 1))
    return []


def _qid(uri: str) -> str:
    return uri.rsplit("/", 1)[-1]


def _chunks(xs: List, n: int):
    for i in range(0, len(xs), n):
        yield xs[i:i + n]


# ---------------------------------------------------------------------------
# Fetch
# ---------------------------------------------------------------------------
def _fetch_subclass_slice(root: str, max_depth: int, max_classes: int
                          ) -> Tuple[Set[str], Set[Tuple[str, str]], Dict[str, str]]:
    """BFS the P279 subclass DAG downward from ``root``; return (classes, child->parent
    edges, labels), then add induced parent edges to capture multiple inheritance."""
    classes: Set[str] = {root}
    labels: Dict[str, str] = {}
    edges: Set[Tuple[str, str]] = set()
    frontier, depth = [root], 0
    while frontier and depth < max_depth and len(classes) < max_classes:
        nxt: List[str] = []
        for batch in _chunks(frontier, 40):
            values = " ".join(f"wd:{q}" for q in batch)
            q = (f'SELECT ?c ?cLabel ?p WHERE {{ VALUES ?p {{ {values} }} '
                 f'?c wdt:P279 ?p. SERVICE wikibase:label '
                 f'{{ bd:serviceParam wikibase:language "en". }} }}')
            for b in _query(q):
                c, p = _qid(b["c"]["value"]), _qid(b["p"]["value"])
                edges.add((c, p))
                if "cLabel" in b:
                    labels[c] = b["cLabel"]["value"]
                if c not in classes and len(classes) < max_classes:
                    classes.add(c)
                    nxt.append(c)
        frontier, depth = nxt, depth + 1
    # Induced multiple-inheritance: extra parents of collected classes that are
    # themselves in the slice.
    for batch in _chunks(list(classes), 40):
        values = " ".join(f"wd:{q}" for q in batch)
        q = f'SELECT ?c ?p WHERE {{ VALUES ?c {{ {values} }} ?c wdt:P279 ?p. }}'
        for b in _query(q):
            c, p = _qid(b["c"]["value"]), _qid(b["p"]["value"])
            if p in classes:
                edges.add((c, p))
    return classes, edges, labels


def _fetch_labels(qids: Set[str]) -> Dict[str, str]:
    labels: Dict[str, str] = {}
    for batch in _chunks(list(qids), 100):
        values = " ".join(f"wd:{q}" for q in batch)
        q = (f'SELECT ?x ?xLabel WHERE {{ VALUES ?x {{ {values} }} '
             f'SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }} }}')
        for b in _query(q):
            if "xLabel" in b:
                labels[_qid(b["x"]["value"])] = b["xLabel"]["value"]
    return labels


def _fetch_instances(leaf_classes: List[str], per_class: int
                     ) -> Tuple[List[Tuple[str, str]], Dict[str, str]]:
    inst_edges: List[Tuple[str, str]] = []
    labels: Dict[str, str] = {}
    for c in leaf_classes:
        q = (f'SELECT ?i ?iLabel WHERE {{ ?i wdt:P31 wd:{c}. SERVICE wikibase:label '
             f'{{ bd:serviceParam wikibase:language "en". }} }} LIMIT {per_class}')
        try:
            rows = _query(q)
        except Exception:  # noqa: BLE001
            continue
        for b in rows:
            i = _qid(b["i"]["value"])
            inst_edges.append((i, c))
            if "iLabel" in b:
                labels[i] = b["iLabel"]["value"]
        time.sleep(0.05)
    return inst_edges, labels


# ---------------------------------------------------------------------------
# Assemble + clean + cache
# ---------------------------------------------------------------------------
def _cache_path(cache_dir: str, root: str, max_depth: int) -> Path:
    return Path(cache_dir) / f"wikidata_{root}_d{max_depth}.json"


def build_wikidata_graph(
    root: str = "Q42889",          # "vehicle"
    max_depth: int = 3,
    max_classes: int = 150,
    instances_per_leaf: int = 4,
    cache_dir: str = "data/raw",
    refresh: bool = False,
    config: Optional[Dict[str, Any]] = None,
) -> nx.DiGraph:
    """Build (or load from cache) a cleaned flat graph from a Wikidata slice."""
    rels = get_relations(config)
    cache = _cache_path(cache_dir, root, max_depth)
    if cache.exists() and not refresh:
        data = json.loads(cache.read_text())
        return nx.node_link_graph(data, directed=True, multigraph=False, edges="edges")

    classes, edges, labels = _fetch_subclass_slice(root, max_depth, max_classes)

    # Class DAG (child -> parent); drop cycle edges, keep the root's component.
    g = nx.DiGraph()
    g.add_nodes_from(classes)
    g.add_edges_from(edges)
    while not nx.is_directed_acyclic_graph(g):
        cycle = nx.find_cycle(g)
        g.remove_edge(cycle[0][0], cycle[0][1])
    comp = next(c for c in nx.weakly_connected_components(g) if root in c)
    classes = set(comp)
    edges = [(u, v) for (u, v) in g.edges() if u in classes and v in classes]

    # Leaves = most-specific classes (never a parent); attach instances there.
    parents = {v for (_, v) in edges}
    leaves = [c for c in classes if c not in parents]
    inst_edges, ilabels = _fetch_instances(leaves, instances_per_leaf)
    inst_edges = [(i, c) for (i, c) in inst_edges if c in classes and i not in classes]
    labels.update(_fetch_labels(classes - set(labels)))
    labels.update(ilabels)

    # Assemble the flat graph in the project's vocabulary.
    flat = nx.DiGraph()
    for c in classes:
        flat.add_node(c, kind=KIND_CONCEPT, label=labels.get(c, c))
    for (i, c) in inst_edges:
        if not flat.has_node(i):
            flat.add_node(i, kind=KIND_INSTANCE, label=labels.get(i, i))
    for (u, v) in edges:
        flat.add_edge(u, v, relation=rels["sub_class_of"])     # child -> parent
    for (i, c) in inst_edges:
        flat.add_edge(i, c, relation=rels["instance_of"])      # instance -> concept

    # Drop concept nodes that ended up isolated (no edges at all).
    isolated = [n for n, d in flat.nodes(data=True)
                if d.get("kind") == KIND_CONCEPT and flat.degree(n) == 0]
    flat.remove_nodes_from(isolated)

    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(nx.node_link_data(flat, edges="edges")))
    logger.info("Wikidata slice %s d%d: %d concepts, %d instances, %d edges (cached %s)",
                root, max_depth,
                sum(1 for _, d in flat.nodes(data=True) if d.get("kind") == KIND_CONCEPT),
                sum(1 for _, d in flat.nodes(data=True) if d.get("kind") == KIND_INSTANCE),
                flat.number_of_edges(), cache)
    return flat
