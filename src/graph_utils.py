"""Graph utilities: configuration, data loading, save/load, and visualization.

This module owns the *vocabulary* and *shape* of the graphs used throughout the
project, so that Processes 1 and 2 and the metrics all agree on conventions.

Conventions
-----------
The graph is a ``networkx.DiGraph``. Every **node** carries a ``kind`` attribute
and every **edge** carries a ``relation`` attribute.

Node kinds:
    * ``concept``    — a flat-graph concept node (pre-expansion)
    * ``instance``   — a concrete individual
    * ``abstract``   — the intension of a concept   (post Process 1)
    * ``extension``  — the set of instances of a concept (post Process 1)

Edge relations (names come from ``config.yaml`` but default to these):
    * ``subClassOf``    child concept  -> parent concept (flat input; abstract-layer mirror)
    * ``instanceOf``    instance       -> concept        (flat input)
    * ``hasExtension``  abstract       -> extension      (created by Process 1)
    * ``supersetOf``    extension      -> extension      (Process 1; general->specific, thread-bearing)
    * ``hasElement``    extension      -> instance       (Process 1; thread-bearing)

A **Class Thread** is the forward directed path
``hasExtension -> (0+ supersetOf) -> hasElement``.

The synthetic data is a small, fully controllable animal/vehicle taxonomy so
that Class Thread behavior is easy to reason about and debug.
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import networkx as nx

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Node-kind constants (single source of truth)
# ---------------------------------------------------------------------------
KIND_CONCEPT = "concept"
KIND_INSTANCE = "instance"
KIND_ABSTRACT = "abstract"
KIND_EXTENSION = "extension"

# ---------------------------------------------------------------------------
# Default relation names. ``config.yaml`` may override these, but the defaults
# keep the module usable without any config file present.
# ---------------------------------------------------------------------------
DEFAULT_RELATIONS: Dict[str, str] = {
    "has_extension": "hasExtension",   # abstract -> extension
    "sub_class_of": "subClassOf",      # flat taxonomy input + abstract-layer mirror
    "instance_of": "instanceOf",       # flat input (instance -> concept)
    "superset_of": "supersetOf",       # extension -> extension (thread-bearing)
    "has_element": "hasElement",       # extension -> instance (thread-bearing)
}


# ===========================================================================
# Configuration
# ===========================================================================
def load_config(path: str | Path = "config.yaml") -> Dict[str, Any]:
    """Load ``config.yaml``. Falls back to sensible defaults if it is missing.

    PyYAML is an optional convenience here; if it (or the file) is unavailable
    we return a minimal default config so the pipeline still runs.
    """
    path = Path(path)
    if not path.exists():
        logger.warning("Config %s not found; using built-in defaults.", path)
        return _default_config()
    try:
        import yaml  # imported lazily so the module loads without PyYAML
    except ImportError:  # pragma: no cover - environment dependent
        logger.warning("PyYAML not installed; using built-in defaults.")
        return _default_config()

    with path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    # Merge missing top-level keys from defaults so callers can rely on them.
    defaults = _default_config()
    for key, value in defaults.items():
        cfg.setdefault(key, value)
    return cfg


def _default_config() -> Dict[str, Any]:
    return {
        "relations": dict(DEFAULT_RELATIONS),
        "data": {"source": "synthetic"},
        "process1": {"strategy": "all", "concepts": []},
        "process2": {
            "repair": {"enabled": True, "min_confidence": 0.75, "max_new_edges": 100}
        },
        "paths": {"raw_dir": "data/raw", "processed_dir": "data/processed"},
    }


def get_relations(config: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
    """Return the relation-name mapping, merged over defaults."""
    rels = dict(DEFAULT_RELATIONS)
    if config:
        rels.update(config.get("relations", {}) or {})
    return rels


# ===========================================================================
# Synthetic data (start small — the default, runs anywhere)
# ===========================================================================
# child concept -> parent concept (i.e. the `subClassOf` direction)
_SYNTHETIC_TAXONOMY: List[Tuple[str, str]] = [
    ("mammal", "animal"),
    ("bird", "animal"),
    ("dog", "mammal"),
    ("cat", "mammal"),
    ("eagle", "bird"),
    ("car", "vehicle"),
    ("truck", "vehicle"),
    ("boat", "vehicle"),
]

# instance -> directly-asserted concept (the `instanceOf` direction)
_SYNTHETIC_INSTANCES: List[Tuple[str, str]] = [
    ("rex", "dog"),
    ("fido", "dog"),
    ("whiskers", "cat"),
    ("freckles", "bird"),     # asserted at a non-leaf concept on purpose
    ("sammy", "eagle"),
    ("my_tesla", "car"),
    ("big_rig", "truck"),
    ("titanic", "boat"),
]


def build_synthetic_graph(
    taxonomy: Optional[Iterable[Tuple[str, str]]] = None,
    instances: Optional[Iterable[Tuple[str, str]]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> nx.DiGraph:
    """Build a flat graph from a small animal/vehicle taxonomy.

    A *flat* graph uses one node per concept and one node per instance, with
    ``subClassOf`` edges between concepts and ``instanceOf`` edges from
    instances to concepts. This is the "before" state that Process 1 expands.

    Parameters
    ----------
    taxonomy : iterable of (child_concept, parent_concept)
    instances : iterable of (instance, concept)
    config : optional config dict (only used for relation names)
    """
    rels = get_relations(config)
    taxonomy = list(taxonomy if taxonomy is not None else _SYNTHETIC_TAXONOMY)
    instances = list(instances if instances is not None else _SYNTHETIC_INSTANCES)

    g = nx.DiGraph()

    def ensure_concept(name: str) -> None:
        if not g.has_node(name):
            g.add_node(name, kind=KIND_CONCEPT, label=name)

    # subClassOf edges (concept -> concept)
    for child, parent in taxonomy:
        ensure_concept(child)
        ensure_concept(parent)
        g.add_edge(child, parent, relation=rels["sub_class_of"])

    # instanceOf edges (instance -> concept)
    for inst, concept in instances:
        ensure_concept(concept)
        g.add_node(inst, kind=KIND_INSTANCE, label=inst)
        g.add_edge(inst, concept, relation=rels["instance_of"])

    logger.info(
        "Built synthetic flat graph: %d nodes, %d edges (%d concepts, %d instances).",
        g.number_of_nodes(),
        g.number_of_edges(),
        sum(1 for _, d in g.nodes(data=True) if d.get("kind") == KIND_CONCEPT),
        sum(1 for _, d in g.nodes(data=True) if d.get("kind") == KIND_INSTANCE),
    )
    return g


# ===========================================================================
# Optional real data: a small WordNet noun subset (used by the notebook)
# ===========================================================================
def build_wordnet_graph(
    roots: Iterable[str] = ("animal.n.01", "vehicle.n.01"),
    max_depth: int = 3,
    instances_per_leaf: int = 1,
    config: Optional[Dict[str, Any]] = None,
) -> nx.DiGraph:
    """Build a flat graph from a WordNet hyponym subset via NLTK.

    Concepts are synsets (named by their lemma); ``subClassOf`` follows the
    hyponym/hypernym relation. WordNet has no individuals, so we synthesize a
    couple of instances under the leaf synsets to exercise Class Threads.

    Raises ``RuntimeError`` (caught by callers) if NLTK/WordNet is unavailable,
    so the synthetic path can always be used as a fallback.
    """
    try:
        from nltk.corpus import wordnet as wn  # type: ignore
        # Trigger the lazy loader so a missing corpus fails here, not later.
        wn.synsets("animal")
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "WordNet via NLTK is unavailable. Install nltk and run "
            "`python -c \"import nltk; nltk.download('wordnet')\"`, or use the "
            "synthetic data source."
        ) from exc

    rels = get_relations(config)
    g = nx.DiGraph()

    def name_of(synset) -> str:
        return synset.name()  # e.g. "dog.n.01" — globally unique & stable

    def add_concept(synset) -> None:
        n = name_of(synset)
        if not g.has_node(n):
            g.add_node(n, kind=KIND_CONCEPT, label=synset.lemmas()[0].name())

    leaves: List[Any] = []
    for root_name in roots:
        try:
            root = wn.synset(root_name)
        except Exception:
            logger.warning("WordNet synset %s not found; skipping.", root_name)
            continue
        add_concept(root)
        # BFS down the hyponym tree to a bounded depth.
        frontier = [(root, 0)]
        while frontier:
            node, depth = frontier.pop()
            hyponyms = node.hyponyms() if depth < max_depth else []
            if not hyponyms:
                leaves.append(node)
                continue
            for child in hyponyms:
                add_concept(child)
                g.add_edge(name_of(child), name_of(node), relation=rels["sub_class_of"])
                frontier.append((child, depth + 1))

    # Synthesize instances under the discovered leaves.
    for leaf in leaves:
        base = leaf.lemmas()[0].name()
        for i in range(instances_per_leaf):
            inst = f"{base}_{i+1}"
            g.add_node(inst, kind=KIND_INSTANCE, label=inst)
            g.add_edge(inst, name_of(leaf), relation=rels["instance_of"])

    logger.info(
        "Built WordNet flat graph: %d nodes, %d edges.",
        g.number_of_nodes(),
        g.number_of_edges(),
    )
    return g


def load_graph_from_config(config: Dict[str, Any]) -> nx.DiGraph:
    """Modular entry point: build the flat graph chosen in ``config['data']``.

    Falls back to synthetic data if WordNet is requested but unavailable.
    """
    source = (config.get("data", {}) or {}).get("source", "synthetic")
    if source == "wordnet":
        wn_cfg = (config.get("data", {}) or {}).get("wordnet", {}) or {}
        try:
            return build_wordnet_graph(
                roots=wn_cfg.get("roots", ("animal.n.01", "vehicle.n.01")),
                max_depth=wn_cfg.get("max_depth", 3),
                config=config,
            )
        except RuntimeError as exc:
            logger.warning("%s Falling back to synthetic data.", exc)
            return build_synthetic_graph(config=config)
    return build_synthetic_graph(config=config)


# ===========================================================================
# Persistence (dispatch on file suffix)
# ===========================================================================
def save_graph(graph: nx.DiGraph, path: str | Path) -> Path:
    """Save a graph. Format chosen by suffix: .json | .graphml | .gpickle/.pkl."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".json":
        data = nx.node_link_data(graph, edges="edges")
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    elif suffix == ".graphml":
        nx.write_graphml(graph, path)
    elif suffix in (".gpickle", ".pkl"):
        with path.open("wb") as fh:
            pickle.dump(graph, fh)
    else:
        raise ValueError(f"Unsupported graph format: {suffix!r}")
    logger.info("Saved graph -> %s", path)
    return path


def load_graph(path: str | Path) -> nx.DiGraph:
    """Load a graph saved by :func:`save_graph`."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        return nx.node_link_graph(data, directed=True, edges="edges")
    if suffix == ".graphml":
        return nx.read_graphml(path)
    if suffix in (".gpickle", ".pkl"):
        with path.open("rb") as fh:
            return pickle.load(fh)
    raise ValueError(f"Unsupported graph format: {suffix!r}")


# ===========================================================================
# Inspection / visualization
# ===========================================================================
def describe_graph(graph: nx.DiGraph) -> Dict[str, Any]:
    """Return a small summary dict (node-kind counts, relation counts)."""
    kinds: Dict[str, int] = {}
    for _, d in graph.nodes(data=True):
        kinds[d.get("kind", "?")] = kinds.get(d.get("kind", "?"), 0) + 1
    relations: Dict[str, int] = {}
    for _, _, d in graph.edges(data=True):
        relations[d.get("relation", "?")] = relations.get(d.get("relation", "?"), 0) + 1
    return {
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "node_kinds": kinds,
        "relations": relations,
    }


# Colors per node kind, used by both matplotlib and pyvis renderers.
_KIND_COLORS: Dict[str, str] = {
    KIND_CONCEPT: "#8ecae6",    # light blue
    KIND_INSTANCE: "#ffb703",   # amber
    KIND_ABSTRACT: "#219ebc",   # teal
    KIND_EXTENSION: "#90be6d",  # green
}


def visualize(
    graph: nx.DiGraph,
    title: str = "Class Thread graph",
    ax=None,
    seed: int = 42,
    show: bool = False,
):
    """Draw the graph with matplotlib, coloring nodes by kind and labeling edges.

    Returns the matplotlib ``Axes``. Importing matplotlib is deferred so the
    core pipeline does not require it.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(11, 8))

    pos = nx.spring_layout(graph, seed=seed, k=0.9)
    node_colors = [
        _KIND_COLORS.get(d.get("kind"), "#cccccc") for _, d in graph.nodes(data=True)
    ]
    nx.draw_networkx_nodes(graph, pos, node_color=node_colors, node_size=1300, ax=ax)
    nx.draw_networkx_labels(graph, pos, font_size=8, ax=ax)
    nx.draw_networkx_edges(
        graph, pos, ax=ax, arrows=True, arrowsize=15, edge_color="#666666",
        connectionstyle="arc3,rad=0.05",
    )
    edge_labels = {(u, v): d.get("relation", "") for u, v, d in graph.edges(data=True)}
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=7, ax=ax)

    # Legend for node kinds actually present.
    present = {d.get("kind") for _, d in graph.nodes(data=True)}
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=_KIND_COLORS[k],
                   markersize=10, label=k)
        for k in _KIND_COLORS if k in present
    ]
    if handles:
        ax.legend(handles=handles, loc="best", fontsize=8)
    ax.set_title(title)
    ax.axis("off")
    if show:
        plt.show()
    return ax


def visualize_pyvis(
    graph: nx.DiGraph,
    path: str | Path = "data/processed/graph.html",
    notebook: bool = False,
) -> Path:
    """Render an interactive HTML view with pyvis. Returns the output path."""
    from pyvis.network import Network

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    net = Network(height="650px", width="100%", directed=True, notebook=notebook,
                  cdn_resources="in_line")
    for n, d in graph.nodes(data=True):
        net.add_node(
            n, label=str(d.get("label", n)),
            color=_KIND_COLORS.get(d.get("kind"), "#cccccc"),
            title=f"{n} ({d.get('kind', '?')})",
        )
    for u, v, d in graph.edges(data=True):
        net.add_edge(u, v, label=d.get("relation", ""), title=d.get("relation", ""))
    net.write_html(str(path), notebook=notebook, open_browser=False)
    logger.info("Wrote interactive graph -> %s", path)
    return path
