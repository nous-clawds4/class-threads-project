# Class Thread Model

A clean, minimal-but-complete prototype of the **Class Thread** dual-node
knowledge-representation model, built on [NetworkX](https://networkx.org/).

## The idea

Every concept is represented by **two** nodes instead of one:

| Node | Meaning |
|------|---------|
| `abstract:C` | the **intension** — the *idea* of concept `C` |
| `extension:C` | the **extension** — the *set of all instances* of `C` |

They are joined by a new typed relation, **`hasExtension`** (`abstract:C → extension:C`).

A **Class Thread** is a directed path that must follow this exact pattern:

```
hasExtension  →  (zero or more) supersetOf  →  hasElement
```

…connecting a concept's abstract node down through the taxonomy to a concrete
instance. For example, *"rex is an animal"* is the single forward path:

```
abstract:animal ─hasExtension→ extension:animal ─supersetOf→ extension:mammal
                ─supersetOf→ extension:dog ─hasElement→ rex
```

The model **enforces** that every instance is connected to each of its expected
concepts by at least one valid Class Thread.

### Why two nodes?

In a *flat* graph, "give me all animals" has no direct answer — no instance is
asserted directly as `animal`, so a naive 1-hop query returns nothing, and a
correct answer needs a bespoke recursive walk mixing `subClassOf` + `instanceOf`
in just the right directions. In the *threaded* graph the same question is **one
uniform path pattern**, and it returns the complete set. See section
[Flat vs. threaded](#flat-vs-threaded-queries) below.

## Project layout

```
.
├── config.yaml                     # relation names, data source, Process 1/2 knobs
├── requirements.txt
├── data/
│   ├── raw/                        # (inputs)
│   └── processed/                  # generated graphs / visualizations (gitignored)
├── src/
│   ├── graph_utils.py              # config, data loading, save/load, visualization
│   ├── process1_dual_node.py       # Process 1: dual-node expansion + edge rewiring
│   ├── process2_thread_enforce.py  # Process 2: thread validation + conservative repair
│   ├── metrics.py                  # coverage, before/after, flat-vs-threaded queries
│   └── neo4j_helpers.py            # optional, stubbed Cypher mapping for later migration
├── notebooks/
│   └── class_thread_demo.ipynb     # end-to-end walkthrough with visualizations
└── tests/
    └── test_thread_logic.py        # thread validation + repair tests
```

## Relation vocabulary

| Relation | Direction | Role |
|----------|-----------|------|
| `hasExtension` | `abstract:C → extension:C` | links the two halves of a concept |
| `supersetOf` | `extension:parent → extension:child` | extension hierarchy, **general → specific** (thread-bearing) |
| `hasElement` | `extension:C → instance` | membership (thread-bearing) |
| `subClassOf` | `abstract:child → abstract:parent` | conventional taxonomy mirror, for fidelity/visualization |

> The **flat input** graph uses the classic `subClassOf` (child → parent) and
> `instanceOf` (instance → concept) edges. Process 1 reads those and produces the
> renamed, thread-bearing `supersetOf` / `hasElement` edges of the dual-node model.

All names are centralized in `config.yaml` and can be overridden there.

## Setup

Requires **Python 3.10+**.

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The core pipeline only needs `networkx` + `pyyaml`. `matplotlib` / `pyvis` are
for visualization, `nltk` for the optional WordNet data path, and `pytest` for
tests.

## Quick start

Run the whole pipeline from a Python shell (or see the notebook):

```python
from src import graph_utils as gu
from src import process1_dual_node as p1
from src import process2_thread_enforce as p2
from src import metrics as m

cfg  = gu.load_config("config.yaml")
flat = gu.load_graph_from_config(cfg)        # synthetic animal/vehicle taxonomy
dual = p1.expand_to_dual_nodes(flat, cfg)    # Process 1: dual-node expansion

report = p2.validate_threads(dual, config=cfg)
print(m.format_metrics(m.compute_metrics(dual, config=cfg, report=report)))

# A single concrete thread:
print(p2.find_thread(dual, "animal", "rex", cfg))
# -> ['abstract:animal', 'extension:animal', 'extension:mammal', 'extension:dog', 'rex']
```

### Validate, break, and repair

```python
EXPECTED = set(p2.validate_threads(dual, config=cfg).expected)   # snapshot ground truth

dual.remove_edge("extension:animal", "extension:mammal")         # introduce a gap
before = p2.validate_threads(dual, config=cfg, expected=EXPECTED)

p2.repair_threads(dual, config=cfg, report=before)               # conservative repair
after  = p2.validate_threads(dual, config=cfg, expected=EXPECTED)

print(m.format_delta(m.enforcement_delta(before, after)))
# coverage 85.0% -> 100.0% ; broken 3 -> 0   (rex~animal, fido~animal, whiskers~animal)
```

Repair is **conservative and confidence-gated**. It only proposes the minimal
canonical edges to restore a thread and applies those at/above
`config.process2.repair.min_confidence` (default `0.75`):

| Repair | Confidence | Applied by default? |
|--------|-----------|---------------------|
| add missing `hasExtension` (structurally implied) | 1.0 | ✅ |
| add missing `supersetOf` (attested by the taxonomy) | 0.9 | ✅ |
| add `hasElement` (asserts a **new** membership) | 0.55 | ⛔ skipped — lower the threshold to allow |

### Flat vs. threaded queries

```python
print(m.query_comparison(flat, dual, "animal", cfg))
# flat_direct      : []                                   <- naive 1-hop misses everything
# flat_transitive  : ['fido','freckles','rex','sammy','whiskers']   <- correct, but bespoke
# threaded         : ['fido','freckles','rex','sammy','whiskers']   <- one uniform pattern
```

## Run the notebook

```bash
jupyter notebook notebooks/class_thread_demo.ipynb
```

It loads the graph, applies Process 1 and Process 2, prints metrics, runs the
flat-vs-threaded query comparison, and renders before/after visualizations
(matplotlib + an interactive pyvis HTML in `data/processed/`).

## Run the tests

```bash
python -m pytest
```

## Configuration (`config.yaml`)

- `data.source` — `synthetic` (default; runs anywhere) or `wordnet` (needs
  `nltk` + the `wordnet` corpus; falls back to synthetic if unavailable).
- `process1.strategy` — `all` (expand every concept) or `list` (only the named
  concepts under `process1.concepts`).
- `process2.repair` — `enabled`, `min_confidence`, `max_new_edges`.

### Optional: WordNet data

```bash
python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"
```

Then set `data.source: wordnet` in `config.yaml`.

## Migrating to Neo4j (later)

`src/neo4j_helpers.py` is intentionally a stub in this version — Neo4j is **not**
a dependency. It sketches the Cypher equivalent of a Class Thread so hot paths
can later move to a graph database without rethinking the model:

```cypher
MATCH (a:Abstract {name: $concept})-[:hasExtension]->(:Extension)
      -[:supersetOf*0..]->(spec:Extension)-[:hasElement]->(i:Instance)
RETURN a.name AS concept, i.name AS instance
```

## License

See [`LICENSE`](LICENSE).
