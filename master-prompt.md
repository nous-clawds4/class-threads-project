# Class Thread Model — Master Prompt for Claude Code

You are an expert Python engineer and knowledge-representation researcher. Your task is to build a clean, well-documented, minimal but complete starter project that demonstrates the **Class Thread model**.

## Core Concept (do not deviate)
Every concept is represented by **two nodes**:
- An **abstract node** (the idea / intension of the concept)
- An **extensional superset node** (the set of all instances of that concept)

These two nodes are connected by a new typed relation called **`hasExtension`**.

A **Class Thread** is a directed path that must follow this exact pattern:
`hasExtension` → (zero or more `subClassOf`) → `instanceOf`

The model must enforce that every instance is connected to its relevant concepts via at least one valid Class Thread.

## Project Goals
- Create a working prototype that shows the value of the dual-node + `hasExtension` + Class Thread structure.
- Start simple (NetworkX) so everything runs on a laptop.
- Make it easy to later migrate hot paths to Neo4j.
- Include clear before/after comparison (flat graph vs. threaded dual-node graph).
- Produce measurable metrics and a runnable notebook.

## Technology Constraints (follow strictly)
- Primary implementation: **NetworkX** (in-memory, easy to visualize and debug).
- Optional later module: Neo4j + Cypher helpers (do **not** make Neo4j mandatory in the first version).
- Language: Python 3.10+
- Use `nltk` for WordNet access (or generate synthetic data if WordNet is not available).
- Keep the code clean, modular, well-commented, and type-hinted where helpful.
- Use a professional project layout.

## Required Project Structure

```
class-thread/ ├── data/ │   ├── raw/ │   └── processed/ ├── src/ │   ├── init.py │   ├── graph_utils.py          # load, save, visualize helpers │   ├── process1_dual_node.py   # expansion logic │   ├── process2_thread_enforce.py  # validation + optional repair │   ├── metrics.py │   └── neo4j_helpers.py        # optional, stubbed for now ├── notebooks/ │   └── class_thread_demo.ipynb ├── tests/ │   └── test_thread_logic.py ├── config.yaml ├── requirements.txt ├── README.md └── .gitignore
```

## Detailed Requirements

### Process 1 — Dual-Node Expansion
- Accept a flat graph (single nodes for concepts + instances, with `subClassOf` and `instanceOf` edges).
- For selected concepts, create:
  - `abstract:{concept_name}`
  - `extension:{concept_name}`
- Add `hasExtension` edge from abstract → extension node.
- Rewire existing edges logically:
  - `subClassOf` relationships should primarily live between abstract nodes.
  - `instanceOf` relationships should point to extension nodes.
- Make the selection of which concepts get dual representation configurable (via a list or heuristic).

### Process 2 — Class Thread Enforcement
- Define a valid Class Thread as a path matching:  
  `hasExtension` → (0 or more `subClassOf`) → `instanceOf`
- For every instance node, check whether it has at least one valid thread to each of its expected concepts.
- Implement both:
  - A validator that reports coverage and broken threads.
  - An optional repair function that can add missing `hasExtension` or `subClassOf` edges (with clear logging).
- Make the repair logic conservative and configurable (e.g., only repair when confidence is high).

### Metrics & Evaluation
Implement at minimum:
- Thread coverage (% of instance–concept pairs that have ≥1 valid thread)
- Number of broken threads before and after enforcement
- Average number of valid threads per instance
- Simple query examples that demonstrate the difference between flat and threaded graphs

### Data Strategy (start small)
- First version: Use a small **synthetic animal/vehicle taxonomy** (easy to control and debug).
- Second version / notebook: Load a real WordNet noun subset using NLTK.
- Make data loading modular so either source can be used.

### Notebook (`class_thread_demo.ipynb`)
The notebook should:
- Load or generate a small graph
- Show the flat version
- Apply Process 1 (dual-node expansion)
- Apply Process 2 (thread enforcement)
- Print clear metrics and show example queries before vs after
- Include visualizations (pyvis or networkx + matplotlib)

### Code Quality
- Clear separation of concerns
- Good docstrings and comments explaining the Class Thread logic
- Type hints where they improve readability
- Error handling and logging
- Easy to extend later with Neo4j

## Development Workflow (follow this order)
1. Create the full project skeleton and `requirements.txt`.
2. Implement data loading (synthetic first).
3. Implement `process1_dual_node.py`.
4. Implement `process2_thread_enforce.py` (validation first, repair second).
5. Implement `metrics.py`.
6. Create the Jupyter notebook that ties everything together with before/after comparison.
7. Write `README.md` with clear instructions on how to run the demo.
8. Add basic tests for the thread validation logic.
9. Make sure everything runs cleanly with `python -m pytest` and the notebook.

## Output Rules
- Always create or edit files in the exact project structure above.
- After creating or modifying code, show the key parts of the file (or the diff) so the user can review.
- When something is ready to test, provide the exact command to run it.
- If you encounter ambiguity, ask for clarification before proceeding.
- Prioritize correctness and clarity over premature optimization.

You are now in "build mode". Begin by creating the project skeleton and `requirements.txt`, then proceed through the workflow above one step at a time. After each major step, summarize what was done and ask for confirmation before continuing to the next step.
