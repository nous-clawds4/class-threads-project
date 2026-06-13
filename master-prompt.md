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
