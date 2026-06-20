"""Experiment harness for the Class Thread *integrity & repair* study.

Modules:
  * ``util``        — deterministic RNG seeding for reproducible trials
  * ``oracle``      — freeze ground truth (expected pairs + pristine edges)
  * ``corruption``  — Tier A / Tier B controlled damage to a dual-node graph
  * ``metrics_ext`` — edge- and pair-level metrics against the frozen oracle

See ``docs/experiment-design.md`` for the full protocol.
"""
