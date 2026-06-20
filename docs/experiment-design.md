# Integrity & Repair — Experiment Design

Empirical spine for the Class Thread preprint. This is the canonical protocol;
the harness in `src/experiment/` is built against it.

## 1. Objective & honest contribution

We do **not** claim the intension/extension distinction (subsumed by OWL
punning, RDFS(FA), DL metamodeling) or the validator/query (a SHACL path
constraint / regular path query) as novel. The defensible contribution is a
**methodological package**:

1. **Materialized membership as a traversable typed path** — `hasExtension →
   supersetOf* → hasElement` — so a break *localizes* to a specific missing edge
   of a known type, not merely "unreachable."
2. **A controlled-corruption evaluation whose repair oracle is independent of
   repair's own guidance signal.**
3. **Conservative, confidence-gated, addition-only repair toward a known-complete
   target** ("graph healing," not knowledge discovery).

**Framing is data-driven:** run the full dual-vs-single ablation, then make the
stronger dual-node claim *only if the data supports it*, else fall back to the
package claim.

## 2. Datasets

| Dataset | Role | Notes |
|---|---|---|
| Synthetic animal/vehicle | control | 10 concepts / 8 instances, fully understood |
| WordNet noun subsets | scale | `vehicle.n.01` d2 (96 nodes) → `animal+vehicle` d3 (784 flat / 1213 dual) |
| **Wikidata/DBpedia slice** | credibility | real individuals, multiple inheritance, real incompleteness (a DAG, not a tree) |

Report per dataset: node/edge counts, branching factor, depth, multiple-inheritance
rate, instances-per-leaf distribution — so reviewers know threads were tested on a DAG.

## 3. The circularity problem and its resolution (the crux)

Freezing the pristine graph as ground truth is **necessary but not sufficient**.
If only thread edges are corrupted and the `subClassOf` taxonomy stays intact,
repair re-derives exactly the missing edges and edge precision is **1.0 by
construction** — a tautology a reviewer will reject. Repair must be *able to be
wrong*. Two tiers:

- **Tier A — thread-only corruption** (`E-HX`/`E-SO`/`E-HE`/`E-MIX`/`N-REM`).
  Guidance intact. Measures graceful **degradation**, **detection**, and
  **recovery**. Edge precision here is high by construction and is **not**
  headlined.
- **Tier B — guidance corruption** (Tier A + `subClassOf` removal + **distractor**
  `subClassOf` injection). Repair can now propose fabricated edges, so
  **edge precision / hallucination become real measurements**.

**Held-out oracle (independent recall):** withhold a fraction `h≈0.15` of gold
memberships *before* building the graph; build the taxonomy from the complement;
measure recovery of held-out pairs repair never saw.

> ⚠️ **Smoke finding (2026-06-20):** on clean *tree-shaped* data, conservative
> repair re-derives the real `shortest_path` and **ignores random distractors** —
> Tier B precision stayed 1.0. Inducing measurable hallucination requires
> **adversarial** distractors that forge a *shorter/only* false entailment path
> between a broken pair's leaf and its concept, the held-out oracle, and/or the
> real multiple-inheritance noise of the **Wikidata** dataset. Random distractor
> injection alone is insufficient. (This is a key reason the Wikidata slice is in
> scope.)

## 4. Experimental factors

| Factor | Levels |
|---|---|
| Dataset | synthetic · WordNet (sizes) · Wikidata slice |
| Node model (headline ablation) | dual-node (thread validator) · single-node (`flat_instances_transitive` closure) |
| Corruption arm | E-HX · E-SO · E-HE · E-MIX · N-REM (+ Tier-B guidance) |
| Tier | A (guidance intact) · B (guidance corrupted: removal + distractors @ 5/10/20%) |
| Selection policy | UNIF (random, ≥30 seeds) · TARG (betweenness spine, deterministic) |
| Corruption rate ρ | 0, .05, .10, .20, .30, .50, .70, .90 |
| Repair | off (degradation) · on (recovery) |
| Confidence threshold θ | 0, .50, .55, .60, **.75**, .85, .90, .95, 1.0, 1.01 (brackets the .55/.90/1.0 gates) |
| Held-out fraction h | 0 · 0.15 |
| Baseline representation | dual-node thread · flat+SPARQL property path · flat+SHACL · flat+transitive closure |

**Pre-registered primary endpoint:** held-out edge recall, dual vs single, at
ρ=0.20, θ=0.75, Tier-B noise=10%. Everything else is exploratory; Holm-Bonferroni
within the declared family.

## 5. Metrics (each tied to an independent oracle)

- **thread_coverage** — frozen `EXPECTED`, never re-derived on a damaged graph.
- **edge_precision / hallucination_rate** — added edges ∩ `PRISTINE_EDGES`
  (observed pristine artifact, not the guiding taxonomy). Headlined under Tier B.
- **edge_recall** — over *recoverable* removed edges (pristine, endpoints survive,
  relation is repair-targetable; excludes `subClassOf` guidance and the N-REM ceiling).
- **recovery_fraction** — `(cov_repair − cov_corrupt)/(1 − cov_corrupt)`.
- **pair_false_positives** — instances threading to a concept ∉ `EXPECTED` (false
  membership; must be empty in the conservative regime).
- **held_out_recovery** — non-circular recall on withheld memberships.
- **detection P/R/F1**, **localization_edit_distance**, **repair_ceiling_pairs** — *(todo)*.

## 6. Ablations
Dual vs single (mandatory, headline; report null honestly) · Tier-A vs Tier-B ·
frozen vs re-derived oracle (justifies the freeze) · confidence-constant sweep ·
`max_new_edges` cap (isolated) · N-REM ceiling.

## 7. Figures (results section)
1. Coverage degradation & recovery vs ρ (dual/single/no-repair, CI bands).
2. **Precision–recall vs θ** under Tier B (edge + pair panels) — the money figure.
3. Hallucination vs θ, faceted by Tier-B noise.
4. Held-out recovery vs θ, faceted by ρ.
5. Detection confusion: dual (exact) vs single (FN>0 from silent rerouting).
6. Baseline head-to-head: detection F1 + localization across representations.

## 8. Baselines (answer the "already solved" objection)
`flat+SHACL` (pyshacl) and `flat+SPARQL property-path` (rdflib) on the same
corruption suite. They *validate* but do not *localize+repair against an
independent gold standard* — that gap is the claim. *(todo: `src/experiment/baselines.py`)*

## 9. Statistics
≥30 seeds/randomized cell (100 for primary + Tier-B precision cells); bootstrap
95% CI (10k); paired Wilcoxon for dual-vs-single (by seed & removal set);
Holm-Bonferroni. Never SEM bars near 0/1. Determinism: same seed → identical
removals and identical repair edges (asserted).

## 10. Implementation status

| Component | File | Status |
|---|---|---|
| Deterministic seeding | `src/experiment/util.py` | ✅ |
| Oracle freeze + held-out split | `src/experiment/oracle.py` | ✅ |
| Corruption (Tier A + Tier B/distractors, UNIF/TARG) | `src/experiment/corruption.py` | ✅ |
| Edge/pair metrics | `src/experiment/metrics_ext.py` | ✅ (detection/localization/held-out-recovery todo) |
| Repair determinism fix | `src/process2_thread_enforce.py` | ✅ |
| Core tests | `tests/test_experiment_core.py` | ✅ (12 tests) |
| Single-node ablation | `src/experiment/ablation.py` | ⬜ todo |
| Adversarial Tier-B distractors | `src/experiment/corruption.py` | ⬜ todo (smoke finding §3) |
| Held-out build-from-complement pipeline | `src/experiment/oracle.py` | ⬜ todo |
| SHACL/SPARQL baselines | `src/experiment/baselines.py` | ⬜ todo |
| Wikidata ingestion | `src/graph_utils.py` / new loader | ⬜ todo |
| Experiment runner (factor grid → parquet) | `src/experiment/experiment.py` | ⬜ todo |
| Stats + figures | `src/experiment/stats.py`, `figures.py` | ⬜ todo |
