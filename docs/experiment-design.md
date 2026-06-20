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
| Single-node ablation | `src/experiment/ablation.py` | ✅ |
| Adversarial Tier-B distractors | `src/experiment/corruption.py` | ✅ (rewiring) |
| Held-out build-from-complement pipeline | `src/experiment/oracle.py` | ⬜ todo |
| SHACL/SPARQL baselines | `src/experiment/baselines.py` | ⬜ todo |
| Wikidata ingestion | `src/experiment/wikidata.py` | ✅ |
| Experiment runner (factor grid → parquet) | `src/experiment/experiment.py` | ✅ |
| Money runner + adversarial Tier-B rewiring | `src/experiment/money.py`, `corruption.py` | ✅ |
| Pluggable per-proposal confidence (EdgeProposal) | `src/process2_thread_enforce.py` | ✅ |
| Corroboration confidence + per-type baseline | `src/experiment/confidence.py` | ✅ |
| Synthetic DAG w/ tunable redundancy | `src/graph_utils.py` (`build_synthetic_dag`) | ✅ |
| Stats (bootstrap CIs, Wilcoxon) | `src/experiment/stats.py` | ⬜ todo |

## 11. Findings so far (2026-06-20)

**Tier A — degradation & recovery (Fig 1, Fig 5).** Repair fully recovers
structurally-implied breaks (E-SO, E-HX) at precision 1.0. Conservatism is
visible: membership loss (E-HE) is *not* auto-repaired at θ=0.75. The dual-node
advantage is **real but narrow** — single-node is blind only to `hasExtension`-
layer breaks (E-HX); for E-SO/E-HE the split buys nothing on detection (the
pre-committed null result).

**Money figure — the pivotal finding (Fig 2, Fig 3).** With repair run
*autonomously* (no oracle handed in) and scored against the pristine graph,
**repair fidelity is a faithful function of guidance integrity**: clean taxonomy
→ precision 1.0; rewiring the taxonomy drives precision down (~1.0 → ~0.3–0.6 at
10–40% rewire) and hallucination up. **Critically, the fixed per-edge-*type*
confidences (1.0/0.9/0.55) make θ an on/off switch, not a graded precision–recall
knob** — hallucination is flat in θ until repair shuts off entirely (θ>0.9).
Implication: conservatism (high θ) can *refuse* repair wholesale when guidance is
suspect, but cannot *selectively filter* good proposals from bad. A genuine
precision–recall knob needs an **evidence-based / corroboration-based per-proposal
confidence**, which requires structural redundancy that near-tree WordNet lacks —
motivating the **Wikidata DAG** and connecting to KG-completion confidence
(AMIE / embeddings). This reframes the contribution toward *when structural
repair is trustworthy* and *what signal makes conservatism selective*.

**Resolution — corroboration confidence makes θ a graded knob, bounded by graph
redundancy (Fig 4, Fig 6; 2026-06-20, adversarially verified).** We replaced the
fixed per-edge-*type* confidence with a pluggable per-*proposal* scorer
(`repair_threads(..., confidence=…)`; `src/experiment/confidence.py`). The
`corroboration_confidence` scorer ranks each proposed `supersetOf(ext:P→ext:C)`
edge (asserting `C ⊑ P`) by how many **independent surviving routes** entail it —
alternative `subClassOf` paths plus the independently-corrupted `supersetOf`
extension layer (a saturating count `k`, `r = 1 − 1/(1+k)`, blended with the
per-type prior via `floor`; `floor = 1` recovers the baseline exactly). It reads
**only the damaged graph's surviving structure** — no `pristine_edges`, `removed`,
`distractor` flag, or `expected` (leakage audit: clean). A lone rewired edge to a
random wrong parent has no alternative route → ≈0 corroboration → low confidence.

What this buys, stated precisely:

- **Frontier extension, *not* a higher-precision win at matched recall.** The two
  scorers coincide at exactly one operating point (θ=0, where nothing is filtered):
  identical edge set, identical precision. The per-type scorer can express *only*
  that point (every `supersetOf` shares confidence 0.9, so θ is on/off and cliffs
  to recall 0 at θ>0.9). Corroboration **reproduces** that point and then traces a
  graded Pareto frontier of higher-precision/lower-recall operating points the
  baseline structurally cannot reach — on the dense synthetic DAG, precision
  0.30→0.63→0.97→1.00 as recall falls 0.62→0.16→0.05→0.01; on Wikidata
  0.27→1.00 as recall falls 0.23→0.01. Do **not** describe this as "beats the
  baseline at matched recall" (false); it is a frontier the baseline lacks.
- **The advantage is a measurable, ~linear function of graph redundancy.** Average
  precision (interpolated; area under the best-precision-at-recall≥r frontier)
  advantage over per-type is **+0.000 on three independent tree datasets**
  (synthetic, WordNet, `sdag mi=0`), rising monotonically to **+0.07** at
  redundancy 0.23; the real `wikidata:Q42889:3` slice (redundancy 0.036) sits on
  that trend at **+0.006**. Redundancy is a property of the *data*, so the knob's
  power is too — demonstrated by sweeping it. (An independent re-derivation
  confirmed the monotone ordering and the no-leakage / not-rigged conclusions.)

Honest caveats — this is a precision–**recall trade**, not a free lunch:

1. **No benefit on a tree, and a *cost* at a fixed threshold.** With no redundancy,
   every `supersetOf` proposal scores corroboration 0, so any θ>0 gates **all** of
   them out — recall collapses to 0 (a cliff, not a graded curve). The
   achievable-frontier AP advantage is 0 (corroboration can still match the
   baseline at θ=0), but at a *fixed* operating θ>0 corroboration on a non-redundant
   graph refuses all repair and is strictly worse than per-type. The knob is only
   useful where multiple inheritance exists.
2. **Precision is bought with recall everywhere off the redundant subgraph.**
   Corroboration achieves ~0 recall on real tree-region edges (it cannot tell a
   genuinely-missing tree edge from a fabrication — neither has an alternative
   route). The headline operating points sit at low recall and small `n_added`
   (e.g. ~2 edges on Wikidata) — high variance; report counts/CIs, not bare
   precision.
3. **A soft prior, not a hard filter.** Rewiring-fabricated edges still receive
   corroboration > 0 at a measurable rate (≈8.7% on the dense DAG; max fabrication
   score 0.675, above some operating θ). The gate **reduces but does not eliminate**
   hallucination.
4. **The adversary is unconstrained and makes the damaged taxonomy cyclic.**
   `guidance_rewire_rate` rewires to *any* abstract node regardless of layer,
   injecting cross-layer/downward `subClassOf` edges and cycles (≈9/10 trials);
   `nx.descendants` on the cyclic damaged graph can then find spurious routes that
   "corroborate" a genuinely-false edge. This is the honest worst case (adversary-
   induced, not a bug); it is rarer on a curated-acyclic ontology. Disclose that
   corroboration trusts reachability in a possibly-cyclic damaged graph.
5. **The exact-typed-edge precision oracle overstates semantic error.**
   `edge_precision`/`hallucination_rate` use exact `(u,v,relation)` identity vs
   `pristine_edges`; semantically-*true* transitive `supersetOf` shortcuts (where
   `C ⊑ P` holds in the pristine closure but the exact edge was never present) are
   counted as hallucinations (on the tree, 14/14 corroborated "fabrications" were
   true shortcuts). A closure-aware precision is *todo* and would only **improve**
   corroboration's apparent precision.
6. **Not rigged (reported as evidence).** Rewiring hits redundant vs tree edges at
   the dataset base rate (sdag 0.326 base vs 0.317 rewired; Wikidata 0.036 vs
   0.038), so "corroborated" ≠ "un-rewired by construction"; the per-type baseline
   is verifiably flat in θ.
7. **Per-layer attribution.** On the dense DAG most fabrication-corroboration came
   via the `subClassOf` leg, not the cleaner `supersetOf` "independent witness"
   (≈248 vs 23). That witness is genuinely independent **only because the E-SO arm
   removes — never rewires — `supersetOf`**; the framing must be re-checked if a
   future arm corrupts the extension layer.

**Net reframing.** Per-type conservatism can only *refuse* repair wholesale;
corroboration makes it *selective* — but only to the degree the graph is
redundant, a measurable property of the data, not of the method. The contribution
is thus *when structural repair is trustworthy and what makes conservatism
selective*, with the redundancy-scaling law (and its tree null) as the evidence.
