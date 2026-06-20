# Handoff — Class Thread Integrity & Repair Preprint

Resume document for the next working session. Pair this with
[`docs/experiment-design.md`](experiment-design.md) (full protocol + findings)
and [`master-prompt.md`](../master-prompt.md) (project origin).

## TL;DR — current state (2026-06-20)

- **Goal:** an arXiv preprint demonstrating the Class Thread / dual-node model
  via an **integrity & repair** empirical study. Framing is **data-driven**:
  make the stronger dual-node claim only if the data supports it.
- **Branch:** `feat/integrity-repair-experiment` (pushed to `origin`), ~7 commits
  ahead of `main`. **68 tests pass** (`.venv/bin/python -m pytest -q`).
  Note: `python` is not on PATH; use `.venv/bin/python`.
- **Datasets wired in:** `synthetic` (control), `wordnet:vehicle.n.01:2` (scale),
  `wikidata:Q42889:3` (credibility — real DAG, 162 concepts / 120 instances / 6
  multiple-inheritance concepts, cached at `data/raw/wikidata_Q42889_d3.json`),
  and `sdag:<levels>:<branching>:<mi_rate>:<seed>` (synthetic DAG with a
  **tunable redundancy rate** — the controlled independent variable; `mi_rate=0`
  is a tree).
- **Figures:** `results/figures/` — fig1 (coverage degradation/recovery), fig2
  (fidelity vs guidance corruption, per-type), fig3 (per-type θ is on/off), fig4
  (**money: graded PR frontier, corroboration vs per-type**), fig5 (detection
  confusion), fig6 (**precision advantage scales with graph redundancy**).

## How to run

```bash
.venv/bin/python -m pytest -q                 # 68 tests  (python is NOT on PATH)
.venv/bin/python -m src.experiment.experiment # Tier-A grid -> results/results.parquet
.venv/bin/python -m src.experiment.money      # money grid  -> results/money.parquet
.venv/bin/python -m src.experiment.figures    # fig1, fig5
# money figures (fig2, fig3, fig4, fig6) are written by the money run itself
```
Wikidata loads from the committed cache (offline). To refresh:
`build_wikidata_graph(..., refresh=True)` (one WDQS fetch; be polite).

## What's built (`src/experiment/`)

| Module | Role |
|---|---|
| `util.py` | deterministic blake2b-seeded RNG (`make_rng`) |
| `oracle.py` | `freeze_oracle` (frozen EXPECTED + PRISTINE_EDGES), `holdout_split` |
| `corruption.py` | Tier-A arms (E-HX/E-SO/E-HE/E-MIX/N-REM, UNIF/TARG) + Tier-B (`guidance_rewire_rate`, distractors) |
| `metrics_ext.py` | edge precision/recall, hallucination, coverage, recovery, pair false-positives, closure-aware (semantic) precision |
| `confidence.py` | per-proposal repair confidence: `per_type_confidence` (baseline) + `corroboration_confidence` |
| `ablation.py` | single-node comparator (homologous flat corruption + closure detection) |
| `experiment.py` | Tier-A factor-grid runner + `build_dataset` dispatch |
| `money.py` | autonomous-repair runner (rewire × θ) + PR-frontier / redundancy figures |
| `stats.py` | bootstrap CIs, paired Wilcoxon, Holm–Bonferroni; money AP-advantage summary |
| `wikidata.py` | bounded P279/P31 slice loader, cleaned + cached |

Repair lives in `src/process2_thread_enforce.py` (`repair_threads`, deterministic
tie-break). Confidence constants: `CONF_HAS_EXTENSION=1.0`, `CONF_SUPERSET=0.9`,
`CONF_HAS_ELEMENT=0.55`.

## Findings so far

1. **Repair recovers structurally-implied breaks** (E-SO, E-HX) at precision 1.0;
   conservatism is visible (membership loss not auto-repaired at θ=0.75).
2. **Dual-node advantage is real but NARROW** — single-node is blind only to
   `hasExtension`-layer breaks (E-HX); for the other arms the split buys nothing
   (the pre-committed null). Holds on all three datasets.
3. **Repair fidelity tracks guidance integrity**, and — pivotally — **the fixed
   per-edge-TYPE confidences make θ an on/off switch, not a graded knob.** On the
   Wikidata DAG, redundancy buffers edge loss (lower edge-recall at clean
   guidance), exposing the corroborating-path signal the next task exploits.
4. **DONE (this session): evidence-based per-proposal confidence makes θ a graded
   knob, bounded by graph redundancy.** A `corroboration_confidence` scorer that
   ranks each proposal by independent surviving routes turns repair's single
   uncontrolled operating point into a graded Pareto PR frontier — *frontier
   extension*, not a higher-precision win at matched recall (the modes tie at
   θ=0). Its average-precision advantage is **0 on three tree datasets** and rises
   monotonically with redundancy (to +0.076 at redundancy 0.28; Wikidata +0.006 on
   trend). It's a precision–**recall trade**, a soft prior (some fabrications
   survive), and useless/harmful on a tree. Adversarially verified (no leakage,
   not rigged). A closure-aware (semantic) precision confirms the exact oracle only
   *understates* corroboration. Full writeup + 7 caveats in
   `docs/experiment-design.md` §11.

## Primary task — COMPLETE

The evidence-based per-proposal confidence is implemented, tested (9 new tests),
wired into the money grid, and verified. See §11 of the design doc. Key code:
`src/process2_thread_enforce.py` (`EdgeProposal` + pluggable `confidence`),
`src/experiment/confidence.py` (`corroboration_confidence`, `per_type_confidence`),
`src/graph_utils.py` (`build_synthetic_dag`), `src/experiment/money.py`
(`fig4_pr_curve`, `fig6_redundancy_scaling`, `precision_advantage`).

### Recommended next steps (in priority order)
1. ~~**Closure-aware precision** (caveat 5)~~ **DONE this session**:
   `metrics_ext.closure_precision` / `build_semantic_oracle` credit an added edge
   if `C ⊑ P` holds in the pristine *closure*. Confirmed it only improves
   corroboration (closure AP advantage ≥ exact at every redundancy); fig4 overlays
   both frontiers. Also fixed a PYTHONHASHSEED-dependence in `build_synthetic_dag`
   that made the sdag results non-reproducible.
2. ~~**Stats**~~ **DONE this session**: `src/experiment/stats.py` (bootstrap CIs,
   paired Wilcoxon, Holm–Bonferroni); `stats_money.csv` + fig6 CI error bars.
   Result: the redundancy-scaling advantage is significant across the synthetic
   sweep (Holm p ≤ 7e-4) and 0 on trees; Wikidata is positive but **marginal**
   (CI excludes 0, raw p=0.028, Holm p=0.11) — its redundancy (0.036) is too low.
3. **Denser real DAG** (now the top open item): `wikidata:Q42889:3` redundancy is
   only 0.036, so the real effect is small/marginal. Fetch a higher-redundancy real
   slice (richer-multiple-inheritance root, or bump `max_classes`/`max_depth`) to
   land a significant real-data point further along the fig6 trend. WDQS was flaky
   (502s) on 2026-06-20.
4. **SHACL / SPARQL baselines** (secondary task, §8/§10) — see below.
5. **Optional realism**: constrain `guidance_rewire_rate` to same-or-higher-layer
   targets so the damaged taxonomy stays acyclic (caveat 4), or keep the
   unconstrained adversary and just disclose it.

## Secondary task: SHACL / SPARQL baselines

Answer the "already solved by validation" objection — `docs/experiment-design.md`
§8 and the build spec in §10. `flat+SHACL` (pyshacl) and `flat+SPARQL property
path` (rdflib); compare detection F1 + localization, not just validation. Both
deps are already in `requirements.txt`.

## Open decisions / backlog
- Corroboration signal: independent-path-count vs sibling-agreement vs hybrid.
- Whether to keep the per-type prior as a multiplier on the evidence score.
- Held-out build-from-complement pipeline (oracle.py has `holdout_split`; the
  full pipeline is still TODO) for a second non-circular recall signal.
- Bump seeds for pre-registered primary cells (design §4) before final figures.

## Kickoff prompt (paste to start the next session)

> We're resuming the **Class Thread integrity & repair preprint** (branch
> `feat/integrity-repair-experiment`). Read `docs/next-session.md`,
> `docs/experiment-design.md` (esp. §11 findings — read the corroboration
> resolution + its 7 caveats), and `master-prompt.md`. The harness is in
> `src/experiment/`; run `.venv/bin/python -m pytest -q` (expect 68 passing) to
> confirm green. (`python` is not on PATH — use `.venv/bin/python`.)
>
> The **evidence-based per-proposal confidence is DONE** (corroboration scorer →
> graded PR frontier scaling with graph redundancy; §11), including the
> closure-aware precision metric, a determinism fix, and the **stats module**
> (bootstrap CIs + paired Wilcoxon + Holm; significant across the synthetic sweep,
> marginal on the sparse real slice). Pick up the recommended next steps in
> `docs/next-session.md` → "Primary task — COMPLETE": (1) a **denser real Wikidata
> slice** to push the real-data point to significance on fig6, and/or (2) the
> **flat+SHACL / flat+SPARQL-property-path baselines** (`docs/experiment-design.md`
> §8/§10).
>
> Framing is data-driven — report honestly, including null/negative results.
> Commit incrementally on this branch and keep `docs/experiment-design.md` §11
> updated.
