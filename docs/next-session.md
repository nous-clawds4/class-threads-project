# Handoff — Class Thread Integrity & Repair Preprint

Resume document for the next working session. Pair this with
[`docs/experiment-design.md`](experiment-design.md) (full protocol + findings)
and [`master-prompt.md`](../master-prompt.md) (project origin).

## TL;DR — current state (2026-06-20)

- **Goal:** an arXiv preprint demonstrating the Class Thread / dual-node model
  via an **integrity & repair** empirical study. Framing is **data-driven**:
  make the stronger dual-node claim only if the data supports it.
- **Branch:** `feat/integrity-repair-experiment` (pushed to `origin`), ~5 commits
  ahead of `main`. **44 tests pass** (`python -m pytest -q`).
- **Datasets wired in:** `synthetic` (control), `wordnet:vehicle.n.01:2` (scale),
  `wikidata:Q42889:3` (credibility — real DAG, 162 concepts / 120 instances / 6
  multiple-inheritance concepts, cached at `data/raw/wikidata_Q42889_d3.json`).
- **Figures:** `results/figures/` — fig1 (coverage degradation/recovery), fig2
  (fidelity vs guidance corruption), fig3 (θ is on/off), fig5 (detection confusion).

## How to run

```bash
python -m pytest -q                       # 44 tests
python -m src.experiment.experiment       # Tier-A grid -> results/results.parquet
python -m src.experiment.money            # money grid  -> results/money.parquet
python -m src.experiment.figures          # fig1, fig5
# money figures (fig2, fig3) are written by the money run itself
```
Wikidata loads from the committed cache (offline). To refresh:
`build_wikidata_graph(..., refresh=True)` (one WDQS fetch; be polite).

## What's built (`src/experiment/`)

| Module | Role |
|---|---|
| `util.py` | deterministic blake2b-seeded RNG (`make_rng`) |
| `oracle.py` | `freeze_oracle` (frozen EXPECTED + PRISTINE_EDGES), `holdout_split` |
| `corruption.py` | Tier-A arms (E-HX/E-SO/E-HE/E-MIX/N-REM, UNIF/TARG) + Tier-B (`guidance_rewire_rate`, distractors) |
| `metrics_ext.py` | edge precision/recall, hallucination, coverage, recovery, pair false-positives |
| `ablation.py` | single-node comparator (homologous flat corruption + closure detection) |
| `experiment.py` | Tier-A factor-grid runner + `build_dataset` dispatch |
| `money.py` | autonomous-repair runner (rewire × θ) + fidelity/θ-gate figures |
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

## Primary task (next session): evidence-based per-proposal confidence

**Why.** θ today only turns repair on/off because confidence is assigned per edge
*type*, so good and fabricated proposals carry identical confidence and cannot be
separated. A real precision–recall knob needs a confidence that scores *each
proposed edge* by its evidential support — which the Wikidata DAG's multiple
inheritance finally provides.

**Design sketch (refine, don't follow blindly):**
1. Make repair's confidence pluggable: `repair_threads(..., confidence_fn=None)`,
   defaulting to today's per-type constants (keep as the baseline/ablation).
2. Implement a `corroboration_confidence(graph, proposal)`: blend the per-type
   prior with a structural corroboration signal, e.g. the number of *independent
   surviving paths* in the taxonomy/extension layer that support the proposed
   link (a lone rewired/distractor edge → ~0 corroboration → low confidence; a
   real edge corroborated by sibling structure / alternative routes → high). A
   sibling-agreement ratio (how many co-hyponyms already thread to the target,
   AMIE-style) is a strong alternative/companion signal.
3. Re-run the money grid with the new confidence and sweep θ.

**Acceptance criteria:** on `wikidata:Q42889:3` under Tier-B rewiring, the
precision–recall points **spread into a real graded curve across θ** (not the
current single collapsed point), and at matched recall the evidence-based
confidence achieves **higher precision** than the per-type baseline. Report the
ablation (per-type vs evidence-based) honestly even if the gain is small. Update
`docs/experiment-design.md` §11.

**Watch for:** on near-tree data (synthetic/WordNet) corroboration ≈ 0 by
construction — the knob should help mainly where redundancy exists (Wikidata).
That contrast is itself a result. Consider a larger/denser Wikidata slice (bump
`max_classes`/`max_depth`, or a root with richer multiple inheritance) if the
redundancy signal is too sparse.

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
- Stats module (bootstrap CIs, paired Wilcoxon) for publication-grade error bars.
- Bump seeds for pre-registered primary cells (design §4) before final figures.

## Kickoff prompt (paste to start the next session)

> We're resuming the **Class Thread integrity & repair preprint** (branch
> `feat/integrity-repair-experiment`). Read `docs/next-session.md`,
> `docs/experiment-design.md` (esp. §11 findings), and `master-prompt.md`. The
> harness is in `src/experiment/`; run `python -m pytest -q` (expect 44 passing)
> to confirm green.
>
> **Primary task:** design and implement an **evidence-based, per-proposal
> confidence** for `repair_threads` so the threshold θ becomes a real
> precision–recall knob (today it's on/off because confidences are fixed per edge
> *type*). Test on the Wikidata DAG (`wikidata:Q42889:3`), whose multiple-
> inheritance redundancy provides corroborating paths. Success = a graded
> precision–recall curve emerges in the money figure and beats the per-type
> baseline at matched recall; see `docs/next-session.md` → "Primary task" for the
> design sketch and acceptance criteria. Keep the per-type confidence as the
> ablation baseline.
>
> **If time:** the flat+SHACL / flat+SPARQL-property-path baselines
> (`docs/experiment-design.md` §8/§10).
>
> Framing is data-driven — report honestly, including null results. Commit
> incrementally on this branch and keep `docs/experiment-design.md` §11 updated.
