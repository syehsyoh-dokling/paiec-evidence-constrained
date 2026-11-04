# Leave-One-Benchmark-Out (LOBO) Local Evaluation — v153 and the Budget-0 Fix

Date: 2026-11-04. Tooling: `localsim/sim_local.py`, `localsim/sim_worker.py`,
`localsim/b0_eval_fold.py`, `localsim/b0_eval_driver.py`, `localsim/make_b0_variants.py`
(all under `C:\Users\Saifuddin\Documents\LOMBA\localsim`).

## 1. Why a local simulator is needed

Hosted formative scores on Codabench are redrawn per submission (8–9 subject–benchmark
pairs from 7 hidden benchmarks). From 172 scored submissions and 38 parsed scoring logs:

| Quantity | Value |
|---|---|
| Pair-level Brier ALC standard deviation | 0.08 |
| Standard deviation of one submission's score (9-pair resample) | 0.027 |
| Same ZIP submitted twice (v40_clean) | 0.2016 vs 0.1932 |
| Median hosted score of v153-family variants, 27–31 Oct | 0.185 |
| Best single draw (v153, submission 1020778) | 0.1276 |

The best-of-172 draw is roughly 3 SD below the family mean, which is what selection on
noise produces. Single-knob comparisons on hosted scores therefore cannot rank candidates.
The final ranking uses one common hidden subset for all teams, so the expected score is
what matters.

## 2. Protocol of the simulator

* Training data: the 8 public measurement-db benchmarks bundled in the submission
  (`data/<benchmark>/*.parquet`). `hle` is excluded from text folds because its local
  `content` field holds only an ID (`hle:<hex>`), not the task text.
* For each held-out benchmark (fold), the submission's fitted artifacts are patched so
  they contain no information from that benchmark: the held-out `bm_deltas` entry is
  removed from `subject_priors.json` (subject deltas re-averaged over the remaining
  benchmarks; subjects known only from the held-out benchmark become unseen), and the
  universal 2PL ridge (`w_ease`, `w_disc`, `feat_mean`, centroids) is refit without it.
  The benchmark ID is anonymised as on Codabench.
* Pairs: subjects with ≥80 distinct items; items split 50/50 into acquisition and
  evaluation pools with a fixed hash; nested label budgets 0, 1, 3, 7, 15, 31; draws of
  1–3 pairs share the benchmark so peer labels are present in `labeled`; the
  submission's own `labeling.py` drives acquisition through the organizers'
  `streaming_ingestion.AcquisitionHook`.
* Brier per pair over all recorded responses; ALC = 0.1·B0 + 0.2·(B1+B3+B7+B15) + 0.1·B31;
  plain mean over pairs. Network calls are disabled (`PAIEC_DISABLE_BLE_NETWORK=1`).
* Full-curve run: 7 folds × 16 pairs = 112 pairs, identical pairs for every variant
  (paired differences). Budget-0 grid: all eligible pairs (905) because B0 predictions
  do not depend on labels.

## 3. Full learning curve: v153 vs the organizers' baselines (112 paired pairs)

| Variant | ALC | diff vs v153 | SE | wins | B0 | B1 | B3 | B7 | B15 | B31 |
|---|---|---|---|---|---|---|---|---|---|---|
| v153 (keys stripped) | 0.2220 | — | — | — | 0.2559 | 0.2497 | 0.2225 | 0.2098 | 0.2026 | 0.1947 |
| smoothed_empirical_bayes (official) | 0.2340 | +0.0120 | 0.0025 | 35/112 | 0.2500 | 0.2643 | 0.2453 | 0.2209 | 0.2112 | 0.2071 |
| empirical_mean (official) | 0.2633 | +0.0413 | 0.0039 | 17/112 | 0.2500 | 0.3868 | 0.2653 | 0.2238 | 0.2118 | 0.2072 |

Per-fold ALC (v153): afrimedqa 0.246, agentdojo 0.205, ai2d_test 0.214, matharena 0.229,
mathvista_mini 0.212, mmbench_v11 0.201, mmlupro 0.248.

Reading:

1. v153 beats both official baselines, by 0.012 and 0.041 ALC, with small standard errors.
   This is the first leakage-free evidence that the model adds value.
2. **Budget 0 is worse than predicting 0.5** (0.256 vs 0.250), matching the hosted logs
   (0.2546 on hidden benchmarks). The cold-start prior transfers nothing.
3. **The first label adds almost nothing** (B1 0.2497 vs B0 0.2559). Budgets 1 and 3
   carry 40 % of the metric weight.
4. **At 31 labels v153 (0.1947) is no better than a constant equal to the pair's true
   pass rate** (oracle floor 0.1905 fold-mean, 0.1727 pair-mean). The kernel ridge,
   semantic transfer and IRT terms produce no item-level discrimination beyond the pair
   mean on held-out benchmarks.

## 4. Budget-0 grid (905 pairs, 7 folds)

Alternative B0 formula: `p = clip(c + s·(subject_prior_mean − 0.5) + e·d_item_ease, 0.02, 0.98)`,
replacing v153's `clip(p0 + 0.238·d, 0.005, 0.495)`.

| Configuration | fold-mean B0 | pair-mean B0 | diff vs v153 | SE | wins |
|---|---|---|---|---|---|
| v153 | 0.2562 | 0.2596 | — | — | — |
| constant 0.5 | 0.2500 | 0.2500 | −0.0096 | 0.0009 | 582/905 |
| c=0.50, s=0.5, e=1.0 (neutral center) | 0.2471 | 0.2449 | −0.0147 | 0.0010 | 619/905 |
| c=0.55, s=0.5, e=1.0 (best in LOBO) | 0.2431 | 0.2377 | −0.0219 | 0.0018 | 589/905 |
| c=0.45, s=0.5, e=1.0 | 0.2561 | 0.2571 | −0.0026 | 0.0005 | 499/905 |
| oracle: constant = pair pass rate | 0.1905 | 0.1727 | −0.0870 | 0.0022 | 887/905 |

Reading:

* Removing the 0.495 cap and using the subject prior with full-weight item ease improves
  B0 on every fold's average, but only by 0.01–0.02 Brier, i.e. 0.001–0.002 ALC.
* `c=0.55` wins locally because the public training benchmarks are easy on average
  (ai2d 0.75, mmbench 0.81 pass rate). The hidden benchmarks look harder (two of seven
  have near-zero pass rates in the formative logs), so `c=0.50` is the robust choice;
  `c=0.45` loses the gain. The center cannot be tuned to the hidden set.
* The gap to the oracle constant (0.19 vs 0.25) is the real cold-start opportunity: it
  needs priors that predict the pair's base rate, which requires far more training
  benchmarks (measurement-db now has hundreds; the submission uses 8) or an item-level
  proxy signal (a local LLM attempting the item).

## 4b. Budget-0 fix verified end to end (full curve, same 112 paired pairs)

| Variant | ALC | diff vs v153 | SE | wins | B0 | B1…B31 |
|---|---|---|---|---|---|---|
| v153 | 0.2220 | — | — | — | 0.2559 | unchanged |
| b0fix c=0.50, s=0.5, e=1.0 | 0.2212 | −0.0008 | 0.0003 | 75/112 | 0.2482 | identical to v153 |
| b0fix c=0.55, s=0.5, e=1.0 | 0.2209 | −0.0011 | 0.0005 | 73/112 | 0.2448 | identical to v153 |

Both fixes improve ALC with a paired standard error well below the effect, but the gain
is about 0.001 ALC: 25 times smaller than the hosted per-submission noise (0.027). It is
worth shipping (it removes a provably harmful cap and costs nothing), but it cannot move a
hosted score to 0.10x by itself; only a lucky formative draw does that, and the final
evaluation does not use formative draws. Both ZIPs pass `check_submission_zip.py`.
The 17 quota ZIPs in `quota_17_b0fix_20261104/` are the 16 best LOBO budget-0
configurations (8 with c=0.55, 8 with c=0.50) plus a v153 control; all pass
`localsim/validate_17_quota.py` (compile, diff limited to the B0 block, no API keys,
unique SHA-256, filename ≤ 40 chars). Nothing has been uploaded.

## 5. Caveats

* The public parquet subsets have 200 items per benchmark; hidden benchmarks have
  44–113 evaluated items per pair and different pass-rate distributions, so absolute
  local ALC (0.22) is not comparable to hosted scores (0.18 median). Paired differences
  between variants are the valid use.
* `hle` is not simulated (ID-only content). Multimodal benchmarks are simulated from text
  only, exactly as the model sees them on Codabench.
* All ~30 hand-set constants in v153 were previously tuned on hosted formative scores;
  the LOBO patch cannot remove that leakage, so v153's local numbers may be optimistic.
