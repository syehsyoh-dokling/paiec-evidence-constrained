# How to rerun this entry

Everything below was run on one Windows laptop (CPU; GPU optional) with Python 3.13, numpy,
pandas, pyarrow, torch, transformers, matplotlib. Set one environment variable first:

    PAIEC_ROOT=<folder that contains localsim/, paiec_v31_ble_hybrid/data/ and the submission archives>

All scripts resolve their paths from `PAIEC_ROOT` (default: the development machine's folder).
The Codabench token file for the upload scripts is read from `PAIEC_CODABENCH_TOKEN_FILE`;
no script needs it for evaluation.

## 1. Check the submission archive (organizers' checker)

    python code/organizers_tools/check_submission_zip.py submission/paiec_submission_v153.zip
    python code/organizers_tools/check_submission_zip.py submission/paiec_submission_v403.zip

## 2. Public training tables

Place the measurement-db tables in `$PAIEC_ROOT/paiec_v31_ble_hybrid/data/<benchmark>/{subjects,items,response}.parquet`
for afrimedqa, agentdojo, ai2d_test, hle, matharena, mathvista_mini, mmbench_v11, mmlupro
(the submission archives carry the same 200-item payload under `data/`). The dual-encoder
feature cache `cached_dual_encoder_1600.npz` at `$PAIEC_ROOT` is rebuilt by
`code/priors/export_v38_universal_irt.py` if absent.

## 3. Leave-one-benchmark-out comparison (Tables 1, 3 and the figures)

    python code/localsim/sim_local.py --zips submission/paiec_submission_v153.zip submission/paiec_submission_v403.zip \
        --pairs-per-fold 16 --context-pairs 2 --tag rerun \
        --folds afrimedqa agentdojo ai2d_test matharena mathvista_mini mmbench_v11 mmlupro
    python code/localsim/rebuild_combined.py rerun
    python code/localsim/paired_vs.py rerun paiec_submission_v153 paiec_submission_v403
    python code/localsim/calib_report.py $PAIEC_ROOT/localsim/results/rerun

One pair takes about 30 s on CPU; 112 pairs per variant plus context pairs take about 1.5 h per variant with 7 workers.
The organizers' baselines are built by `code/localsim/make_ablations.py` (empirical mean, smoothed empirical Bayes) and run the same way.

## 4. Budget-0 grid and nested selection

    python code/localsim/b0_eval_driver.py
    python code/localsim/stats_rigor.py

## 5. QA gate on a candidate folder (static gates + LOBO verdict)

    python code/localsim/qa_gate.py <folder_with_zips> --out $PAIEC_ROOT/localsim/results/qa_rerun

## 6. Hosted-score statistics (from the saved scoring logs in results/)

    python code/localsim/hosted_defensible.py
    python code/localsim/hosted_noise_cause.py

## 7. Within-benchmark unseen-item test (Appendix)

    python code/localsim/seen_unseen_b0.py <measurement-db long-form root> out.json submission/paiec_submission_v153.zip

## 8. Agent loop (development tool, not part of prediction)

    python code/agent/qwen_generate_70.py --out batch --n 69 --provider hf_local
    python code/localsim/autopilot.py batch --name rerun --jobs 4

Expected outputs and the numbers reported in the paper are in `results/` (per-pair results in
`results/calib16_ctx2_combined.json`; statistics in `results/stats_rigor.json`).
