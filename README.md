# Supplementary material: PAIEC entry (selected Codabench submission 1034049 = v403; leaderboard entry 1020778 = v153)

submission/paiec_submission_v153.zip  The Codabench submission (model.py, labeling.py, models.txt,
    submission_config.json without credentials, data/, src/). Identical predictor code to the
    selected submission; API keys removed and three code comments reworded.
code/localsim/           Leave-one-benchmark-out simulator, budget-0 grid, QA gate, ablations.
code/agent/              Slot-constrained code-revision agent (local Qwen2.5-Coder) and its contract.
code/priors/             Scripts that fitted data/subject_priors.json and data/universal_item_irt.json.
code/organizers_tools/   Copies of the organizers' acquisition hook and ZIP checker used by the simulator.
results/                 LOBO results referenced in the report.

## Reproduce the submission check
    python code/organizers_tools/check_submission_zip.py submission/paiec_submission_v153.zip

## Reproduce the LOBO comparison (needs the public measurement-db parquet files inside the
## submission's data/ folder, PyTorch, transformers, numpy, pandas; GPU optional)
    python code/localsim/sim_local.py --zips submission/paiec_submission_v153.zip \
        --pairs-per-fold 16 --folds afrimedqa agentdojo ai2d_test matharena mathvista_mini mmbench_v11 mmlupro
    python code/localsim/b0_eval_driver.py
Paths at the top of each script point to the development machine; set ROOT/DATA to the
unpacked submission folder. Network calls are disabled in the simulator
(PAIEC_DISABLE_BLE_NETWORK=1).

## Agent loop
    python code/agent/qwen_generate_70.py --out batch --n 69 --provider hf_local
    python code/localsim/qa_gate.py batch --out qa
The Kaggle kernel in code/agent/kaggle_kernel runs the same loop on a T4 with the submission
folder mounted as a dataset.

## Note on the first Kaggle kernel
The first agent kernel (v1, code/agent/kaggle_kernel/paiec_qwen_slot_generator_70.py) let the student write code edits; all of its budget-0 edits failed the QA gate for the same None-handling error, which is why the second kernel renders edits from templates. An even earlier script-driven kernel is not included because its submissions were not agent-generated.
