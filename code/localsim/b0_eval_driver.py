"""Run b0_eval_fold.py for every text fold and aggregate the budget-0 grid."""
import json
import os
import pathlib
import subprocess
import sys

import numpy as np

SIM = pathlib.Path(r"c:\Users\Saifuddin\Documents\LOMBA\localsim")
FOLDS = ["afrimedqa", "agentdojo", "ai2d_test", "matharena", "mathvista_mini", "mmbench_v11", "mmlupro"]
OUT = SIM / "results" / "b0eval"
OUT.mkdir(parents=True, exist_ok=True)

for bm in FOLDS:
    out = OUT / f"{bm}.json"
    if out.exists():
        continue
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    p = subprocess.run([sys.executable, str(SIM / "b0_eval_fold.py"), bm, str(out)], capture_output=True, text=True, env=env)
    print(p.stdout.strip())
    if p.returncode != 0:
        print("FAIL", bm, p.stderr[-2000:])

res = {bm: json.load(open(OUT / f"{bm}.json")) for bm in FOLDS if (OUT / f"{bm}.json").exists()}
configs = sorted(next(iter(res.values()))["configs"])
# fold-balanced mean (each fold weighted equally, as Codabench draws pairs across benchmarks)
table = []
for k in configs:
    fold_means = [res[bm]["configs"][k]["mean_b0"] for bm in res]
    pooled = np.concatenate([res[bm]["configs"][k]["per_pair"] for bm in res])
    base = np.concatenate([res[bm]["configs"]["v153"]["per_pair"] for bm in res])
    diff = pooled - base
    table.append((k, float(np.mean(fold_means)), float(np.mean(pooled)), float(np.mean(diff)), float(np.std(diff, ddof=1) / np.sqrt(len(diff))), int((diff < 0).sum()), len(diff), fold_means))
table.sort(key=lambda t: t[1])
print(f"\n{'config':28s} {'foldMeanB0':>10s} {'pairMeanB0':>10s} {'diff_vs_v153':>12s} {'SE':>7s} {'wins':>9s}  " + " ".join(f"{bm[:8]:>8s}" for bm in res))
for k, fm, pm, dm, se, w, n, fms in table[:25] + [t for t in table if t[0] in ("v153", "const_0.5", "oracle_pair_const")]:
    print(f"{k:28s} {fm:10.4f} {pm:10.4f} {dm:+12.4f} {se:7.4f} {w:4d}/{n:<4d}  " + " ".join(f"{x:8.4f}" for x in fms))
json.dump({"table": table}, open(OUT / "summary.json", "w"))
