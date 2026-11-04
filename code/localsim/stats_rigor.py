"""Statistical-rigor pass on existing results (no new simulations).

1. Nested LOBO for the budget-0 grid: for each held-out fold, select the grid configuration
   on the other six folds (fold-mean B0), evaluate it on the held-out fold; report the
   out-of-selection improvement vs v153 and which configuration each fold selected.
2. Fold-level SE and a two-tier significance label for every verdict file (pair-level and
   fold-level), written back as lobo["se_fold"], lobo["fold_significant"].
3. Multiple testing: one-sided p-values (normal approx on pair-level SE) for every candidate
   with a LOBO result; Benjamini-Hochberg at q=0.05; how many PASS survive.
4. Oracle from the acquisition pool: constant = pass rate of the pair's acquisition items
   (out-of-sample w.r.t. the evaluation items), Brier on the 112 evaluation pairs.
5. LLM student vs no-LLM sampler on the budget-0 slot: Fisher exact test on PASS counts and
   a two-sample comparison of the mean dALC.
Writes localsim/results/stats_rigor.json and prints a summary.
"""
import os
import glob
import json
import math
import pathlib
import statistics
from itertools import combinations

import numpy as np

ROOT = pathlib.Path(os.environ.get("PAIEC_ROOT", r"c:\Users\Saifuddin\Documents\LOMBA"))
SIM = ROOT / "localsim"
FOLDS = ["afrimedqa", "agentdojo", "ai2d_test", "matharena", "mathvista_mini", "mmbench_v11", "mmlupro"]
W0 = 0.1
out = {}

# ---------------- 1. nested LOBO on the budget-0 grid
b0 = {bm: json.load(open(SIM / "results" / "b0eval" / f"{bm}.json")) for bm in FOLDS}
configs = [k for k in b0[FOLDS[0]]["configs"] if k.startswith("c0.")]
fold_mean = {bm: {k: b0[bm]["configs"][k]["mean_b0"] for k in configs} for bm in FOLDS}
nested = []
for held in FOLDS:
    others = [f for f in FOLDS if f != held]
    best = min(configs, key=lambda k: statistics.mean(fold_mean[f][k] for f in others))
    d = np.array(b0[held]["configs"][best]["per_pair"]) - np.array(b0[held]["configs"]["v153"]["per_pair"])
    nested.append({"held_out": held, "selected": best, "n_pairs": len(d), "delta_b0": float(d.mean()), "delta_alc": W0 * float(d.mean())})
nested_mean_alc = statistics.mean(r["delta_alc"] for r in nested)
nested_se_fold = statistics.stdev([r["delta_alc"] for r in nested]) / math.sqrt(len(nested))
out["nested_lobo_b0"] = {"per_fold": nested, "mean_delta_alc": nested_mean_alc, "se_fold": nested_se_fold,
                         "folds_improved": sum(r["delta_alc"] < 0 for r in nested),
                         "selected_configs": sorted({r["selected"] for r in nested})}
print(f"1. nested LOBO budget-0: out-of-selection dALC = {nested_mean_alc:+.4f} (fold SE {nested_se_fold:.4f}), folds improved {out['nested_lobo_b0']['folds_improved']}/7; selected: {out['nested_lobo_b0']['selected_configs']}")

# ---------------- 2. fold SE + tiers on verdict files
tiers = {"PASS_both": 0, "PASS_pair_only": 0, "HOLD": 0, "REJECT": 0}
cands = []
for f in sorted(glob.glob(str(SIM / "results" / "qa_*" / "qa_*.json"))):
    r = json.load(open(f))
    if not isinstance(r, dict) or "verdict" not in r:
        continue  # qa_summary.json and other non-verdict files
    lobo = r.get("lobo")
    if lobo and lobo.get("per_fold"):
        pf = list(lobo["per_fold"].values())
        se_f = statistics.stdev(pf) / math.sqrt(len(pf)) if len(pf) > 1 else float("nan")
        if lobo["metric"] == "B0":
            se_f *= W0  # per_fold values are B0 differences; ALC weight 0.1
        lobo["se_fold"] = se_f
        lobo["fold_significant"] = bool(lobo["diff_alc"] + 2 * se_f < 0)
        json.dump(r, open(f, "w"), indent=1)
        cands.append({"file": r["file"], "batch": pathlib.Path(f).parent.name, "verdict": r["verdict"], "diff": lobo["diff_alc"], "se_p": lobo["se_alc"], "se_f": se_f,
                      "fold_sig": lobo["fold_significant"], "regions": r["info"].get("regions_hit")})
# one entry per distinct candidate: the same file name recurs across batches (cand_001_b0.zip in
# every student batch) and the same candidate has a screening and a confirmation ("_full") verdict;
# the confirmation verdict wins.
ded = {}
for c in cands:
    key = (c["batch"].replace("_full", ""), c["file"])
    if key not in ded or c["batch"].endswith("_full"):
        ded[key] = c
cands = list(ded.values())
for c in cands:
    if c["verdict"] == "PASS_UPLOAD":
        tiers["PASS_both" if c["fold_sig"] else "PASS_pair_only"] += 1
    elif c["verdict"] in ("HOLD", "REJECT"):
        tiers[c["verdict"]] += 1
out["tiers"] = tiers
print(f"2. tiers over {len(cands)} distinct candidates with a LOBO result:", tiers)

# ---------------- 3. multiple testing (BH on one-sided p-values, pair-level SE)
from math import erf, sqrt
def p_one_sided(diff, se):
    if not se or se <= 0:
        return 1.0 if diff >= 0 else 0.0
    z = -diff / se
    return 0.5 * (1 - erf(z / sqrt(2)))
tested = [c for c in cands if c["se_p"] is not None]
pvals = sorted((p_one_sided(c["diff"], c["se_p"]), (c["batch"].replace("_full", ""), c["file"])) for c in tested)
m = len(pvals); q = 0.05
k_max = 0
for i, (p, _) in enumerate(pvals, 1):
    if p <= q * i / m:
        k_max = i
bh_survivors = {name for _, name in pvals[:k_max]}
pass_files = {(c["batch"].replace("_full", ""), c["file"]) for c in tested if c["verdict"] == "PASS_UPLOAD"}
out["multiple_testing"] = {"n_tests": m, "bh_q": q, "n_bh_significant": k_max, "n_pass": len(pass_files), "n_pass_surviving_bh": len(pass_files & bh_survivors)}
print(f"3. BH q=0.05 over {m} tested candidates: {k_max} significant; PASS {len(pass_files)}, PASS surviving BH {len(pass_files & bh_survivors)}")

# ---------------- 4. oracle from the acquisition pool on the 112 pairs
res = json.load(open(SIM / "results" / "full_v153_vs_baselines" / "combined.json"))
base = res["v153_nokeys"]
orc_acq, orc_eval, b31 = [], [], []
for bm in FOLDS:
    spec = json.load(open(SIM / "specs" / f"{bm}_s0_p16.json"))
    for draw in spec["draws"]:
        for p in draw["pairs"]:
            if p["pair_id"] not in base:
                continue
            acq_rate = statistics.mean(a["label"] for a in p["acq"]) if p["acq"] else 0.5
            ys = [y for ev in p["eval"] for y in ev["labels"]]
            eval_rate = statistics.mean(ys)
            orc_acq.append(statistics.mean((acq_rate - y) ** 2 for y in ys))
            orc_eval.append(statistics.mean((eval_rate - y) ** 2 for y in ys))
            b31.append(base[p["pair_id"]]["briers"][5])
out["oracle"] = {"n_pairs": len(b31), "in_sample_eval_const": statistics.mean(orc_eval), "acquisition_pool_const": statistics.mean(orc_acq), "v153_B31": statistics.mean(b31),
                 "v153_minus_acq_oracle": statistics.mean(b31) - statistics.mean(orc_acq)}
print(f"4. oracle on {len(b31)} pairs: in-sample {statistics.mean(orc_eval):.4f} | acquisition-pool {statistics.mean(orc_acq):.4f} | v153 B31 {statistics.mean(b31):.4f}")

# ---------------- 5. student vs sampler on the budget-0 slot
def b0_pass_stats(batch):
    rows = [c for c in cands if c["batch"] == batch and c["regions"] == ["B0"]]
    passed = [c for c in rows if c["verdict"] == "PASS_UPLOAD"]
    return len(rows), len(passed), [c["diff"] for c in rows if c["diff"] is not None]
n_s, k_s, d_s = b0_pass_stats("qa_qwen_v2")
n_r, k_r, d_r = b0_pass_stats("qa_random_slot")
# Fisher exact (two-sided) on 2x2 [[k_s, n_s-k_s],[k_r, n_r-k_r]]
from math import comb
def fisher(a, b, c, d):
    n = a + b + c + d; r1, c1 = a + b, a + c
    p_obs = comb(r1, a) * comb(n - r1, c1 - a) / comb(n, c1)
    p = 0.0
    for x in range(max(0, c1 - (n - r1)), min(r1, c1) + 1):
        px = comb(r1, x) * comb(n - r1, c1 - x) / comb(n, c1)
        if px <= p_obs + 1e-12:
            p += px
    return p
fisher_p = fisher(k_s, n_s - k_s, k_r, n_r - k_r) if n_s and n_r else None
mean_s, mean_r = statistics.mean(d_s), statistics.mean(d_r)
se_diff = math.sqrt(statistics.variance(d_s) / len(d_s) + statistics.variance(d_r) / len(d_r))
out["student_vs_sampler_b0"] = {"student": {"n": n_s, "pass": k_s, "mean_dalc": mean_s}, "sampler": {"n": n_r, "pass": k_r, "mean_dalc": mean_r},
                                "fisher_p_two_sided": fisher_p, "diff_mean_dalc": mean_s - mean_r, "se_diff": se_diff}
print(f"5. budget-0 slot: student {k_s}/{n_s} PASS mean dALC {mean_s:+.4f} | sampler {k_r}/{n_r} PASS mean dALC {mean_r:+.4f} | Fisher p={fisher_p:.3f} | diff {mean_s-mean_r:+.4f} (SE {se_diff:.4f})")
json.dump(out, open(SIM / "results" / "stats_rigor.json", "w"), indent=1)
