"""Build qwen_knowledge_v2.json: everything Qwen needs to pick a baseline and propose
candidates, with the noise facts that make 'lowest hosted score' the wrong selector.

Sources: online_meta_dataset.csv (hosted scores), all_scoring_logs_cache.json (per-pair
formative tables), localsim results (LOBO), quota_17 ledger (pending submissions).
"""
from __future__ import annotations
import csv, json, re, statistics, collections, random
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LOMBA = ROOT.parent
OUT = ROOT / "qwen_knowledge_v2.json"

rows = list(csv.DictReader((ROOT / "online_meta_dataset.csv").open(encoding="utf-8")))
scored = [r for r in rows if r.get("online_score")]
scores = sorted(float(r["online_score"]) for r in scored)
by_date = collections.defaultdict(list)
for r in scored:
    by_date[r["date"][:10]].append(float(r["online_score"]))
by_sha = collections.defaultdict(list)
for r in scored:
    if r["zip_sha256"]:
        by_sha[r["zip_sha256"]].append(float(r["online_score"]))
replicates = [{"sha256": k[:12], "scores": v} for k, v in by_sha.items() if len(v) > 1]

# formative per-pair tables
logs = json.load((LOMBA / "all_scoring_logs_cache.json").open(encoding="utf-8"))
row_re = re.compile(r"^(subject_\d+)\s+(benchmark_\d+)\s+(\d+)\s+([0-9.]+)\s+([0-9.]+)\s*$")
bm = collections.defaultdict(lambda: {"alc": [], "b0": [], "b31": [], "n_items": []})
all_pair_alc = []
npairs = []
for sid, rec in logs.items():
    section = "alc"; cnt = 0
    for line in rec.get("txt", "").splitlines():
        m = re.match(r"^Label budget:\s*(\d+)", line)
        if m:
            section = int(m.group(1)); continue
        m = row_re.match(line.strip())
        if not m:
            continue
        b, n, val = m.group(2), int(m.group(3)), float(m.group(4))
        if section == "alc":
            bm[b]["alc"].append(val); bm[b]["n_items"].append(n); all_pair_alc.append(val); cnt += 1
        elif section == 0:
            bm[b]["b0"].append(val)
        elif section == 31:
            bm[b]["b31"].append(val)
    if cnt:
        npairs.append(cnt)
random.seed(0)
k = int(statistics.median(npairs))
sim_sd = statistics.pstdev([statistics.mean(random.sample(all_pair_alc, k)) for _ in range(5000)])
per_benchmark = {
    b: {"n_pairs_seen": len(v["alc"]), "mean_alc": round(statistics.mean(v["alc"]), 4),
        "mean_b0": round(statistics.mean(v["b0"]), 4) if v["b0"] else None,
        "mean_b31": round(statistics.mean(v["b31"]), 4) if v["b31"] else None,
        "median_items": statistics.median(v["n_items"])}
    for b, v in sorted(bm.items(), key=lambda kv: kv[1]["alc"] and statistics.mean(kv[1]["alc"]))
}

# LOBO results
lobo = {}
comb = LOMBA / "localsim" / "results" / "full_v153_vs_baselines" / "combined.json"
if comb.exists():
    res = json.load(comb.open())
    base = res.get("v153_nokeys", {})
    common = sorted(set.intersection(*[set(v) for v in res.values()]))
    for name, v in res.items():
        d = [v[p]["alc"] - base[p]["alc"] for p in common]
        bud = [statistics.mean(v[p]["briers"][i] for p in common) for i in range(6)]
        lobo[name] = {"mean_alc": round(statistics.mean(v[p]["alc"] for p in common), 4),
                      "diff_vs_v153": round(statistics.mean(d), 4),
                      "se": round(statistics.stdev(d) / len(d) ** 0.5, 4) if len(d) > 1 else None,
                      "wins": sum(x < 0 for x in d), "n_pairs": len(common),
                      "brier_by_budget_0_1_3_7_15_31": [round(x, 4) for x in bud]}
b0grid = {}
summ = LOMBA / "localsim" / "results" / "b0eval" / "summary.json"
if summ.exists():
    t = json.load(summ.open())["table"]
    for row in t[:12] + [r for r in t if r[0] in ("v153", "const_0.5", "oracle_pair_const")]:
        b0grid[row[0]] = {"fold_mean_b0": round(row[1], 4), "diff_vs_v153": round(row[3], 4), "se": round(row[4], 4), "wins": f"{row[5]}/{row[6]}"}

pending = []
led = LOMBA / "quota_17_b0fix_20261104" / "upload_ledger.json"
if led.exists():
    pending = [{"sid": r.get("sid"), "file": r["filename"], "config": r["config"], "status": "scored_pending" if r.get("sid") else "quota_rejected"}
               for r in json.load(led.open())]

slots = {
    "B0": {"anchor": "if not labeled:  (zero-label branch of predict)", "params": {"center": [0.48, 0.56], "shrink": [0.0, 1.0], "ease": [0.0, 1.2]},
           "baseline_values": {"center": 0.449, "shrink": 0.20, "ease": 0.238, "hard_cap": 0.495}, "evidence": "LOBO B0 0.2559 vs const 0.25; fix c0.50/s0.5/e1.0 -> 0.2482"},
    "LOW_BUDGET_SHRINK": {"anchor": "if 0 < successes < count: ... count == 1 / count <= 3 blends", "params": {"blend_count1": [0.20, 0.60], "blend_count_le3": [0.05, 0.35], "max_count": [3, 7], "debias": [0.20, 0.32]},
                          "baseline_values": {"blend_count1": 0.340, "blend_count_le3": 0.100, "max_count": 3, "debias": 0.265}, "evidence": "LOBO B1 0.2497 ~ B0 0.2559: first label adds almost nothing"},
    "MIXTURE_PRIOR": {"anchor": "_mixture_bayes_posterior comps (w,a,b)", "params": {"a_b_each_regime": [0.3, 8.0], "w1_w3": [0.10, 0.40]},
                      "baseline_values": {"regime1": [0.38, 6.80], "regime2": [3.10, 5.10], "regime3": [6.20, 2.80], "w1": 0.24, "w3": 0.26}, "evidence": "pseudo-counts control over-reaction at 1-3 labels"},
    "PEER_SHIFT": {"anchor": "prior_shift += 0.45 * (peer_bm_mean - 0.386) * n/(n+2.0)", "params": {"gain": [0.25, 0.70], "center": [0.35, 0.45], "shrink_n": [1.0, 6.0]},
                   "baseline_values": {"gain": 0.45, "center": 0.386, "shrink_n": 2.0}, "evidence": "peers are the only signal at budget 1"},
    "IRT_ENVELOPE": {"anchor": "env = clip(p(1-p)/0.06, 0.15, 1.0); p_final = p_base + env*0.32*d_rel_2pl", "params": {"env_scale": [0.04, 0.12], "env_floor": [0.05, 0.40], "irt_weight": [0.0, 0.50]},
                     "baseline_values": {"env_scale": 0.06, "env_floor": 0.15, "irt_weight": 0.32}, "evidence": "LOBO B31 0.1947 ~ oracle pair constant 0.1905: IRT adds no item discrimination"},
}

pack = {
    "purpose": "Knowledge for the Qwen candidate generator (PAIEC, Codabench 17828 / phase 29785).",
    "read_first": [
        "Hosted formative scores are redrawn per submission from 8-9 subject-benchmark pairs; one submission's score has sd ~%.3f." % sim_sd,
        "The lowest hosted score (%.6f) is the best of %d draws of near-identical code; the family median is %.3f. Do NOT select a baseline by lowest hosted score." % (scores[0], len(scores), statistics.median(scores)),
        "Select the baseline by LOBO expected ALC (localsim) and by code lineage; v153 is the current baseline because it beats the official baselines by 0.012 ALC in LOBO and every variant family descends from it.",
        "A candidate is judged only by paired LOBO dALC vs v153 (qa_gate.py). Expected hosted score ~ 0.185 + dALC.",
        "Changes smaller than 1% of a slot's range are replicates, not candidates. Replicates are allowed only as explicit controls.",
    ],
    "hosted_scores": {"n_scored": len(scores), "min": scores[0], "median": statistics.median(scores), "max": scores[-1],
                      "sd_all": round(statistics.pstdev(scores), 4), "sd_single_submission_resampled": round(sim_sd, 4),
                      "same_zip_replicates": replicates,
                      "daily": {d: {"n": len(v), "best": min(v), "median": round(statistics.median(v), 4)} for d, v in sorted(by_date.items())}},
    "all_scored_submissions": [{"sid": r["submission_id"], "file": r["filename"], "date": r["date"], "score": float(r["online_score"]), "sha256": r["zip_sha256"][:12]} for r in scored],
    "hidden_benchmarks_from_formative_logs": per_benchmark,
    "lobo_full_curve_112_pairs": lobo,
    "lobo_budget0_grid_905_pairs": b0grid,
    "baseline": {"name": "v153", "zip": "localsim/variants/v153_nokeys.zip", "hosted_best_draw": scores[0], "hosted_family_median": statistics.median(scores),
                 "control_rule": "one byte-identical copy of v153_nokeys.zip per upload day, tagged control; qa_gate exempts it by tag"},
    "slots": slots,
    "pending_submissions_2026_11_04": pending,
}
OUT.write_text(json.dumps(pack, indent=1, ensure_ascii=False), encoding="utf-8")
print(json.dumps({"out": str(OUT), "n_scored": len(scores), "sd_single": round(sim_sd, 4), "benchmarks": list(per_benchmark), "lobo_entries": list(lobo), "size_kb": OUT.stat().st_size // 1024}))
