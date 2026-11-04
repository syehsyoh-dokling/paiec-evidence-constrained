"""Refresh hosted scores for every submission we know of (CSV + all upload ledgers) and save
each submission's scoring-output log (per-pair tables) for paired analysis.
Output: localsim/results/hosted_scores.json  {sid: {file, score, ece, status, created, batch}}
        scoring_logs_all/<sid>.txt (only when not already saved)
"""
import os
import csv
import glob
import json
import pathlib
import time
import urllib.request

ROOT = pathlib.Path(os.environ.get("PAIEC_ROOT", r"c:\Users\Saifuddin\Documents\LOMBA"))
TOK = open(os.environ.get("PAIEC_CODABENCH_TOKEN_FILE", r"C:/Users/Saifuddin/.config/paiec/codabench-token")).read().strip()
HDR = {"Authorization": f"Token {TOK}", "Accept": "application/json"}
OUT = ROOT / "localsim" / "results" / "hosted_scores.json"
LOGS = ROOT / "scoring_logs_all"
LOGS.mkdir(exist_ok=True)

known = {}
for r in csv.DictReader(open(ROOT / "auto_discovery_paiec" / "online_meta_dataset.csv", encoding="utf-8")):
    known[int(r["submission_id"])] = {"file": r["filename"], "batch": "pre_lobo", "version": r.get("version")}
for led in sorted(ROOT.glob("*/upload_ledger.json")):
    for rec in json.load(open(led)):
        if rec.get("sid"):
            known[int(rec["sid"])] = {"file": rec.get("filename"), "batch": led.parent.name, "version": rec.get("version"), "config": rec.get("config"), "dalc_local": rec.get("dalc")}
prev = json.load(open(OUT)) if OUT.exists() else {}
res = {}
for i, (sid, meta) in enumerate(sorted(known.items())):
    cached = prev.get(str(sid))
    if cached and cached.get("status") == "Finished" and cached.get("score") is not None:
        res[str(sid)] = cached; continue
    try:
        req = urllib.request.Request(f"https://www.codabench.org/api/submissions/{sid}/", headers=HDR)
        d = json.loads(urllib.request.urlopen(req, timeout=30).read().decode())
    except Exception as e:
        res[str(sid)] = {**meta, "status": f"error: {str(e)[:60]}", "score": None}; time.sleep(1.0); continue
    score = ece = None
    for s in d.get("scores") or []:
        key = (s.get("column_key") or "").lower()
        try:
            v = float(s.get("score"))
        except Exception:
            continue
        if key == "brier" or (score is None and s.get("is_primary")):
            score = v
        elif "ece" in key or "calibration" in key:
            ece = v
    rec = {**meta, "status": d.get("status"), "score": score, "ece": ece, "created": d.get("created_when")}
    res[str(sid)] = rec
    logp = LOGS / f"{sid}.txt"
    if score is not None and not logp.exists():
        try:
            g = json.loads(urllib.request.urlopen(urllib.request.Request(f"https://www.codabench.org/api/submissions/{sid}/get_details/", headers=HDR), timeout=30).read().decode())
        except Exception:
            g = {}
        for lg in g.get("logs") or []:
            if lg.get("name") == "scoring_stdout" or (lg.get("data_file") and "scor" in (lg.get("name") or "")):
                try:
                    txt = urllib.request.urlopen(lg["data_file"], timeout=20).read().decode("utf-8", "replace")
                    if "Brier" in txt:
                        logp.write_text(txt, encoding="utf-8"); break
                except Exception:
                    pass
    time.sleep(0.8)
    if i % 25 == 0:
        json.dump(res, open(OUT, "w"), indent=1)
json.dump(res, open(OUT, "w"), indent=1)
scored = {k: v for k, v in res.items() if v.get("score") is not None}
print(f"known {len(known)} | scored {len(scored)} | pending {sum(1 for v in res.values() if v.get('score') is None)}")
