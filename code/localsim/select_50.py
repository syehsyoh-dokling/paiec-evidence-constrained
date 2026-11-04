"""Select the 50 most upload-worthy candidates from QA results and stage them for upload.

Inputs: one or more qa_summary.json files (from qa_gate.py) with the candidate ZIP folders.
Rule: exactly one control first; then PASS_UPLOAD by most negative LOBO dALC; then HOLD by
dALC; REJECT never. Ties broken by smaller SE. Output: staging folder with upload-ready
names (<= 40 chars) and manifest.json compatible with upload_batch.py.

Usage: python select_50.py --qa <qa_dir>:<zip_dir> [--qa ...] --out quota_50_YYYYMMDD --n 50 --start-version 357
"""
import argparse
import json
import pathlib
import shutil
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from strip_hidden_ids import process as strip_hidden_ids  # comment-only cleanup, AST-verified

ap = argparse.ArgumentParser()
ap.add_argument("--qa", action="append", nargs=2, required=True, metavar=("QA_DIR", "ZIP_DIR"), help="pair: qa results dir, candidate zip dir")
ap.add_argument("--out", type=pathlib.Path, required=True)
ap.add_argument("--n", type=int, default=50)
ap.add_argument("--start-version", type=int, default=357)
a = ap.parse_args()

rows = []
for qa_dir, zip_dir in a.qa:
    qa_dir, zip_dir = pathlib.Path(qa_dir), pathlib.Path(zip_dir)
    if (qa_dir / "qa_summary.json").exists():
        summ = json.load(open(qa_dir / "qa_summary.json"))
    else:  # QA still running: use the per-candidate verdict files written so far
        summ = [json.load(open(p)) for p in sorted(qa_dir.glob("qa_*.json"))]
    for r in summ:
        zp = zip_dir / r["file"]
        if not zp.exists():
            continue
        lobo = r.get("lobo") or {}
        rows.append({"file": str(zp), "name": zp.name, "verdict": r["verdict"], "control": bool(r["info"].get("control")),
                     "regions": r["info"].get("regions_hit"), "dalc": lobo.get("diff_alc"), "se": lobo.get("se_alc"),
                     "wins": lobo.get("wins"), "n": lobo.get("n"), "folds_improved": lobo.get("folds_improved"), "reasons": r.get("reasons")})

controls = [r for r in rows if r["control"] and r["verdict"] == "PASS_UPLOAD"][:1]
passes = sorted([r for r in rows if r["verdict"] == "PASS_UPLOAD" and not r["control"]], key=lambda r: (r["dalc"], r["se"] or 0))
holds = sorted([r for r in rows if r["verdict"] == "HOLD"], key=lambda r: (r["dalc"] if r["dalc"] is not None else 0, r["se"] or 0))
rejects = [r for r in rows if r["verdict"] == "REJECT"]
selected = (controls + passes + holds)[: a.n]

a.out.mkdir(parents=True, exist_ok=True)
manifest = []
v = a.start_version
for r in selected:
    tag = "control" if r["control"] else "_".join(r["regions"] or ["x"]).lower()[:14]
    fname = f"sub_paiec_v{v}_{tag}.zip"
    assert len(fname) <= 40
    shutil.copy(r["file"], a.out / fname)
    strip_res = strip_hidden_ids(a.out / fname, a.out / fname)
    assert not strip_res["leftover_files_with_ids"], strip_res
    manifest.append({"version": f"v{v}", "file": fname, "source": r["name"], "verdict": r["verdict"], "dalc": r["dalc"], "se": r["se"],
                     "wins": r["wins"], "n": r["n"], "folds_improved": r["folds_improved"], "regions": r["regions"], "config": tag})
    v += 1
json.dump(manifest, open(a.out / "manifest.json", "w"), indent=1)
json.dump({"selected": len(selected), "controls": len(controls), "pass": len(passes), "hold": len(holds), "reject": len(rejects),
           "rejected": [{"name": r["name"], "reasons": r["reasons"]} for r in rejects]}, open(a.out / "selection_report.json", "w"), indent=1)
print(f"selected {len(selected)}/{a.n}: control={len(controls)} pass={len(passes)} hold_used={max(0, len(selected) - len(controls) - len(passes))} reject={len(rejects)}")
for m in manifest:
    print(f"{m['version']:5s} {m['verdict']:11s} dALC={m['dalc'] if m['dalc'] is None else round(m['dalc'], 4)!s:8s} SE={m['se']!s:7s} wins={m['wins']}/{m['n']} {m['source']}")
