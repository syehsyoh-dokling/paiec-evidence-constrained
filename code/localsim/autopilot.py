"""Autopilot: unattended evaluation, decision, upload and feedback for a candidate batch.

One command replaces the manual review loop:
  python localsim/autopilot.py <candidate_dir> --name <batch> [--upload] [--max-upload N]
                               [--screen-folds f1 f2 ...] [--screen-pairs 8] [--near -0.0003]

Stages (all deterministic, all resumable; every stage writes JSON the next stage reads):
  A  screening   qa_gate.py on every ZIP without a verdict yet (static gates G1-G7, then LOBO:
                 fast budget-0 evaluation on all pairs for B0-only edits, otherwise a 4-fold x 8-pair
                 paired simulation against v153)
  B  confirm     every candidate whose screening dALC <= --near is re-evaluated on the full protocol
                 (7 folds x 16 pairs); the full-protocol verdict is final
  C  decide      PASS_UPLOAD -> staged (hidden-ID comments stripped, names <= 40 chars, manifest);
                 HOLD / REJECT -> logged with reasons. One control per batch at most.
  D  upload      only with --upload: upload_batch.py on the staged folder, stops at the daily quota
  E  feedback    feedback_for_student.json: per slot, every proposal with its verdict and effect,
                 plus derived rules (parameter spans that proved indistinguishable from v153)
  F  report      AUTOPILOT_REPORT.md with the tables a human would otherwise assemble by hand
Anything the rules cannot settle (tool errors, timeouts) is listed under "needs a human".
"""
import argparse
import json
import pathlib
import shutil
import subprocess
import sys
import time

ROOT = pathlib.Path(r"c:\Users\Saifuddin\Documents\LOMBA")
SIM = ROOT / "localsim"
TEXT_FOLDS = ["afrimedqa", "agentdojo", "ai2d_test", "matharena", "mathvista_mini", "mmbench_v11", "mmlupro"]
sys.path.insert(0, str(SIM))
from strip_hidden_ids import process as strip_ids  # noqa: E402


def run(cmd, log):
    with open(log, "a", encoding="utf-8") as f:
        f.write("\n$ " + " ".join(str(c) for c in cmd) + "\n")
        p = subprocess.run([str(c) for c in cmd], stdout=f, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
    return p.returncode


def load_verdicts(qa_dir):
    out = {}
    for p in sorted(pathlib.Path(qa_dir).glob("qa_*.json")):
        try:
            r = json.load(open(p))
            out[r["file"]] = r
        except Exception:
            pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cand_dir", type=pathlib.Path)
    ap.add_argument("--name", required=True)
    ap.add_argument("--upload", action="store_true")
    ap.add_argument("--max-upload", type=int, default=50)
    ap.add_argument("--screen-folds", nargs="*", default=["afrimedqa", "agentdojo", "matharena", "mmbench_v11"])
    ap.add_argument("--screen-pairs", type=int, default=8)
    ap.add_argument("--near", type=float, default=-0.0003, help="screening dALC at or below which a candidate is confirmed on the full protocol")
    ap.add_argument("--jobs", type=int, default=5)
    ap.add_argument("--start-version", type=int, default=400)
    a = ap.parse_args()

    cand_dir = a.cand_dir.resolve()
    qa_screen = SIM / "results" / f"qa_{a.name}"
    qa_full = SIM / "results" / f"qa_{a.name}_full"
    staged = ROOT / f"quota_{a.name}"
    qa_screen.mkdir(parents=True, exist_ok=True); qa_full.mkdir(parents=True, exist_ok=True)
    log = qa_screen / "autopilot.log"
    t0 = time.time()
    zips = sorted(cand_dir.glob("*.zip"))
    needs_human = []

    # ---- A: screening (skip ZIPs that already have a verdict)
    done = load_verdicts(qa_screen)
    todo = [z for z in zips if z.name not in done]
    if todo:
        tmp = qa_screen / "_todo"
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir()
        for z in todo:
            shutil.copy(z, tmp / z.name)
        tag = "full_v153_vs_baselines" if a.screen_pairs == 16 else f"screen_p{a.screen_pairs}"
        run([sys.executable, SIM / "qa_gate.py", tmp, "--out", qa_screen, "--pairs", a.screen_pairs, "--folds", *a.screen_folds, "--tag", tag, "--jobs", a.jobs], log)
        shutil.rmtree(tmp, ignore_errors=True)
    screen = load_verdicts(qa_screen)

    # ---- B: confirmation on the full protocol
    # budget-0-only edits are already evaluated on every eligible pair of all 7 folds (fast path),
    # so only edits in other regions need the full-protocol confirmation
    confirm = [z for z in zips if (r := screen.get(z.name)) and not r["info"].get("control")
               and r["verdict"] != "REJECT" and r.get("lobo") and r["lobo"]["diff_alc"] <= a.near
               and r["info"].get("regions_hit") != ["B0"]]
    full_done = load_verdicts(qa_full)
    for z in confirm:
        if z.name in full_done:
            continue
        run([sys.executable, SIM / "qa_gate.py", z, "--out", qa_full, "--pairs", 16, "--folds", *TEXT_FOLDS, "--tag", "full_v153_vs_baselines", "--jobs", a.jobs], log)
    full = load_verdicts(qa_full)

    # ---- C: decide
    final = {}
    for z in zips:
        r = screen.get(z.name)
        if r is None:
            needs_human.append(f"{z.name}: no screening verdict (tool error?)"); continue
        if any("LOBO error" in x for x in r.get("reasons", [])):
            needs_human.append(f"{z.name}: {r['reasons'][0][:120]}")
        f = full.get(z.name)
        final[z.name] = {"stage": "full" if f else "screen", "verdict": (f or r)["verdict"], "lobo": (f or r).get("lobo"), "reasons": (f or r).get("reasons"),
                         "control": bool(r["info"].get("control")), "regions": r["info"].get("regions_hit"), "screen_lobo": r.get("lobo")}
    passes = sorted([n for n, v in final.items() if v["verdict"] == "PASS_UPLOAD" and not v["control"]], key=lambda n: final[n]["lobo"]["diff_alc"])
    controls = [n for n, v in final.items() if v["control"] and v["verdict"] == "PASS_UPLOAD"][:1]
    staged.mkdir(exist_ok=True)
    manifest_path = staged / "manifest.json"
    manifest = json.load(open(manifest_path)) if manifest_path.exists() else []
    already = {m["source"] for m in manifest}
    v = a.start_version + len(manifest)
    for n in controls + passes:
        if n in already:
            continue
        tag = "control" if final[n]["control"] else "_".join(final[n]["regions"] or ["x"]).lower()[:14]
        fname = f"sub_paiec_v{v}_{tag}.zip"
        shutil.copy(cand_dir / n, staged / fname)
        res = strip_ids(staged / fname, staged / fname)
        assert not res["leftover_files_with_ids"], res
        lobo = final[n]["lobo"] or {}
        manifest.append({"version": f"v{v}", "file": fname, "source": n, "verdict": final[n]["verdict"], "stage": final[n]["stage"],
                         "dalc": lobo.get("diff_alc"), "se": lobo.get("se_alc"), "folds_improved": lobo.get("folds_improved"), "config": tag})
        v += 1
    json.dump(manifest, open(manifest_path, "w"), indent=1)

    # ---- D: upload
    upload_note = "not requested"
    if a.upload and manifest:
        rc = run([sys.executable, SIM / "upload_batch.py", staged, "--max", a.max_upload], log)
        upload_note = {0: "all staged candidates submitted", 2: "stopped at the daily quota"}.get(rc, f"upload script exit {rc}")

    # ---- E: feedback for the student
    feedback = {"batch": a.name, "rules": [], "by_slot": {}}
    for z in zips:
        meta_p = cand_dir / (z.stem.split("_")[0] + "_" + z.stem.split("_")[1] + ".meta.json")
        meta = json.load(open(meta_p)) if meta_p.exists() else {}
        slot = meta.get("slot") or (final[z.name]["regions"] or ["?"])[0]
        lobo = final[z.name]["lobo"] or {}
        feedback["by_slot"].setdefault(slot, []).append({"params": meta.get("params"), "verdict": final[z.name]["verdict"], "dalc": lobo.get("diff_alc"),
                                                         "reason": (final[z.name]["reasons"] or [None])[0]})
    for slot, items in feedback["by_slot"].items():
        flat = [i for i in items if i["dalc"] is not None]
        if flat and all(abs(i["dalc"]) < 0.0002 for i in flat):
            feedback["rules"].append(f"{slot}: every proposal so far is indistinguishable from v153 (|dALC| < 0.0002); propose values near the ends of the ranges, not near the baseline")
        goods = [i for i in flat if i["verdict"] == "PASS_UPLOAD"]
        if goods:
            feedback["rules"].append(f"{slot}: {len(goods)} proposals passed; explore around them: " + "; ".join(json.dumps(g["params"]) for g in goods[:3]))
    json.dump(feedback, open(qa_screen / "feedback_for_student.json", "w"), indent=1)

    # ---- F: report
    counts = {k: sum(1 for x in final.values() if x["verdict"] == k) for k in ("PASS_UPLOAD", "HOLD", "REJECT")}
    lines = [f"# Autopilot report: {a.name}", "", f"Candidates: {len(zips)} | PASS {counts['PASS_UPLOAD']} | HOLD {counts['HOLD']} | REJECT {counts['REJECT']} | confirmed on full protocol: {len(full)} | elapsed {int(time.time() - t0)} s",
             f"Staged for upload: {len(manifest)} in `{staged.name}` | upload: {upload_note}", "",
             "| candidate | stage | verdict | dALC | SE | folds+ | reason |", "|---|---|---|---|---|---|---|"]
    for n, x in sorted(final.items(), key=lambda kv: (kv[1]["verdict"] != "PASS_UPLOAD", (kv[1]["lobo"] or {}).get("diff_alc", 0))):
        lobo = x["lobo"] or {}
        d = f"{lobo['diff_alc']:+.4f}" if lobo.get("diff_alc") is not None else ""
        s = f"{lobo['se_alc']:.4f}" if lobo.get("se_alc") is not None else ""
        lines.append(f"| {n} | {x['stage']} | {x['verdict']} | {d} | {s} | {lobo.get('folds_improved', '')} | {((x['reasons'] or [''])[0] or '')[:70]} |")
    lines += ["", "## Rules learned for the student", *[f"- {r}" for r in feedback["rules"]], "", "## Needs a human", *([f"- {h}" for h in needs_human] or ["- nothing"])]
    (qa_screen / "AUTOPILOT_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:4]))
    print(f"report: {qa_screen / 'AUTOPILOT_REPORT.md'}")


if __name__ == "__main__":
    main()
