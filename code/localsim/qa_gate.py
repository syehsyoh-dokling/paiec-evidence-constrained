"""QA gate for agent-generated PAIEC candidates. Reviewer tool: it never edits candidates.

Static gates (seconds):
  G1 compiles; G2 ZIP entry set identical to v153; G3 no api_keys / secret-looking strings /
  new network imports; G4 every changed line of model.py lies inside an allowed edit region;
  G5 filename <= 40 chars, SHA-256 unique within the batch and vs upload ledgers;
  G6 no hidden benchmark/subject IDs added; G7 predict() contract smoke (0 and 1 labels).
LOBO gate (minutes): B0-only diffs -> fast budget-0 eval on all pairs of each fold;
  otherwise -> full-curve paired simulation vs cached v153 (same 112 pairs).
Verdict: PASS_UPLOAD / HOLD / REJECT with reasons, written to <out>/qa_<name>.json.

Usage: python qa_gate.py <zip_or_dir> [--out DIR] [--folds f1 f2 ...] [--no-lobo] [--jobs N]
"""
import argparse
import difflib
import hashlib
import json
import os
import pathlib
import py_compile
import re
import subprocess
import sys
import tempfile
import zipfile

import numpy as np

ROOT = pathlib.Path(os.environ.get("PAIEC_ROOT", r"c:\Users\Saifuddin\Documents\LOMBA"))
SIM = ROOT / "localsim"
sys.path.insert(0, str(SIM))
from strip_hidden_ids import clean_model  # normalises the hidden-ID comment lines before diffing
BASE_ZIP = SIM / "variants" / "v153_nokeys.zip"
TEXT_FOLDS = ["afrimedqa", "agentdojo", "ai2d_test", "matharena", "mathvista_mini", "mmbench_v11", "mmlupro"]
LEDGERS = sorted(ROOT.glob("quota_*/upload_ledger.json")) + sorted(ROOT.glob("pending_*/upload_ledger.json")) + [ROOT / "auto_discovery_paiec" / "online_meta_dataset.csv"]

# Allowed edit regions in v153 model.py, located by anchor text (start anchor, end anchor inclusive)
ALLOWED_REGIONS = {
    "B0": ("    if not labeled:\n        base_b0 = p0", "return float(min(max(base_b0 + 0.238 * d_target_2pl, 0.005), 0.495))"),
    "MIXTURE_PRIOR": ("def _mixture_bayes_posterior(", "return float(sum(p * m for p, m in zip(probs, post_means)) / max(Z, 1e-12))"),
    "LOW_BUDGET_SHRINK": ("    if count > 0:\n        mean_sampled_d", "        p_base = float(min(max(p_base, 0.005), 0.995))"),
    "PEER_SHIFT": ("        if total_peer_n > 0:\n            peer_bm_mean = sum(peer_ests) / total_peer_n", "            )"),
    "IRT_ENVELOPE": ("    # Boundary damping envelope", "    p_final = p_base + env * 0.32 * d_rel_2pl"),
    # the IRT weight also appears inside the kernel-ridge block and its fallback; the template edits all three
    "IRT_ENVELOPE_KR": ("                    w_kr = min(n_tot / (n_tot + 4.5), 0.65)", "            p_final = p_base + env * 0.32 * d_rel_2pl"),
}
SECRET_RE = re.compile(r"(nvapi-[A-Za-z0-9_-]{20,}|sk-or-v1-[0-9a-f]{20,}|eyJ[A-Za-z0-9_-]{30,}\.[A-Za-z0-9_-]{20,}|AKIA[0-9A-Z]{16}|hf_[A-Za-z0-9]{30,})")
HIDDEN_ID_RE = re.compile(r"\b(benchmark|subject)_\d{6}\b")
NET_RE = re.compile(r"^\s*(import|from)\s+(requests|socket|urllib|http\.client|aiohttp|httpx)\b", re.M)


def read_zip(path):
    with zipfile.ZipFile(path) as z:
        return {n: z.read(n) for n in z.namelist()}


def region_ranges(base_text):
    ranges = {}
    for name, (start, end) in ALLOWED_REGIONS.items():
        i = base_text.find(start)
        j = base_text.find(end, i) if i >= 0 else -1
        if i < 0 or j < 0:
            continue
        s_line = base_text.count("\n", 0, i) + 1
        e_line = base_text.count("\n", 0, j + len(end)) + 1
        ranges[name] = (s_line, e_line)
    return ranges


def changed_base_lines(base_text, cand_text):
    """Return (set of base line numbers touched, list of added lines) from a unified diff."""
    base_lines, cand_lines = base_text.splitlines(), cand_text.splitlines()
    touched, added = set(), []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, base_lines, cand_lines).get_opcodes():
        if tag == "equal":
            continue
        # an insertion at i1 touches the line before it
        touched.update(range(max(i1, 1), max(i2, i1 + 1) + 1))
        added.extend(cand_lines[j1:j2])
    return touched, added


def smoke_predict(zpath):
    code = (
        "import os,sys,json,importlib.util;os.environ['PAIEC_DISABLE_BLE_NETWORK']='1';os.environ['PREDICTIVE_EVAL_LOCAL_SMOKE_TEST']='1';"
        "d=sys.argv[1];sys.path.insert(0,d);os.chdir(d);sp=importlib.util.spec_from_file_location('model',os.path.join(d,'model.py'));"
        "m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);"
        "S={'normalized_name':'gpt-4o','provider':'','release_date':'','access_date':'','harness':'','harness_version':'','reasoning_effort':'','subject_features_extra':''};"
        "U={'normalized_name':'zz-unknown-model-9000-xl','provider':'','release_date':'','access_date':'','harness':'h','harness_version':'1','reasoning_effort':'high','subject_features_extra':''};"
        "I={'item_content':'What is 2+2? A: 3 B: 4 C: 5 D: 6','item_features':'','interactors':'','benchmark_id':'bm_local_smoke'};"
        "J={'item_content':'Name the capital of France.','item_features':'','interactors':'','benchmark_id':'bm_local_smoke'};"
        "K={'item_content':'Compute the derivative of x^3.','item_features':'','interactors':'','benchmark_id':'bm_other'};"
        "outs=[];"
        "[outs.extend([m.predict([X,I],[]),m.predict([X,I],[[[X,J],1]]),m.predict([X,I],[[[X,J],0],[[X,K],1],[[S,J],0]])]) for X in (S,U)];"
        "assert all(0.0<=float(o)<=1.0 for o in outs),outs;print(json.dumps([float(o) for o in outs]))"
    )
    with tempfile.TemporaryDirectory() as td:
        with zipfile.ZipFile(zpath) as z:
            z.extractall(td)
        p = subprocess.run([sys.executable, "-c", code, td], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=600)
    return p.returncode == 0, (p.stdout.strip() or p.stderr[-800:])


def known_shas():
    shas = set()
    for p in LEDGERS:
        if not p.exists():
            continue
        txt = p.read_text(encoding="utf-8", errors="replace")
        shas.update(re.findall(r"\b[0-9a-f]{64}\b", txt))
    return shas


def static_gates(zpath, base, batch_shas, allow_unchanged=False):
    """allow_unchanged: the reference is not v153 (a data-only revision such as v403), so an
    unchanged model.py is expected rather than a sign of a mislabelled control."""
    reasons, info = [], {}
    cand = read_zip(zpath)
    if len(zpath.name) > 40:
        reasons.append("G5 filename > 40 chars")
    sha = hashlib.sha256(zpath.read_bytes()).hexdigest()
    info["sha256"] = sha
    if sha in batch_shas:
        reasons.append("G5 duplicate SHA-256 (batch or ledger)")
    if set(cand) != set(base):
        reasons.append(f"G2 zip entries differ: +{sorted(set(cand) - set(base))[:3]} -{sorted(set(base) - set(cand))[:3]}")
    for n, data in cand.items():
        if n.endswith((".py", ".json", ".txt", ".md")) and SECRET_RE.search(data.decode("utf-8", "replace")):
            reasons.append(f"G3 secret-looking string in {n}")
    try:
        cfg = json.loads(cand.get("submission_config.json", b"{}"))
        if cfg.get("api_keys"):
            reasons.append("G3 api_keys present in submission_config.json")
    except Exception:
        reasons.append("G3 submission_config.json unreadable")
    model = clean_model(cand.get("model.py", b"").decode("utf-8", "replace"))
    base_model = clean_model(base["model.py"].decode("utf-8"))
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(model); tmp = f.name
    try:
        py_compile.compile(tmp, doraise=True)
    except Exception as e:
        reasons.append(f"G1 compile error: {str(e)[:200]}")
    finally:
        os.unlink(tmp)
    for n in cand:
        if n.endswith(".py") and n != "model.py" and cand[n] != base.get(n):
            reasons.append(f"G4 non-model file changed: {n}")
    touched, added = changed_base_lines(base_model, model)
    ranges = region_ranges(base_model)
    outside = [ln for ln in touched if not any(s <= ln <= e for s, e in ranges.values())]
    regions_hit = sorted({name for name, (s, e) in ranges.items() if any(s <= ln <= e for ln in touched)})
    info["regions_hit"] = regions_hit
    info["lines_touched"] = len(touched)
    if outside:
        reasons.append(f"G4 edits outside allowed regions at base lines {sorted(outside)[:6]}")
    is_control = "control" in zpath.stem.lower()
    info["control"] = is_control
    if not touched and not is_control and not allow_unchanged:
        reasons.append("G4 model.py identical to v153 (no change); name it *_control.zip if it is the daily control")
    if is_control and touched:
        reasons.append("G4 control ZIP must be byte-identical to v153 in model.py")
    if NET_RE.search("\n".join(added)):
        reasons.append("G3 network import added")
    if HIDDEN_ID_RE.search("\n".join(added)):
        reasons.append("G6 hidden benchmark/subject ID added")
    if "def predict(" in "\n".join(added):
        reasons.append("G4 predict signature touched")
    return reasons, info


def lobo_b0(zpath, folds, out_dir, jobs=4):
    """Fast budget-0 paired eval (candidate vs cached v153 per pair); folds run in parallel."""
    from concurrent.futures import ThreadPoolExecutor
    diffs, per_fold = [], {}

    def run(bm):
        out = out_dir / f"b0_{zpath.stem}_{bm}.json"
        try:
            p = subprocess.run([sys.executable, str(SIM / "b0_eval_fold.py"), bm, str(out), str(zpath)],
                               capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=900)
        except subprocess.TimeoutExpired:
            p = subprocess.CompletedProcess(args=[], returncode=124, stdout="", stderr="budget-0 evaluation exceeded 900 s (candidate too slow or hung)")
        return bm, out, p

    for bm in folds:
        if not (SIM / "results" / "b0eval" / f"{bm}.json").exists():
            return None, f"missing baseline b0eval for {bm}"
    with ThreadPoolExecutor(max(1, jobs)) as ex:
        results = list(ex.map(run, folds))
    for bm, out, p in results:
        if p.returncode != 0:
            return None, f"b0 eval failed on {bm}: {p.stderr[-400:]}"
        cand = json.load(open(out)); base = json.load(open(SIM / "results" / "b0eval" / f"{bm}.json"))
        c = np.array(cand["configs"]["v153"]["per_pair"]); b = np.array(base["configs"]["v153"]["per_pair"])
        if len(c) != len(b):
            return None, f"pair count mismatch on {bm}"
        d = c - b
        diffs.append(d); per_fold[bm] = float(d.mean())
    d = np.concatenate(diffs)
    # B0 carries weight 0.1 in ALC
    return {"metric": "B0", "diff_alc": 0.1 * float(d.mean()), "se_alc": 0.1 * float(d.std(ddof=1) / np.sqrt(len(d))),
            "wins": int((d < 0).sum()), "n": int(len(d)), "folds_improved": sum(v < 0 for v in per_fold.values()),
            "per_fold": per_fold}, None


def lobo_full(zpath, folds, jobs, pairs=16, tag="full_v153_vs_baselines"):
    cmd = [sys.executable, str(SIM / "sim_local.py"), "--zips", str(BASE_ZIP), str(zpath), "--pairs-per-fold", str(pairs),
           "--folds", *folds, "--tag", tag, "--jobs", str(jobs)]
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=4 * 3600)
    if p.returncode != 0:
        return None, f"sim failed: {p.stderr[-400:]}"
    res = json.load(open(SIM / "results" / tag / "combined.json"))
    b, c = res[BASE_ZIP.stem], res[zpath.stem]
    common = sorted(set(b) & set(c))
    d = np.array([c[k]["alc"] - b[k]["alc"] for k in common])
    per_fold = {f: float(np.mean([c[k]["alc"] - b[k]["alc"] for k in common if k.startswith(f + "|")])) for f in folds}
    bud = np.mean([np.array(c[k]["briers"]) - np.array(b[k]["briers"]) for k in common], axis=0)
    return {"metric": "ALC", "diff_alc": float(d.mean()), "se_alc": float(d.std(ddof=1) / np.sqrt(len(d))),
            "wins": int((d < 0).sum()), "n": int(len(d)), "folds_improved": sum(v < 0 for v in per_fold.values()),
            "per_fold": per_fold, "budget_diffs": [float(x) for x in bud]}, None


def verdict(reasons, lobo, n_folds):
    if reasons:
        return "REJECT"
    if lobo is None:
        return "HOLD"
    pf = list((lobo.get("per_fold") or {}).values())
    if len(pf) > 1:
        import statistics
        se_f = statistics.stdev(pf) / len(pf) ** 0.5 * (0.1 if lobo.get("metric") == "B0" else 1.0)
        lobo["se_fold"] = se_f
        lobo["fold_significant"] = bool(lobo["diff_alc"] + 2 * se_f < 0)
    good = lobo["diff_alc"] <= -0.0005 and lobo["diff_alc"] + 2 * lobo["se_alc"] < 0 and lobo["folds_improved"] >= max(1, (n_folds + 1) // 2)
    # REJECT only for a material, significant degradation; a +0.00001 with a tiny SE is "no effect", i.e. HOLD
    bad = lobo["diff_alc"] - 2 * lobo["se_alc"] > 0 and lobo["diff_alc"] >= 0.0002
    return "PASS_UPLOAD" if good else ("REJECT" if bad else "HOLD")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("target")
    ap.add_argument("--out", default=str(SIM / "results" / "qa"))
    ap.add_argument("--folds", nargs="*", default=TEXT_FOLDS)
    ap.add_argument("--no-lobo", action="store_true")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--pairs", type=int, default=16, help="pairs per fold for the full-curve sim (8 = screening)")
    ap.add_argument("--tag", default=None, help="sim result tag; default full_v153_vs_baselines for 16 pairs, screen_p<N> otherwise")
    ap.add_argument("--base", default=str(BASE_ZIP), help="reference ZIP for G2/G4 (default v153); use the parent revision for data-only revisions")
    a = ap.parse_args()
    base_path = pathlib.Path(a.base).resolve()
    allow_unchanged = base_path != BASE_ZIP.resolve()
    sim_tag = a.tag or ("full_v153_vs_baselines" if a.pairs == 16 else f"screen_p{a.pairs}")
    out_dir = pathlib.Path(a.out).resolve(); out_dir.mkdir(parents=True, exist_ok=True)
    target = pathlib.Path(a.target).resolve()
    zips = sorted(target.glob("*.zip")) if target.is_dir() else [target]
    base = read_zip(base_path)
    seen = known_shas()
    summary = []
    n_controls = 0
    for zp in zips:
        reasons, info = static_gates(zp, base, seen, allow_unchanged=allow_unchanged)
        seen.add(info["sha256"])
        if info.get("control"):
            n_controls += 1
            if n_controls > 1:
                reasons.append("G5 more than one control in this batch")
            if not reasons:
                rec = {"file": zp.name, "verdict": "PASS_UPLOAD", "reasons": ["control: byte-identical v153, exempt from LOBO"], "info": info, "smoke": "skipped", "lobo": None}
                json.dump(rec, open(out_dir / f"qa_{zp.stem}.json", "w"), indent=1); summary.append(rec)
                print(f"{'PASS_UPLOAD':12s} {zp.name:42s} control", flush=True)
                continue
        smoke_ok, smoke_out = (True, "skipped") if reasons else smoke_predict(zp)
        if not smoke_ok:
            reasons.append(f"G7 predict smoke failed: {smoke_out[:200]}")
        lobo, lobo_err = None, None
        if not reasons and not a.no_lobo:
            if set(info["regions_hit"]) <= {"IRT_ENVELOPE", "IRT_ENVELOPE_KR"}:
                info["regions_hit"] = ["IRT_ENVELOPE"]
            if info["regions_hit"] == ["B0"]:
                # budget-0 edits are cheap to evaluate: always use all text folds (905 pairs),
                # whatever screening folds were requested for the full-curve simulations
                lobo, lobo_err = lobo_b0(zp, TEXT_FOLDS, out_dir, a.jobs)
            else:
                lobo, lobo_err = lobo_full(zp, a.folds, a.jobs, a.pairs, sim_tag)
            if lobo_err:
                reasons.append(f"LOBO error: {lobo_err}")
        v = verdict(reasons, lobo, len(a.folds))
        rec = {"file": zp.name, "verdict": v, "reasons": reasons, "info": info, "smoke": smoke_out, "lobo": lobo}
        json.dump(rec, open(out_dir / f"qa_{zp.stem}.json", "w"), indent=1)
        summary.append(rec)
        lob = f" dALC={lobo['diff_alc']:+.4f} SE={lobo['se_alc']:.4f} wins={lobo['wins']}/{lobo['n']} folds+={lobo['folds_improved']}" if lobo else ""
        print(f"{v:12s} {zp.name:42s} regions={info.get('regions_hit')}{lob} {('; '.join(reasons))[:160]}", flush=True)
    json.dump(summary, open(out_dir / "qa_summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
