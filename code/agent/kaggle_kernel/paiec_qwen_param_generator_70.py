#!/usr/bin/env python3
"""PAIEC parameter-proposal Qwen generator (Kaggle T4), version 2.

Division of labour that makes every packaged candidate upload-ready:
  * the student (Qwen2.5-Coder, local on the GPU) proposes ONLY numbers: a slot and a
    JSON of parameter values inside published ranges, plus a one-sentence hypothesis;
  * the harness renders the edit from tested templates (None-safe budget-0 branch,
    exact constant substitutions for the other slots), compiles it, runs predict() with a
    KNOWN and an UNKNOWN subject at 0/1/2 labels, and packages the ZIP without api_keys;
  * candidates must differ from every earlier candidate by >= 1% of a parameter range.
Whether a candidate improves the score is decided afterwards by the LOBO QA gate.
"""
import copy
import hashlib
import importlib.util
import json
import os
import py_compile
import re
import shutil
import sys
import tempfile
import time
import zipfile
from pathlib import Path

START = time.time()
MAX_RUNTIME = int(os.environ.get("MAX_RUNTIME_SECONDS", str(50 * 60)))
N_CANDIDATES = int(os.environ.get("N_CANDIDATES", "69"))
WORK = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(".")
OUT = WORK / "qwen_batch"
OUT.mkdir(parents=True, exist_ok=True)
print("PAIEC PARAM-PROPOSAL QWEN GENERATOR v2 | start", time.strftime("%Y-%m-%d %H:%M:%S"), flush=True)

base_dir = None
for cb in [Path("/kaggle/input/paiec-baseline-and-knowledge/paiec_v153_baseline"), Path("/kaggle/input/paiec-baseline-and-knowledge"), Path("paiec_v153_baseline")]:
    if (cb / "model.py").exists():
        base_dir = cb; break
if base_dir is None:
    for p in Path("/kaggle/input").rglob("model.py"):
        if (p.parent / "labeling.py").exists():
            base_dir = p.parent; break
assert base_dir is not None, "baseline not found"
BASE = (base_dir / "model.py").read_text(encoding="utf-8")
BASE_SHA = hashlib.sha256(BASE.encode()).hexdigest()

# ------------------------------------------------------------------ slots and templates
SLOTS = {
    "B0": {"params": {"center": [0.48, 0.56], "shrink": [0.0, 1.0], "ease": [0.0, 1.2]},
           "baseline": {"center": 0.449, "shrink": 0.20, "ease": 0.238},
           "evidence": "LOBO: budget-0 Brier 0.2559 worse than constant 0.5; center 0.50/shrink 0.5/ease 1.0 gives 0.2482"},
    "LOW_BUDGET_SHRINK": {"params": {"blend_count1": [0.20, 0.60], "blend_count_le3": [0.05, 0.35], "max_count": [3, 7], "debias": [0.20, 0.32]},
                          "baseline": {"blend_count1": 0.340, "blend_count_le3": 0.100, "max_count": 3, "debias": 0.265},
                          "evidence": "LOBO: Brier at 1 label (0.2497) ~ 0 labels (0.2559); budgets 1 and 3 carry 40% of the metric"},
    "MIXTURE_PRIOR": {"params": {"a1": [0.3, 8.0], "b1": [0.3, 8.0], "a2": [0.3, 8.0], "b2": [0.3, 8.0], "a3": [0.3, 8.0], "b3": [0.3, 8.0], "w1": [0.10, 0.40], "w3": [0.10, 0.40]},
                      "baseline": {"a1": 0.38, "b1": 6.80, "a2": 3.10, "b2": 5.10, "a3": 6.20, "b3": 2.80, "w1": 0.24, "w3": 0.26},
                      "evidence": "pseudo-counts of the 3 Beta regimes control over-reaction at 1-3 labels"},
    "PEER_SHIFT": {"params": {"gain": [0.25, 0.70], "center": [0.35, 0.45], "shrink_n": [1.0, 6.0]},
                   "baseline": {"gain": 0.45, "center": 0.386, "shrink_n": 2.0},
                   "evidence": "ablation: peer pooling is the largest component (+0.0045 ALC)"},
    "IRT_ENVELOPE": {"params": {"env_scale": [0.04, 0.12], "env_floor": [0.05, 0.40], "irt_weight": [0.0, 0.50]},
                     "baseline": {"env_scale": 0.06, "env_floor": 0.15, "irt_weight": 0.32},
                     "evidence": "ablation: envelope term worth +0.0006 ALC; B31 equals the pair-constant oracle"},
}
ALLOCATION = [("B0", 15), ("LOW_BUDGET_SHRINK", 21), ("MIXTURE_PRIOR", 13), ("PEER_SHIFT", 10), ("IRT_ENVELOPE", 10)]
# Feedback from the previous batch's autopilot (rules + judged proposals), injected at push time.
FEEDBACK = json.loads(r'''__FEEDBACK_JSON__''') if "__FEEDBACK_JSON__"[:2] != "__" else {"rules": [], "by_slot": {}}
TIERS = ["r1 (+-25% of each range around baseline)", "r2 (+-50% of range)", "r3 (anywhere in range)"]

OLD_B0 = '''    if not labeled:
        base_b0 = p0
        features_str = str(item.get("item_features") or "").lower()
        if any(
            k in features_str
            for k in ("hard", "tier=frontier", "difficult", "level=5", "olympiad")
        ):
            base_b0 = float(max(base_b0 - 0.038, 0.395))
        return float(min(max(base_b0 + 0.238 * d_target_2pl, 0.005), 0.495))
'''
NEW_B0 = '''    if not labeled:
        # student-proposed budget-0 prior (None-safe for subjects absent from the priors)
        _info = _lookup_db_subject_info(subject)
        _m = float(_info.get("mean", 0.50)) if _info else 0.50
        base_b0 = {center:.4f} + {shrink:.4f} * (_m - 0.50)
        return float(min(max(base_b0 + {ease:.4f} * d_target_2pl, 0.02), 0.98))
'''


def rep(t, old, new):
    n = t.count(old)
    if n != 1:
        raise ValueError(f"template anchor not unique ({n}): {old[:60]!r}")
    return t.replace(old, new, 1)


def render(slot, p):
    t = BASE
    if slot == "B0":
        return rep(t, OLD_B0, NEW_B0.format(**p))
    if slot == "LOW_BUDGET_SHRINK":
        b1, b3, k, db = p["blend_count1"], p["blend_count_le3"], int(round(p["max_count"])), p["debias"]
        t = rep(t, "(1.0 - 0.340)", f"(1.0 - {b1:.4f})"); t = rep(t, ") * p_base + 0.340 * (0.50", f") * p_base + {b1:.4f} * (0.50")
        t = rep(t, "(1.0 - 0.100)", f"(1.0 - {b3:.4f})"); t = rep(t, ") * p_base + 0.100 * (0.50", f") * p_base + {b3:.4f} * (0.50")
        t = rep(t, "            elif count <= 3:\n                p_base = (1.0 - ", f"            elif count <= {k}:\n                p_base = (1.0 - ")
        return rep(t, "0.265 * count * mean_sampled_d", f"{db:.4f} * count * mean_sampled_d")
    if slot == "MIXTURE_PRIOR":
        t = rep(t, "(w1, 0.38, 6.80)", f"(w1, {p['a1']:.3f}, {p['b1']:.3f})")
        t = rep(t, "(w2, 3.10, 5.10)", f"(w2, {p['a2']:.3f}, {p['b2']:.3f})")
        t = rep(t, "(w3, 6.20, 2.80)", f"(w3, {p['a3']:.3f}, {p['b3']:.3f})")
        t = rep(t, "w1 = max(0.04, 0.24 - 0.85 * shift)", f"w1 = max(0.04, {p['w1']:.3f} - 0.85 * shift)")
        return rep(t, "w3 = max(0.04, 0.26 + 0.85 * shift)", f"w3 = max(0.04, {p['w3']:.3f} + 0.85 * shift)")
    if slot == "PEER_SHIFT":
        return rep(t, "                0.45\n                * (peer_bm_mean - 0.386)\n                * (total_peer_n / (total_peer_n + 2.0))",
                   f"                {p['gain']:.4f}\n                * (peer_bm_mean - {p['center']:.4f})\n                * (total_peer_n / (total_peer_n + {p['shrink_n']:.3f}))")
    if slot == "IRT_ENVELOPE":
        w = p["irt_weight"]
        t = rep(t, "env = float(min(max(p_base * (1.0 - p_base) / 0.06, 0.15), 1.0))",
                f"env = float(min(max(p_base * (1.0 - p_base) / {p['env_scale']:.4f}, {p['env_floor']:.4f}), 1.0))")
        t = rep(t, "\n    p_final = p_base + env * 0.32 * d_rel_2pl", f"\n    p_final = p_base + env * {w:.4f} * d_rel_2pl")
        t = rep(t, "\n            p_final = p_base + env * 0.32 * d_rel_2pl", f"\n            p_final = p_base + env * {w:.4f} * d_rel_2pl")
        return rep(t, "+ env * (0.32 * d_rel_2pl + w_kr * delta_kr)", f"+ env * ({w:.4f} * d_rel_2pl + w_kr * delta_kr)")
    raise ValueError(slot)


# self-test: every template renders at its baseline values
for s, spec in SLOTS.items():
    render(s, spec["baseline"])
print("templates OK", flush=True)


def validate_params(slot, params, used):
    spec = SLOTS[slot]["params"]
    if not isinstance(params, dict):
        return "params must be an object"
    missing = [k for k in spec if k not in params]
    if missing:
        return f"missing params {missing}"
    clean = {}
    for k, (lo, hi) in spec.items():
        try:
            v = float(params[k])
        except Exception:
            return f"param {k} is not a number"
        if not (lo - 1e-9 <= v <= hi + 1e-9):
            return f"param {k}={v} outside range [{lo}, {hi}]"
        clean[k] = v
    for prev in used:
        if all(abs(clean[k] - prev[k]) < 0.01 * (spec[k][1] - spec[k][0]) for k in spec):
            return f"too close to an existing candidate {prev}; move at least one parameter by >= 1% of its range"
    return clean


SUBJ_KNOWN = {"normalized_name": "gpt-4o", "provider": "", "release_date": "", "access_date": "", "harness": "", "harness_version": "", "reasoning_effort": "", "subject_features_extra": ""}
SUBJ_UNKNOWN = {"normalized_name": "zz-unknown-model-9000-xl", "provider": "", "release_date": "", "access_date": "", "harness": "h", "harness_version": "1", "reasoning_effort": "high", "subject_features_extra": ""}
ITEM_A = {"item_content": "What is 2+2? A: 3 B: 4 C: 5 D: 6", "item_features": "", "interactors": "", "benchmark_id": "bm_smoke"}
ITEM_B = {"item_content": "Name the capital of France.", "item_features": "", "interactors": "", "benchmark_id": "bm_smoke"}
ITEM_C = {"item_content": "Compute the derivative of x^3.", "item_features": "", "interactors": "", "benchmark_id": "bm_other"}


def smoke(model_text):
    d = Path(tempfile.mkdtemp(prefix="smoke_", dir=str(WORK)))
    try:
        for root, dirs, files in os.walk(base_dir):
            if "__pycache__" in root:
                continue
            for fn in files:
                if fn.endswith(".pyc"):
                    continue
                full = Path(root) / fn; rel = full.relative_to(base_dir)
                (d / rel).parent.mkdir(parents=True, exist_ok=True)
                if rel.as_posix() == "model.py":
                    (d / rel).write_text(model_text, encoding="utf-8")
                else:
                    shutil.copy(full, d / rel)
        os.environ["PAIEC_DISABLE_BLE_NETWORK"] = "1"; os.environ["PREDICTIVE_EVAL_LOCAL_SMOKE_TEST"] = "1"
        sys.path.insert(0, str(d)); cwd = os.getcwd(); os.chdir(d)
        try:
            spec = importlib.util.spec_from_file_location(f"m_{d.name}", str(d / "model.py"))
            m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
            outs = []
            for S in (SUBJ_KNOWN, SUBJ_UNKNOWN):
                lab1 = [[[copy.deepcopy(S), copy.deepcopy(ITEM_B)], 1]]
                lab2 = lab1 + [[[copy.deepcopy(S), copy.deepcopy(ITEM_C)], 0], [[copy.deepcopy(SUBJ_KNOWN), copy.deepcopy(ITEM_B)], 0]]
                for lab in ([], lab1, lab2):
                    v = float(m.predict([copy.deepcopy(S), copy.deepcopy(ITEM_A)], copy.deepcopy(lab)))
                    if not (0.0 <= v <= 1.0) or v != v:
                        return False, f"prediction out of range: {v}"
                    outs.append(round(v, 4))
            return True, outs
        finally:
            os.chdir(cwd); sys.path.remove(str(d))
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"
    finally:
        shutil.rmtree(d, ignore_errors=True)


def build_zip(out, model_text):
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(base_dir):
            if "__pycache__" in root:
                continue
            for fn in files:
                if fn.endswith(".pyc"):
                    continue
                full = Path(root) / fn; rel = full.relative_to(base_dir).as_posix()
                if rel == "model.py":
                    zf.writestr(rel, (model_text if model_text is not None else BASE).encode("utf-8"))
                elif rel == "submission_config.json":
                    cfg = json.loads(full.read_text(encoding="utf-8")); cfg["api_keys"] = {}
                    zf.writestr(rel, json.dumps(cfg, indent=2).encode("utf-8"))
                else:
                    zf.write(full, rel)


if os.environ.get("DRY_RUN"):
    # local check without a student: render every slot at baseline values and at a
    # perturbed point, run the known/unknown-subject smoke, report and exit.
    import random as _r
    _r.seed(1)
    for s, spec in SLOTS.items():
        for label, p in (("baseline", spec["baseline"]), ("perturbed", {k: _r.uniform(lo, hi) for k, (lo, hi) in spec["params"].items()})):
            ok, info = smoke(render(s, p))
            print(f"DRY {s:18s} {label:9s} smoke={'OK' if ok else 'FAIL'} {info}", flush=True)
            assert ok, (s, label, info)
    build_zip(OUT / "dry_control.zip", None)
    print("DRY_RUN OK", flush=True)
    sys.exit(0)

# ------------------------------------------------------------------ student
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = mdl = None; MODEL_ID = None
for cand_id, quant in [("Qwen/Qwen2.5-Coder-7B-Instruct", True), ("Qwen/Qwen2.5-Coder-3B-Instruct", False), ("Qwen/Qwen2.5-Coder-1.5B-Instruct", False)]:
    try:
        print("loading", cand_id, flush=True)
        tok = AutoTokenizer.from_pretrained(cand_id)
        if quant:
            from transformers import BitsAndBytesConfig
            q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4")
            mdl = AutoModelForCausalLM.from_pretrained(cand_id, quantization_config=q, device_map="auto")
        else:
            mdl = AutoModelForCausalLM.from_pretrained(cand_id, torch_dtype=torch.float16, device_map="auto")
        mdl.eval(); MODEL_ID = cand_id; print("student:", MODEL_ID, flush=True); break
    except Exception as e:
        print("  failed:", str(e)[:160], flush=True); tok = mdl = None; torch.cuda.empty_cache()
assert mdl is not None

SYSTEM = """You propose parameter values for one slot of a probability predictor. You never write code.
Output exactly one block and nothing else:
<PARAMS>{"slot": "<slot>", "params": {"<name>": <number>, ...}, "hypothesis": "<one testable sentence>"}</PARAMS>
Rules: every parameter of the slot must be present; every value must lie inside its allowed range;
the proposal must differ from each earlier candidate by at least 1% of some parameter's range;
prefer values that test the stated evidence; no prose outside the block."""


def ask(user_obj, max_new=220, temperature=0.5):
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": json.dumps(user_obj)}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tok([text], return_tensors="pt").to(mdl.device)
    with torch.no_grad():
        out = mdl.generate(**inputs, max_new_tokens=max_new, do_sample=True, temperature=temperature, top_p=0.9, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()


def parse(reply):
    m = re.search(r"<PARAMS>\s*(\{.*?\})\s*</PARAMS>", reply, re.S) or re.search(r"(\{[\s\S]*?\"params\"[\s\S]*?\})", reply)
    if not m:
        return None, "no <PARAMS> JSON found"
    try:
        return json.loads(m.group(1)), None
    except Exception as e:
        return None, f"invalid JSON: {e}"


# baseline choice (recorded, QA-checked)
facts = {"lobo_112_pairs_alc": {"v153": 0.2220, "smoothed_empirical_bayes": 0.2340, "empirical_mean": 0.2633},
         "note": "hosted formative scores are redrawn per submission; do not select by a single hosted score"}
q = {"task": "Choose the baseline for this batch and justify it in <= 2 sentences. Answer ONLY JSON: {\"baseline\": <name>, \"reason\": <text>}", **facts, "candidates": ["v153", "smoothed_empirical_bayes", "empirical_mean"]}
msgs = [{"role": "system", "content": "Answer with one JSON object only."}, {"role": "user", "content": json.dumps(q)}]
text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
inputs = tok([text], return_tensors="pt").to(mdl.device)
with torch.no_grad():
    out = mdl.generate(**inputs, max_new_tokens=120, do_sample=False, pad_token_id=tok.eos_token_id)
choice = tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
json.dump({"raw": choice, "baseline_used": "v153 (contract)"}, open(OUT / "baseline_choice.json", "w"), indent=1)
print("baseline choice:", choice[:200], flush=True)

build_zip(OUT / "cand_000_control.zip", None)
json.dump({"id": 0, "slot": "control", "params": {}, "model_sha256": BASE_SHA, "student_model": MODEL_ID}, open(OUT / "cand_000.meta.json", "w"), indent=1)

plan = [s for s, k in ALLOCATION for _ in range(k)][:N_CANDIDATES]
used = {s: [] for s in SLOTS}
log = open(OUT / "generation_log.jsonl", "a", encoding="utf-8")
ok_count = 0; attempts_total = 0
for idx, slot in enumerate(plan, 1):
    if time.time() - START > MAX_RUNTIME:
        print(f"[watchdog] stop at candidate {idx}", flush=True); break
    spec = SLOTS[slot]; tier = TIERS[(idx - 1) % 3]
    accepted = None; feedback = None
    for attempt in range(1, 4):
        attempts_total += 1
        judged = [j for j in FEEDBACK.get("by_slot", {}).get(slot, []) if j.get("params")][:12]
        user = {"candidate_index": idx, "slot": slot, "radius_tier": tier, "allowed_ranges": spec["params"], "baseline_values": spec["baseline"],
                "evidence": spec["evidence"], "earlier_candidates_in_this_slot": used[slot][-10:],
                "rules_learned_from_previous_batch": [r for r in FEEDBACK.get("rules", []) if r.startswith(slot) or not any(r.startswith(s) for s in SLOTS)],
                "measurement_science_primer": FEEDBACK.get("measurement_science_primer", []),
                "previous_batch_proposals_with_verdict": judged,
                "instruction": "Propose one set of values for this slot at the given radius tier. Avoid regions that were indistinguishable from the baseline; explore around proposals that passed."}
        if feedback:
            user["previous_attempt_error"] = feedback
        try:
            reply = ask(user)
        except Exception as e:
            feedback = f"generation error: {e}"; torch.cuda.empty_cache(); continue
        obj, err = parse(reply)
        if err:
            feedback = err; log.write(json.dumps({"id": idx, "attempt": attempt, "error": err, "raw": reply[:400]}) + "\n"); continue
        params = validate_params(slot, obj.get("params"), used[slot])
        if isinstance(params, str):
            feedback = params; log.write(json.dumps({"id": idx, "attempt": attempt, "error": params, "proposal": obj.get("params")}) + "\n"); continue
        try:
            text = render(slot, params)
            with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as f:
                f.write(text); tmp = f.name
            py_compile.compile(tmp, doraise=True); os.unlink(tmp)
        except Exception as e:
            feedback = f"render/compile error: {str(e)[:120]}"; log.write(json.dumps({"id": idx, "attempt": attempt, "error": feedback}) + "\n"); continue
        ok, info = smoke(text)
        if not ok:
            feedback = f"smoke test failed: {info}"; log.write(json.dumps({"id": idx, "attempt": attempt, "error": feedback}) + "\n"); continue
        accepted = (text, params, obj.get("hypothesis"), attempt, info); break
    if accepted is None:
        log.write(json.dumps({"id": idx, "slot": slot, "status": "FAILED_3_ATTEMPTS", "last_error": feedback}) + "\n"); log.flush()
        print(f"[{idx:03d}] {slot:18s} FAILED: {str(feedback)[:100]}", flush=True); continue
    text, params, hyp, attempt, smoke_out = accepted
    zpath = OUT / f"cand_{idx:03d}_{slot.lower()[:12]}.zip"
    build_zip(zpath, text); used[slot].append(params)
    meta = {"id": idx, "slot": slot, "radius_tier": tier, "params": params, "hypothesis": hyp, "attempts": attempt, "smoke_predictions": smoke_out,
            "student_model": MODEL_ID, "zip": zpath.name, "zip_sha256": hashlib.sha256(zpath.read_bytes()).hexdigest(), "base_model_sha256": BASE_SHA}
    json.dump(meta, open(OUT / f"cand_{idx:03d}.meta.json", "w"), indent=1)
    log.write(json.dumps({"id": idx, "slot": slot, "status": "OK", "attempts": attempt, "params": params}) + "\n"); log.flush()
    ok_count += 1
    print(f"[{idx:03d}] {slot:18s} OK attempt={attempt} params={params} | {int(time.time() - START)}s", flush=True)

summary = {"student_model": MODEL_ID, "mode": "parameter proposal + templated render + known/unknown-subject smoke", "requested": len(plan),
           "attempted": min(len(plan), idx), "accepted": ok_count, "llm_attempts_total": attempts_total, "elapsed_s": round(time.time() - START, 1),
           "zips": sorted(p.name for p in OUT.glob("cand_*.zip")), "base_model_sha256": BASE_SHA}
json.dump(summary, open(OUT / "generation_summary.json", "w"), indent=1)
shutil.make_archive(str(WORK / "qwen_batch"), "zip", OUT)
print("DONE", json.dumps({k: v for k, v in summary.items() if k != "zips"}), flush=True)
