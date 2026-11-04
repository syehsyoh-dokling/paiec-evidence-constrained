#!/usr/bin/env python3
"""PAIEC slot-based Qwen candidate generator (Kaggle T4 kernel).

Contract: auto_discovery_paiec/QWEN_70_SUBMISSION_CONTRACT.md.
Student: Qwen2.5-Coder (7B 4-bit, fallback 3B / 1.5B fp16) running locally on the GPU.
Unlike the previous kernel, the student's <CHANGE> is the ONLY thing applied to model.py;
nothing is pre-scripted. The script enforces the slot/format rules, gives the student
error feedback (max 3 attempts), packages accepted edits into ZIPs derived from the v153
baseline with api_keys stripped, and writes cand_NNN.meta.json + generation_log.jsonl.
Candidate 000 is the byte-identical control.
"""
import difflib
import hashlib
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
print("=" * 80)
print("PAIEC SLOT-BASED QWEN GENERATOR | start", time.strftime("%Y-%m-%d %H:%M:%S"), "| budget", MAX_RUNTIME // 60, "min")

# ---------------------------------------------------------------- baseline
base_dir = None
for cb in [Path("/kaggle/input/paiec-baseline-and-knowledge/paiec_v153_baseline"), Path("/kaggle/input/paiec-baseline-and-knowledge"), Path("paiec_v153_baseline")]:
    if (cb / "model.py").exists():
        base_dir = cb; break
if base_dir is None:
    for p in Path("/kaggle/input").rglob("model.py"):
        if (p.parent / "labeling.py").exists():
            base_dir = p.parent; break
assert base_dir is not None, "baseline not found"
print("baseline:", base_dir)
BASE = (base_dir / "model.py").read_text(encoding="utf-8")
BASE_SHA = hashlib.sha256(BASE.encode()).hexdigest()

# ---------------------------------------------------------------- slots (from QWEN_70_SUBMISSION_CONTRACT.md §2)
REGIONS = {
    "B0": ("    if not labeled:\n        base_b0 = p0", "return float(min(max(base_b0 + 0.238 * d_target_2pl, 0.005), 0.495))"),
    "MIXTURE_PRIOR": ("def _mixture_bayes_posterior(", "return float(sum(p * m for p, m in zip(probs, post_means)) / max(Z, 1e-12))"),
    "LOW_BUDGET_SHRINK": ("    if count > 0:\n        mean_sampled_d", "        p_base = float(min(max(p_base, 0.005), 0.995))"),
    "PEER_SHIFT": ("        if total_peer_n > 0:\n            peer_bm_mean = sum(peer_ests) / total_peer_n", "            )"),
    "IRT_ENVELOPE": ("    # Boundary damping envelope", "    p_final = p_base + env * 0.32 * d_rel_2pl"),
}
SLOTS = {
    "B0": {"params": {"center": [0.48, 0.56], "shrink": [0.0, 1.0], "ease": [0.0, 1.2]},
           "baseline_values": {"center": 0.449, "shrink": 0.20, "ease": 0.238, "hard_cap": 0.495},
           "evidence": "LOBO: budget-0 Brier 0.2559 is worse than constant 0.5 (0.2500); fix center 0.50/shrink 0.5/ease 1.0 gives 0.2482",
           "hint": "Replace the whole zero-label return with a data-driven prior: center + shrink*(subject prior mean - 0.5) + ease*d_target_2pl, clipped to [0.02, 0.98]; _lookup_db_subject_info(subject).get('mean', 0.5) gives the subject prior mean."},
    "LOW_BUDGET_SHRINK": {"params": {"blend_count1": [0.20, 0.60], "blend_count_le3": [0.05, 0.35], "max_count": [3, 7], "debias": [0.20, 0.32]},
                          "baseline_values": {"blend_count1": 0.340, "blend_count_le3": 0.100, "max_count": 3, "debias": 0.265},
                          "evidence": "LOBO: Brier at 1 label (0.2497) is almost equal to 0 labels (0.2559): the first label adds nearly nothing; budgets 1 and 3 carry 40% of the metric",
                          "hint": "Change the 0.340 / 0.100 blend weights, the 'count <= 3' bound, or the 0.265 debias slope; keep both occurrences of a weight on the same line consistent."},
    "MIXTURE_PRIOR": {"params": {"a_b_each_regime": [0.3, 8.0], "w1_w3": [0.10, 0.40]},
                      "baseline_values": {"regime1": [0.38, 6.80], "regime2": [3.10, 5.10], "regime3": [6.20, 2.80], "w1": 0.24, "w3": 0.26},
                      "evidence": "pseudo-counts of the 3 Beta regimes control over-reaction at 1-3 labels",
                      "hint": "Edit the (w, a, b) tuples of the three regimes and/or the w1/w3 base weights; keep w2 = 0.50."},
    "PEER_SHIFT": {"params": {"gain": [0.25, 0.70], "center": [0.35, 0.45], "shrink_n": [1.0, 6.0]},
                   "baseline_values": {"gain": 0.45, "center": 0.386, "shrink_n": 2.0},
                   "evidence": "peers on the same benchmark are the only signal at budget 1",
                   "hint": "Edit 0.45 (gain), 0.386 (center) or 2.0 (shrink_n) in the prior_shift update."},
    "IRT_ENVELOPE": {"params": {"env_scale": [0.04, 0.12], "env_floor": [0.05, 0.40], "irt_weight": [0.0, 0.50]},
                     "baseline_values": {"env_scale": 0.06, "env_floor": 0.15, "irt_weight": 0.32},
                     "evidence": "LOBO: Brier at 31 labels (0.1947) equals a constant at the pair pass rate (0.1905): the IRT term adds no item discrimination",
                     "hint": "The line 'p_final = p_base + env * 0.32 * d_rel_2pl' also appears elsewhere in model.py, so 'old' must include the preceding 'env = ...' line (multi-line old) to be unique."},
}
ALLOCATION = [("B0", 15), ("LOW_BUDGET_SHRINK", 21), ("MIXTURE_PRIOR", 13), ("PEER_SHIFT", 10), ("IRT_ENVELOPE", 10)]
TIERS = ["r1 (+-25% of each range around baseline)", "r2 (+-50% of range)", "r3 (anywhere in range)"]
SYSTEM = """You are a code-editing agent for the PAIEC predictor. You will be given ONE live code
snippet from model.py (exact text) and ONE edit slot. Your only output is a single <CHANGE> block:
<CHANGE>{"old": "<exact text copied from the snippet>", "new": "<replacement>", "hypothesis": "<one sentence, testable>", "slot": "<slot name>", "params": {"<name>": <number>}}</CHANGE>
Hard rules (violating any one = your output is discarded):
1. "old" must be copied character-for-character from the snippet and occur exactly once in model.py.
2. Edit only inside the given slot. Never touch: def predict signature, imports, labeling.py,
   submission_config.json, data/, anything outside the snippet.
3. No network, no file writes, no new imports, no API keys, no benchmark_###### or
   subject_###### identifiers, no comments that mention hidden Codabench IDs or scores.
4. Numbers must stay inside the slot's allowed ranges. A change of less than 1% of a range is NOT a candidate.
5. Each candidate must differ from every earlier candidate in this batch in at least one
   parameter value by >= 1% of its range. Do not repeat a rejected candidate.
6. Return nothing else: no prose, no markdown fences, no explanation outside "hypothesis".
7. "params" must list every numeric constant you changed, with the NEW values, using the parameter names given."""


def bounds(slot):
    s, e = REGIONS[slot]
    i = BASE.find(s); j = BASE.find(e, i)
    assert i >= 0 and j >= 0, slot
    return i, j + len(e)


def snippet(slot):
    i, j = bounds(slot); return BASE[i:j]


def region_lines(slot):
    i, j = bounds(slot); return BASE.count("\n", 0, i) + 1, BASE.count("\n", 0, j) + 1


def touched_lines(cand):
    b, c = BASE.splitlines(), cand.splitlines(); t = set()
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, b, c).get_opcodes():
        if tag != "equal":
            t.update(range(max(i1, 1), max(i2, i1 + 1) + 1))
    return t


def parse_change(reply):
    m = re.search(r"<CHANGE>\s*(\{.*?\})\s*</CHANGE>", reply, re.S) or re.search(r"(\{[\s\S]*?\"old\"[\s\S]*?\"new\"[\s\S]*?\})", reply)
    if not m:
        return None, "no <CHANGE> JSON found"
    try:
        return json.loads(m.group(1)), None
    except Exception as e:
        return None, f"invalid JSON inside <CHANGE>: {e}"


def validate(change, slot, used, ranges):
    old, new = str(change.get("old", "")), str(change.get("new", ""))
    if not old or old == new:
        return None, "old is empty or equal to new"
    if BASE.count(old) != 1:
        return None, f"'old' occurs {BASE.count(old)} times in model.py; it must occur exactly once (copy it verbatim from the snippet)"
    cand = BASE.replace(old, new, 1)
    s, e = region_lines(slot)
    outside = [ln for ln in touched_lines(cand) if not (s <= ln <= e)]
    if outside:
        return None, f"edit touches lines outside slot {slot} (base lines {sorted(outside)[:4]}); stay inside the snippet"
    if re.search(r"\b(benchmark|subject)_\d{6}\b", new) or re.search(r"^\s*(import|from)\s", new, re.M) or "def predict(" in new:
        return None, "new text contains a hidden ID, an import, or the predict signature"
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(cand); tmp = f.name
    try:
        py_compile.compile(tmp, doraise=True)
    except Exception as ex:
        return None, f"compile error: {str(ex)[:160]}"
    finally:
        os.unlink(tmp)
    params = change.get("params") or {}
    if not isinstance(params, dict) or not params:
        return None, "params must be a non-empty object of the constants you changed"
    for k, v in params.items():
        if k in ranges:
            try:
                fv = float(v)
            except Exception:
                return None, f"param {k} is not a number"
            lo, hi = ranges[k]
            if not (lo - 1e-9 <= fv <= hi + 1e-9):
                return None, f"param {k}={fv} outside allowed range [{lo}, {hi}]"
    for prev in used:
        shared = [k for k in params if k in prev and k in ranges]
        if shared and set(params) == set(prev) and all(abs(float(params[k]) - float(prev[k])) < 0.01 * (ranges[k][1] - ranges[k][0]) for k in shared):
            return None, f"too close to an existing candidate {prev}; move at least one parameter by >= 1% of its range"
    return cand, None


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


# ---------------------------------------------------------------- student model
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = mdl = None
MODEL_ID = None
for cand_id, kw in [("Qwen/Qwen2.5-Coder-7B-Instruct", {"quant": True}), ("Qwen/Qwen2.5-Coder-3B-Instruct", {}), ("Qwen/Qwen2.5-Coder-1.5B-Instruct", {})]:
    try:
        print("loading", cand_id, flush=True)
        tok = AutoTokenizer.from_pretrained(cand_id)
        if kw.get("quant"):
            from transformers import BitsAndBytesConfig
            q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4")
            mdl = AutoModelForCausalLM.from_pretrained(cand_id, quantization_config=q, device_map="auto")
        else:
            mdl = AutoModelForCausalLM.from_pretrained(cand_id, torch_dtype=torch.float16, device_map="auto")
        mdl.eval(); MODEL_ID = cand_id
        print("student loaded:", MODEL_ID, "| VRAM GB", round(torch.cuda.memory_allocated() / 1e9, 2), flush=True)
        break
    except Exception as e:
        print("  failed:", str(e)[:200], flush=True)
        tok = mdl = None; torch.cuda.empty_cache()
assert mdl is not None, "no student model could be loaded"


def ask(user_obj, max_new=500, temperature=0.3):
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": json.dumps(user_obj)}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tok([text], return_tensors="pt").to(mdl.device)
    with torch.no_grad():
        out = mdl.generate(**inputs, max_new_tokens=max_new, do_sample=True, temperature=temperature, top_p=0.9, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()


# ---------------------------------------------------------------- step 2: baseline choice (recorded, QA-checked)
hosted_facts = {"hosted_scores": {"n_scored": 172, "best_single_draw": 0.127586, "family_median": 0.185, "sd_single_submission": 0.027, "same_zip_twice": [0.2016, 0.1932]},
                "lobo_112_pairs": {"v153": 0.2220, "smoothed_empirical_bayes": 0.2340, "empirical_mean": 0.2633},
                "read_first": ["Formative scores are redrawn per submission from 8-9 pairs; one score has sd ~0.027.",
                               "The lowest hosted score is the best of 172 draws of near-identical code. Do NOT select a baseline by lowest hosted score.",
                               "Select by LOBO expected ALC and code lineage."]}
q = {"task": "Choose the baseline submission for this batch and justify it in <= 3 sentences. Answer ONLY JSON: {\"baseline\": <name>, \"reason\": <text>}",
     **hosted_facts, "candidates": ["v153", "smoothed_empirical_bayes (official)", "empirical_mean (official)"]}
msgs = [{"role": "system", "content": "Answer with one JSON object only."}, {"role": "user", "content": json.dumps(q)}]
text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
inputs = tok([text], return_tensors="pt").to(mdl.device)
with torch.no_grad():
    out = mdl.generate(**inputs, max_new_tokens=200, do_sample=False, pad_token_id=tok.eos_token_id)
choice = tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
flag = bool(re.search(r"lowest|minimum|smallest|0\.1275|0\.1276", choice, re.I)) and not re.search(r"LOBO|paired|expected|lineage", choice, re.I)
json.dump({"raw": choice, "qa_flag_selection_on_noise": flag, "baseline_used": "v153 (contract)"}, open(OUT / "baseline_choice.json", "w"), indent=1)
print("baseline choice:", choice[:300], "| QA flag:", flag, flush=True)

# ---------------------------------------------------------------- control
build_zip(OUT / "cand_000_control.zip", None)
json.dump({"id": 0, "slot": "control", "params": {}, "hypothesis": "byte-identical v153 control", "model_sha256": BASE_SHA, "student_model": MODEL_ID}, open(OUT / "cand_000.meta.json", "w"), indent=1)

# ---------------------------------------------------------------- generation loop
plan = [s for s, k in ALLOCATION for _ in range(k)][:N_CANDIDATES]
used = {s: [] for s in SLOTS}
rejected = {s: [] for s in SLOTS}
log = open(OUT / "generation_log.jsonl", "a", encoding="utf-8")
ok_count = 0
for idx, slot in enumerate(plan, 1):
    if time.time() - START > MAX_RUNTIME:
        print(f"[watchdog] {MAX_RUNTIME // 60} min reached at candidate {idx}; stopping", flush=True); break
    spec = SLOTS[slot]
    ranges = {k: v for k, v in spec["params"].items() if isinstance(v, list) and len(v) == 2}
    tier = TIERS[(idx - 1) % 3]
    accepted = None; feedback = None
    for attempt in range(1, 4):
        user = {"candidate_index": idx, "slot": slot, "radius_tier": tier, "live_snippet_from_model_py": snippet(slot),
                "allowed_param_ranges": spec["params"], "baseline_values": spec["baseline_values"], "evidence": spec["evidence"], "slot_hint": spec["hint"],
                "params_already_used_in_this_batch": used[slot][-12:], "rejected_in_this_batch": rejected[slot][-6:],
                "instruction": "Propose ONE change inside this slot at the given radius tier. Copy 'old' verbatim from the live snippet. List all changed constants in params."}
        if feedback:
            user["previous_attempt_error"] = feedback
        try:
            reply = ask(user)
        except Exception as e:
            feedback = f"generation error: {e}"; torch.cuda.empty_cache(); continue
        change, err = parse_change(reply)
        if err:
            feedback = err; log.write(json.dumps({"id": idx, "attempt": attempt, "error": err, "raw": reply[:600]}) + "\n"); continue
        cand, err = validate(change, slot, used[slot], ranges)
        if err:
            feedback = err; rejected[slot].append({"params": change.get("params"), "reason": err[:120]})
            log.write(json.dumps({"id": idx, "attempt": attempt, "error": err, "change": change}) + "\n"); continue
        accepted = (cand, change, attempt, reply); break
    if accepted is None:
        log.write(json.dumps({"id": idx, "slot": slot, "status": "FAILED_3_ATTEMPTS", "last_error": feedback}) + "\n"); log.flush()
        print(f"[{idx:03d}] {slot:18s} FAILED: {str(feedback)[:110]}", flush=True); continue
    cand, change, attempt, reply = accepted
    zpath = OUT / f"cand_{idx:03d}_{slot.lower()[:12]}.zip"
    build_zip(zpath, cand)
    used[slot].append(change.get("params"))
    meta = {"id": idx, "slot": slot, "radius_tier": tier, "params": change.get("params"), "hypothesis": change.get("hypothesis"),
            "old": change.get("old"), "new": change.get("new"), "attempts": attempt, "student_model": MODEL_ID,
            "zip": zpath.name, "zip_sha256": hashlib.sha256(zpath.read_bytes()).hexdigest(), "base_model_sha256": BASE_SHA}
    json.dump(meta, open(OUT / f"cand_{idx:03d}.meta.json", "w"), indent=1)
    log.write(json.dumps({"id": idx, "slot": slot, "status": "OK", "attempts": attempt, "params": change.get("params")}) + "\n"); log.flush()
    ok_count += 1
    print(f"[{idx:03d}] {slot:18s} OK attempt={attempt} params={change.get('params')} | {int(time.time() - START)}s", flush=True)

summary = {"student_model": MODEL_ID, "requested": len(plan), "accepted": ok_count, "elapsed_s": round(time.time() - START, 1),
           "zips": sorted(p.name for p in OUT.glob("cand_*.zip")), "base_model_sha256": BASE_SHA}
json.dump(summary, open(OUT / "generation_summary.json", "w"), indent=1)
shutil.make_archive(str(WORK / "qwen_batch"), "zip", OUT)
print("DONE", json.dumps({k: v for k, v in summary.items() if k != "zips"}), flush=True)
