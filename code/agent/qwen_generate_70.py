"""Qwen candidate generator under QWEN_70_SUBMISSION_CONTRACT.md.

Qwen (via local_llm.complete, provider from --provider / LLM_PROVIDER) proposes one
<CHANGE> per candidate inside an assigned slot of v153 model.py. This script only:
builds prompts from the live code + knowledge pack, enforces the slot/format rules with
immediate feedback (max 3 attempts), packages accepted edits into ZIPs from
localsim/variants/v153_nokeys.zip, and writes cand_NNN.meta.json. It never edits code
itself. One byte-identical control ZIP is added as candidate 000.

Usage: python qwen_generate_70.py --out qwen_batch_20261104 --n 69 [--provider openrouter]
"""
from __future__ import annotations
import argparse, difflib, hashlib, json, os, py_compile, re, sys, tempfile, time, zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LOMBA = ROOT.parent
sys.path.insert(0, str(ROOT))
from local_llm import complete  # noqa: E402

BASE_ZIP = LOMBA / "localsim" / "variants" / "v153_nokeys.zip"
KNOWLEDGE = ROOT / "qwen_knowledge_v2.json"
ALLOWED_REGIONS = {
    "B0": ("    if not labeled:\n        base_b0 = p0", "return float(min(max(base_b0 + 0.238 * d_target_2pl, 0.005), 0.495))"),
    "MIXTURE_PRIOR": ("def _mixture_bayes_posterior(", "return float(sum(p * m for p, m in zip(probs, post_means)) / max(Z, 1e-12))"),
    "LOW_BUDGET_SHRINK": ("    if count > 0:\n        mean_sampled_d", "        p_base = float(min(max(p_base, 0.005), 0.995))"),
    "PEER_SHIFT": ("        if total_peer_n > 0:\n            peer_bm_mean = sum(peer_ests) / total_peer_n", "            )"),
    "IRT_ENVELOPE": ("    # Boundary damping envelope", "    p_final = p_base + env * 0.32 * d_rel_2pl"),
}
ALLOCATION = [("B0", 15), ("LOW_BUDGET_SHRINK", 21), ("MIXTURE_PRIOR", 13), ("PEER_SHIFT", 10), ("IRT_ENVELOPE", 10)]
RADIUS_TIERS = ["r1 (+-25% of each range around baseline)", "r2 (+-50% of range)", "r3 (anywhere in range)"]
SLOT_HINTS = {
    "IRT_ENVELOPE": "The line 'p_final = p_base + env * 0.32 * d_rel_2pl' also appears elsewhere in model.py, so 'old' must include the preceding 'env = ...' line (multi-line old) to be unique.",
    "LOW_BUDGET_SHRINK": "Change the 0.340 / 0.100 blend weights, the 'count <= 3' bound, or the 0.265 debias slope; keep both occurrences of a weight on the same line consistent.",
    "MIXTURE_PRIOR": "Edit the (w, a, b) tuples of the three regimes and/or the w1/w3 base weights; keep w2 = 0.50.",
    "PEER_SHIFT": "Edit 0.45 (gain), 0.386 (center) or 2.0 (shrink_n) in the prior_shift update.",
    "B0": "Replace the whole zero-label return with a data-driven prior: center + shrink*(subject prior mean - 0.5) + ease*d_target_2pl, clipped to [0.02, 0.98]; _lookup_db_subject_info(subject).get('mean', 0.5) gives the subject prior mean.",
}

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


def region_bounds(text: str, slot: str):
    start, end = ALLOWED_REGIONS[slot]
    i = text.find(start); j = text.find(end, i) if i >= 0 else -1
    assert i >= 0 and j >= 0, slot
    return i, j + len(end)


def snippet(text: str, slot: str) -> str:
    i, j = region_bounds(text, slot)
    return text[i:j]


def lines_touched(base: str, cand: str):
    b, c = base.splitlines(), cand.splitlines()
    touched = set()
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, b, c).get_opcodes():
        if tag != "equal":
            touched.update(range(max(i1, 1), max(i2, i1 + 1) + 1))
    return touched


def region_lines(base: str, slot: str):
    i, j = region_bounds(base, slot)
    return base.count("\n", 0, i) + 1, base.count("\n", 0, j) + 1


def parse_change(reply: str):
    m = re.search(r"<CHANGE>\s*(\{.*?\})\s*</CHANGE>", reply, re.S) or re.search(r"(\{[\s\S]*?\"old\"[\s\S]*?\"new\"[\s\S]*?\})", reply)
    if not m:
        return None, "no <CHANGE> JSON found"
    try:
        return json.loads(m.group(1)), None
    except Exception as e:
        return None, f"invalid JSON inside <CHANGE>: {e}"


def validate(base: str, change: dict, slot: str, used_params: list[dict], ranges: dict):
    old, new = str(change.get("old", "")), str(change.get("new", ""))
    if not old or old == new:
        return None, "old is empty or equal to new"
    if base.count(old) != 1:
        return None, f"'old' occurs {base.count(old)} times in model.py; it must occur exactly once (copy it verbatim from the snippet)"
    cand = base.replace(old, new, 1)
    s, e = region_lines(base, slot)
    outside = [ln for ln in lines_touched(base, cand) if not (s <= ln <= e)]
    if outside:
        return None, f"edit touches lines outside slot {slot} (base lines {sorted(outside)[:4]}); stay inside the snippet"
    if re.search(r"\b(benchmark|subject)_\d{6}\b", new) or re.search(r"^\s*(import|from)\s", new, re.M):
        return None, "new text contains a hidden ID or an import"
    if "def predict(" in new:
        return None, "predict signature touched"
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
            lo, hi = ranges[k]
            try:
                fv = float(v)
            except Exception:
                return None, f"param {k} is not a number"
            if not (lo - 1e-9 <= fv <= hi + 1e-9):
                return None, f"param {k}={fv} outside allowed range [{lo}, {hi}]"
    for prev in used_params:
        shared = [k for k in params if k in prev and k in ranges]
        if shared and all(abs(float(params[k]) - float(prev[k])) < 0.01 * (ranges[k][1] - ranges[k][0]) for k in shared) and set(params) == set(prev):
            return None, f"too close to an existing candidate {prev}; move at least one parameter by >= 1% of its range"
    return cand, None


def build_zip(out: Path, model_text: str | None):
    with zipfile.ZipFile(BASE_ZIP) as zin, zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zout:
        for info in zin.infolist():
            data = zin.read(info.filename)
            if info.filename == "model.py" and model_text is not None:
                data = model_text.encode("utf-8")
            zout.writestr(info, data)


def ask(prompt: str, provider: str, max_tokens: int = 700) -> str:
    text, _ = complete(prompt, provider=provider, system=SYSTEM, max_tokens=max_tokens, temperature=0.3)
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=69)
    ap.add_argument("--provider", default=os.environ.get("LLM_PROVIDER", "openrouter"))
    ap.add_argument("--sleep", type=float, default=3.0)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    know = json.loads(KNOWLEDGE.read_text(encoding="utf-8"))
    with zipfile.ZipFile(BASE_ZIP) as z:
        base = z.read("model.py").decode("utf-8")
    base_sha = hashlib.sha256(base.encode()).hexdigest()

    # Step 2 of the strategy: Qwen must choose and justify the baseline (recorded, QA-checked)
    bq = a.out / "baseline_choice.json"
    if not bq.exists():
        q = {"task": "Choose the baseline submission for this batch and justify it in <= 3 sentences. Answer ONLY JSON: {\"baseline\": <file or version>, \"reason\": <text>}",
             "read_first": know["read_first"], "hosted_scores": know["hosted_scores"], "lobo_full_curve_112_pairs": know["lobo_full_curve_112_pairs"],
             "candidates": ["v153 (localsim/variants/v153_nokeys.zip)", "smoothed_empirical_bayes (official)", "empirical_mean (official)", "any version from all_scored_submissions"]}
        text, _ = complete(json.dumps(q), provider=a.provider, system="Answer with one JSON object only.", max_tokens=300, temperature=0.1)
        flag = bool(re.search(r"lowest|minimum|smallest|terendah|0\.1275|0\.1276", text, re.I)) and not re.search(r"LOBO|paired|expected|lineage", text, re.I)
        json.dump({"raw": text, "qa_flag_selection_on_noise": flag, "baseline_used": "v153_nokeys.zip (contract)"}, open(bq, "w"), indent=1)
        print("baseline choice:", text.strip()[:300], "| QA flag:", flag, flush=True)
        time.sleep(a.sleep)

    # candidate 000 = byte-identical control
    ctrl = a.out / "cand_000_control.zip"
    if not ctrl.exists():
        build_zip(ctrl, None)
        json.dump({"id": 0, "slot": "control", "params": {}, "hypothesis": "byte-identical v153 control", "model_sha256": base_sha}, open(a.out / "cand_000.meta.json", "w"), indent=1)

    plan = [slot for slot, k in ALLOCATION for _ in range(k)][: a.n]
    used_params: dict[str, list[dict]] = {s: [] for s in ALLOWED_REGIONS}
    rejected: dict[str, list[dict]] = {s: [] for s in ALLOWED_REGIONS}
    for meta_file in sorted(a.out.glob("cand_*.meta.json")):
        m = json.loads(meta_file.read_text())
        if m.get("slot") in used_params and m.get("params"):
            used_params[m["slot"]].append(m["params"])
    log = open(a.out / "generation_log.jsonl", "a", encoding="utf-8")
    idx = 1
    for slot in plan:
        zpath = a.out / f"cand_{idx:03d}_{slot.lower()[:12]}.zip"
        if zpath.exists():
            idx += 1; continue
        spec = know["slots"][slot]
        ranges = {k: v for k, v in spec["params"].items() if isinstance(v, list) and len(v) == 2 and all(isinstance(x, (int, float)) for x in v)}
        tier = RADIUS_TIERS[(idx - 1) % 3]
        accepted = None
        feedback = None
        for attempt in range(1, 4):
            user = {
                "candidate_index": idx, "slot": slot, "radius_tier": tier,
                "live_snippet_from_model_py": snippet(base, slot),
                "allowed_param_ranges": spec["params"], "baseline_values": spec["baseline_values"], "evidence": spec["evidence"],
                "slot_hint": SLOT_HINTS.get(slot, ""),
                "params_already_used_in_this_batch": used_params[slot][-12:],
                "rejected_in_this_batch": rejected[slot][-6:],
                "instruction": "Propose ONE change inside this slot at the given radius tier. Copy 'old' verbatim from the live snippet. List all changed constants in params.",
            }
            if feedback:
                user["previous_attempt_error"] = feedback
            try:
                reply = ask(json.dumps(user), a.provider)
            except Exception as e:
                feedback = f"provider error: {e}"; time.sleep(10); continue
            change, err = parse_change(reply)
            if err:
                feedback = err; log.write(json.dumps({"id": idx, "attempt": attempt, "error": err, "raw": reply[:800]}) + "\n"); continue
            cand, err = validate(base, change, slot, used_params[slot], ranges)
            if err:
                feedback = err; rejected[slot].append({"params": change.get("params"), "reason": err[:120]})
                log.write(json.dumps({"id": idx, "attempt": attempt, "error": err, "change": change}) + "\n"); continue
            accepted = (cand, change, attempt, reply)
            break
        if accepted is None:
            log.write(json.dumps({"id": idx, "slot": slot, "status": "FAILED_3_ATTEMPTS", "last_error": feedback}) + "\n"); log.flush()
            print(f"[{idx:03d}] {slot:18s} FAILED: {str(feedback)[:120]}", flush=True)
            idx += 1; time.sleep(a.sleep); continue
        cand, change, attempt, reply = accepted
        build_zip(zpath, cand)
        used_params[slot].append(change.get("params"))
        meta = {"id": idx, "slot": slot, "radius_tier": tier, "params": change.get("params"), "hypothesis": change.get("hypothesis"),
                "old": change.get("old"), "new": change.get("new"), "attempts": attempt, "provider": a.provider,
                "zip": zpath.name, "zip_sha256": hashlib.sha256(zpath.read_bytes()).hexdigest(), "base_model_sha256": base_sha}
        json.dump(meta, open(a.out / f"cand_{idx:03d}.meta.json", "w"), indent=1)
        log.write(json.dumps({"id": idx, "slot": slot, "status": "OK", "attempts": attempt, "params": change.get("params")}) + "\n"); log.flush()
        print(f"[{idx:03d}] {slot:18s} OK attempt={attempt} params={change.get('params')}", flush=True)
        idx += 1
        time.sleep(a.sleep)
    print("done:", len(list(a.out.glob("cand_*.zip"))), "zips in", a.out)


if __name__ == "__main__":
    main()
