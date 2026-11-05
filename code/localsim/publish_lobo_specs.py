"""Package the fixed LOBO fold specifications (identical cases for every variant) for other teams,
WITHOUT benchmark item text: every item_content is replaced by its SHA-256 so the specification is
verifiable and reproducible from measurement-db tables (sim_local.build_spec rebuilds the same
pairs, splits and acquisition order from the same hash rules), while no upstream benchmark content
is redistributed. Output: openreview_report/lobo_specs_v1.zip"""
import hashlib, io, json, os, pathlib, zipfile
ROOT = pathlib.Path(os.environ.get("PAIEC_ROOT", r"c:\Users\Saifuddin\Documents\LOMBA"))
SRC = ROOT / "paiec_eval_resource.zip"
OUT = ROOT / "openreview_report" / "lobo_specs_v1.zip"
README = """# PAIEC leave-one-benchmark-out specifications, v1

Fixed evaluation cases used for every paired comparison in the technical report
"Evidence-Constrained Probabilistic Prediction of AI Response Outcomes ..." (PAIEC, NeurIPS 2026).

Each specs/<benchmark>_s0_p16[_c2].json holds one held-out public benchmark ("fold"): 16 subject-benchmark
pairs (plus up to 2 unscored context pairs per draw in the _c2 files), each with the subject record, the
acquisition items in their fixed order with labels, and the evaluation items with their observed labels.
Item text is NOT included: every item_content is replaced by "sha256:<hex>" of the original text, so a
holder of the measurement-db tables can verify the exact items, and sim_local.build_spec (released with
the report) regenerates the same pairs, 50/50 splits and acquisition order from the same hash rules.
To run a predictor on these cases, rebuild the specs with build_spec on your measurement-db copy and
compare the item hashes, or use sim_local.py directly (it does the same thing).

Benchmark identifiers inside the specs are anonymised exactly as the simulator presents them to a
predictor. Licence: CC BY 4.0 for the specification files; the underlying benchmark data stay under
their upstream licences and are not redistributed here.
"""
with zipfile.ZipFile(SRC) as zs, zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as zo:
    zo.writestr("README.md", README)
    n_items = 0
    for name in zs.namelist():
        if not (name.startswith("specs/") and name.endswith(".json")):
            continue
        spec = json.loads(zs.read(name))
        # local anonymised ids are prefixed so they cannot be mistaken for the platform's hidden ids
        for d in spec["draws"]:
            for p in d["pairs"]:
                p["subject_id"] = "lobo_" + str(p["subject_id"])
                for a in p.get("acq", []):
                    a["item"]["benchmark_id"] = "lobo_" + str(a["item"]["benchmark_id"])
                for e in p.get("eval", []):
                    e["item"]["benchmark_id"] = "lobo_" + str(e["item"]["benchmark_id"])
                for a in p.get("acq", []):
                    a["item"]["item_content"] = "sha256:" + hashlib.sha256(a["item"]["item_content"].encode("utf-8")).hexdigest(); n_items += 1
                for e in p.get("eval", []):
                    e["item"]["item_content"] = "sha256:" + hashlib.sha256(e["item"]["item_content"].encode("utf-8")).hexdigest(); n_items += 1
        zo.writestr(name, json.dumps(spec, separators=(",", ":")))
print("wrote", OUT, OUT.stat().st_size // 1024, "KB; items hashed:", n_items)
# safety check: no residual plain text
with zipfile.ZipFile(OUT) as z:
    bad = [n for n in z.namelist() if n.endswith(".json") and '"item_content":"sha256:' not in z.read(n).decode()[:5000]]
    leaks = sum(1 for n in z.namelist() if n.endswith(".json") for m in [z.read(n).decode()] if '"item_content":"' in m and '"item_content":"sha256:' not in m)
print("files without hashed content:", bad, "| files with plain content:", leaks)
