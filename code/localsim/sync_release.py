"""Copy the current report, scripts and result summaries into paiec_release/ (the public
evidence repository) and refuse if any copied text file contains a secret-looking string."""
import os
import pathlib, re, shutil, sys
ROOT = pathlib.Path(os.environ.get("PAIEC_ROOT", r"c:\Users\Saifuddin\Documents\LOMBA"))
REL = ROOT / "paiec_release"
SECRET = re.compile(r"(nvapi-[A-Za-z0-9_-]{10,}|sk-or-[A-Za-z0-9_-]{10,}|sk-[A-Za-z0-9]{20,}|hf_[A-Za-z0-9]{20,}|Token [0-9a-f]{20,}|api_key\s*[:=]\s*['\"][^'\"]{8,})")
SCRIPTS = """autopilot.py b0_eval_driver.py b0_eval_fold.py calib_report.py make_ablations.py make_b0_variants.py
make_random_slot_candidates.py make_v401.py make_v402.py qa_gate.py select_50.py sim_local.py sim_worker.py
stats_rigor.py strip_hidden_ids.py hosted_defensible.py hosted_noise_cause.py batch_slot_table.py paired_vs.py
rebuild_combined.py make_expanded_zip.py build_priors_expanded.py make_acq_variants.py labeling_uncertainty.py
make_replicate_batch.py make_v403_upload.py upload_batch.py fetch_hosted_scores.py sync_release.py""".split()
copies = [(ROOT / "openreview_report" / "paiec_report.tex", REL / "report.tex")]
copies += [(ROOT / "localsim" / s, REL / "code" / "localsim" / s) for s in SCRIPTS]
copies += [(ROOT / "comparator_logreg" / "model.py", REL / "code" / "comparator_logreg" / "model.py")]
RES = ROOT / "localsim" / "results"
for name in ["calib16_ctx2_report.txt", "stats_rigor.json", "hosted_defensible.txt", "hosted_noise_cause.txt", "batch_slot_table.txt"]:
    copies.append((RES / name, REL / "results" / name))
copies.append((RES / "calib16_ctx2" / "combined.json", REL / "results" / "calib16_ctx2_combined.json"))
for b in ["random_slot", "random_slot2", "qwen_v2", "qwen_v3", "qwen_1p5b"]:
    copies.append((RES / f"qa_{b}" / "AUTOPILOT_REPORT.md", REL / "results" / f"autopilot_{b}.md"))
copies.append((ROOT / "auto_discovery_paiec" / "LOBO_RESULTS.md", REL / "results" / "LOBO_RESULTS.md"))
for f in (ROOT / "openreview_report" / "figures").glob("*.p*"):
    copies.append((f, REL / "figures" / f.name))
missing = [s for s, _ in copies if not s.exists()]
if missing:
    print("missing:", [str(m) for m in missing]); sys.exit(1)
bad = []
for s, d in copies:
    if s.suffix in (".py", ".tex", ".txt", ".md", ".json"):
        txt = s.read_text(encoding="utf-8", errors="replace")
        if SECRET.search(txt):
            bad.append(s.name)
if bad:
    print("secret-looking strings in:", bad); sys.exit(2)
for s, d in copies:
    d.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(s, d)
print(f"copied {len(copies)} files into {REL}")
