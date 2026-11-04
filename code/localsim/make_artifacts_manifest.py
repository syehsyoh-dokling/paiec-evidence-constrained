"""Write paiec_release/submission/ARTIFACTS.json: SHA-256 of every archive and of the files
inside them that define the predictor (model.py, labeling.py, data/*.json), with the scripts
and arguments that built the fitted artifacts; and paiec_release/requirements.txt with the
versions installed in the environment that produced every number in the report."""
import hashlib, importlib, json, os, pathlib, platform, sys, zipfile
ROOT = pathlib.Path(os.environ.get("PAIEC_ROOT", r"c:\Users\Saifuddin\Documents\LOMBA"))
REL = ROOT / "paiec_release"
ARCHIVES = {
    "paiec_submission_v153.zip": ROOT / "localsim" / "variants" / "v153_nokeys_clean.zip",
    "paiec_submission_v403.zip": ROOT / "localsim" / "variants" / "v403_expanded_upload.zip",
    "v402b (hosted 1032562)": ROOT / "localsim" / "variants" / "v402b_irt_k3_ts2_tb2_w060.zip",
}
BUILD = {
    "data/subject_priors.json": {
        "v153": "generate_v38_subject_priors_and_simulate_alc.py on the 8-benchmark payload (200 items each); alternating subject x item fixed effects, 40 iterations, ridge 2 or 14",
        "v403": "localsim/build_priors_expanded.py --root <measurement-db long-form root> --max-items-per-benchmark 4000 --frontier matharena hle livecodebench swebench cybench --min-items-per-subject 20 (12 benchmarks, 1186 subject keys)",
    },
    "data/universal_item_irt.json": {
        "v153": "export_v38_universal_irt.py: ridge 35 (ease) / 55 (discrimination) on 781-d pool-centred dual-encoder features of the 8 x 200 items (hle rows included)",
        "v403": "localsim/build_priors_expanded.py (same ridge, 18998 items of 12 benchmarks); localsim/make_expanded_zip.py assembled the archive; localsim/make_v403_upload.py dropped the simulator-only data/expanded_features.npz",
    },
}


def sha(b):
    return hashlib.sha256(b).hexdigest()


out = {"archives": {}, "build": BUILD, "note": "model.py of v402b and v403 is byte-identical; v403 differs from v402b only in the two data/*.json files. Archives are byte-identical to the uploaded submissions and were not edited after upload."}
for name, p in ARCHIVES.items():
    rec = {"source_file": p.name, "sha256": sha(p.read_bytes()), "bytes": p.stat().st_size, "members": {}}
    with zipfile.ZipFile(p) as z:
        for n in z.namelist():
            if n in ("model.py", "labeling.py", "models.txt", "submission_config.json") or (n.startswith("data/") and n.endswith(".json")):
                rec["members"][n] = sha(z.read(n))
    out["archives"][name] = rec
(REL / "submission").mkdir(parents=True, exist_ok=True)
json.dump(out, open(REL / "submission" / "ARTIFACTS.json", "w"), indent=1)
same = out["archives"]["paiec_submission_v403.zip"]["members"]["model.py"] == out["archives"]["v402b (hosted 1032562)"]["members"]["model.py"]
print("ARTIFACTS.json written; v403 model.py == v402b model.py:", same)

pins = []
for mod, dist in [("numpy", "numpy"), ("pandas", "pandas"), ("pyarrow", "pyarrow"), ("torch", "torch"), ("transformers", "transformers"), ("matplotlib", "matplotlib"), ("huggingface_hub", "huggingface_hub"), ("tokenizers", "tokenizers"), ("safetensors", "safetensors")]:
    try:
        m = importlib.import_module(mod); pins.append(f"{dist}=={m.__version__}")
    except Exception as e:
        pins.append(f"# {dist}: not importable ({e.__class__.__name__})")
hdr = [f"# Versions installed in the environment that produced every number in the report.",
       f"# Python {platform.python_version()} on {platform.system()} {platform.release()}; CPU inference (GPU optional).",
       "# The Codabench evaluator installs its own environment; these pins reproduce the local LOBO numbers."]
(REL / "requirements.txt").write_text("\n".join(hdr + pins) + "\n", encoding="utf-8")
print("\n".join(pins))
