"""Push the parameter-proposal kernel with the previous batch's feedback injected.
Usage: python push_kernel_v3.py <feedback_for_student.json> [--slug paiec-qwen-param-generator-v3]
Writes kernel_v3/ (script with FEEDBACK filled in + metadata) and runs `kaggle kernels push`.
"""
import argparse
import json
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("feedback", type=pathlib.Path)
ap.add_argument("--slug", default="paiec-qwen-param-generator-v3")
ap.add_argument("--owner", default="syehsyoh")
a = ap.parse_args()

src = (HERE / "kernel_v2" / "paiec_qwen_param_generator_70.py").read_text(encoding="utf-8")
fb = json.load(open(a.feedback, encoding="utf-8"))
primer_path = HERE.parent / "auto_discovery_paiec" / "aims_measurement_primer.json"
primer = json.load(open(primer_path, encoding="utf-8")) if primer_path.exists() else {}
compact = {"rules": fb.get("rules", []) + primer.get("rules_for_proposals", []),
           "measurement_science_primer": [{"name": c["name"], "formula": c["formula"], "meaning": c["meaning"]} for c in primer.get("concepts", [])],
           "by_slot": {s: [{"params": i["params"], "verdict": i["verdict"], "dalc": i["dalc"]} for i in items if i.get("params")][:12]
                       for s, items in fb.get("by_slot", {}).items()}}
payload = json.dumps(compact).replace("'", "\\u0027")
out_dir = HERE / "kernel_v3"
out_dir.mkdir(exist_ok=True)
script = src.replace("__FEEDBACK_JSON__", payload)
assert "__FEEDBACK_JSON__" not in script
(out_dir / "paiec_qwen_param_generator_v3.py").write_text(script, encoding="utf-8")
meta = json.load(open(HERE / "kernel_v2" / "kernel-metadata.json"))
meta.update({"id": f"{a.owner}/{a.slug}", "title": a.slug, "code_file": "paiec_qwen_param_generator_v3.py"})
json.dump(meta, open(out_dir / "kernel-metadata.json", "w"), indent=2)
print("feedback rules:", len(compact["rules"]), "| judged proposals:", sum(len(v) for v in compact["by_slot"].values()))
p = subprocess.run([sys.executable, "-m", "kaggle", "kernels", "push", "-p", str(out_dir)], capture_output=True, text=True, encoding="utf-8", errors="replace")
print(p.stdout.strip(), p.stderr.strip()[-300:])
