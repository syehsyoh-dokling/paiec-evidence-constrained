"""Build component-ablation ZIPs of v153 for the LOBO simulator.

abl_no_peer        : peer labels on the same benchmark are ignored
abl_no_kr_cos      : kernel-ridge + cosine-transfer block disabled
abl_no_cosine      : cosine-transfer residual only disabled
abl_no_irt_env     : IRT envelope weight 0.32 -> 0.0 (zero-shot ease still used at budget 0)
abl_random_acq     : labeling.py removed -> evaluator's random acquisition
Each edit is a minimal text substitution verified to occur exactly as expected.
"""
import os
import pathlib
import zipfile

ROOT = pathlib.Path(os.environ.get("PAIEC_ROOT", r"c:\Users\Saifuddin\Documents\LOMBA"))
SRC = ROOT / "localsim" / "variants" / "v153_nokeys.zip"
OUT = ROOT / "localsim" / "variants"

with zipfile.ZipFile(SRC) as z:
    base = z.read("model.py").decode("utf-8")

EDITS = {
    "abl_no_peer": [("        elif obs_bm and obs_bm == benchmark:\n            peer_key = str(", "        elif False and obs_bm == benchmark:\n            peer_key = str(")],
    "abl_no_kr_cos": [("    if (count + total_peer_n) >= 2 and target_fvec is not None:", "    if False and (count + total_peer_n) >= 2 and target_fvec is not None:")],
    "abl_no_cosine": [("                    delta_cos = 0.0\n                    if count >= 2:", "                    delta_cos = 0.0\n                    if False:")],
    "abl_no_irt_env": [("\n    p_final = p_base + env * 0.32 * d_rel_2pl", "\n    p_final = p_base + env * 0.0 * d_rel_2pl"),
                       ("\n            p_final = p_base + env * 0.32 * d_rel_2pl", "\n            p_final = p_base + env * 0.0 * d_rel_2pl"),
                       ("                        + env * (0.32 * d_rel_2pl + w_kr * delta_kr)", "                        + env * (0.0 * d_rel_2pl + w_kr * delta_kr)")],
}


def build(name, model_text, drop_labeling=False):
    out = OUT / f"{name}.zip"
    with zipfile.ZipFile(SRC) as zin, zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zout:
        for info in zin.infolist():
            if drop_labeling and info.filename == "labeling.py":
                continue
            data = zin.read(info.filename)
            if info.filename == "model.py" and model_text is not None:
                data = model_text.encode("utf-8")
            zout.writestr(info, data)
    print("wrote", out.name)


for name, edits in EDITS.items():
    text = base
    for old, new in edits:
        n = text.count(old)
        assert n == 1, f"{name}: expected 1 occurrence, found {n}: {old[:60]!r}"
        text = text.replace(old, new, 1)
    compile(text, name, "exec")
    build(name, text)
build("abl_random_acq", None, drop_labeling=True)
