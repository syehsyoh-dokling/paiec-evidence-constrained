"""v401 = budget-0 fix (c=0.50, s=0.5, e=1.0) + the mixture-prior parameters of the best
student candidate (cand_047), built from the clean v153 base. Both effects act on different
budgets (B0 vs B1-B3), so they should add if they are real."""
import json
import pathlib
import zipfile

ROOT = pathlib.Path(r"c:\Users\Saifuddin\Documents\LOMBA")
SRC = ROOT / "localsim" / "variants" / "b0fix_clean_ref.zip"
META = ROOT / "kaggle_push_qwen70" / "output_v2" / "qwen_batch"
OUT = ROOT / "localsim" / "variants"

with zipfile.ZipFile(SRC) as z:
    base = z.read("model.py").decode("utf-8")


def rep(t, old, new):
    assert t.count(old) == 1, (old, t.count(old))
    return t.replace(old, new, 1)


def with_mixture(text, p):
    text = rep(text, "(w1, 0.38, 6.80)", f"(w1, {p['a1']:.3f}, {p['b1']:.3f})")
    text = rep(text, "(w2, 3.10, 5.10)", f"(w2, {p['a2']:.3f}, {p['b2']:.3f})")
    text = rep(text, "(w3, 6.20, 2.80)", f"(w3, {p['a3']:.3f}, {p['b3']:.3f})")
    text = rep(text, "w1 = max(0.04, 0.24 - 0.85 * shift)", f"w1 = max(0.04, {p['w1']:.3f} - 0.85 * shift)")
    return rep(text, "w3 = max(0.04, 0.26 + 0.85 * shift)", f"w3 = max(0.04, {p['w3']:.3f} + 0.85 * shift)")


for cid in ("047", "041"):
    p = json.load(open(META / f"cand_{cid}.meta.json"))["params"]
    text = with_mixture(base, p)
    compile(text, "v401", "exec")
    out = OUT / f"v401_b0fix_mix{cid}.zip"
    with zipfile.ZipFile(SRC) as zin, zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zout:
        for info in zin.infolist():
            data = zin.read(info.filename)
            if info.filename == "model.py":
                data = text.encode("utf-8")
            zout.writestr(info, data)
    print("wrote", out.name, p)
