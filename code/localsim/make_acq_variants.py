"""Build acquisition-policy variants on top of v402b: random (no labeling.py), the shipped
complementary policy, and uncertainty-weighted policies with mix 0.5 and 1.0."""
import pathlib
import zipfile

ROOT = pathlib.Path(r"c:\Users\Saifuddin\Documents\LOMBA")
SRC = ROOT / "localsim" / "variants" / "v402b_irt_k3_ts2_tb2_w060.zip"
TPL = (ROOT / "localsim" / "labeling_uncertainty.py").read_text(encoding="utf-8")
OUT = ROOT / "localsim" / "variants"


def build(name, labeling_text):
    out = OUT / f"{name}.zip"
    with zipfile.ZipFile(SRC) as zin, zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zout:
        for info in zin.infolist():
            if info.filename == "labeling.py":
                if labeling_text is None:
                    continue
                zout.writestr(info, labeling_text.encode("utf-8"))
            else:
                zout.writestr(info, zin.read(info.filename))
    print("wrote", out.name)


build("v402b_acq_random", None)
for mix in (0.5, 1.0):
    text = TPL.replace("__MIX__", f"{mix:.2f}")
    compile(text, "labeling", "exec")
    build(f"v402b_acq_unc{int(mix*100):03d}", text)
