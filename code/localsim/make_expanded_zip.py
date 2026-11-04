"""Ship expanded artifacts inside a candidate ZIP: replaces data/subject_priors.json and
data/universal_item_irt.json with the outputs of build_priors_expanded.py and adds
data/expanded_features.npz so the LOBO patch can refit the regressor without the held-out
benchmark. Usage: python make_expanded_zip.py <base.zip> <artifacts_dir> <out.zip>"""
import pathlib
import sys
import zipfile

base, art, out = (pathlib.Path(p) for p in sys.argv[1:4])
with zipfile.ZipFile(base) as zin, zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zout:
    for info in zin.infolist():
        if info.filename == "data/subject_priors.json":
            zout.writestr(info, (art / "subject_priors.json").read_bytes())
        elif info.filename == "data/universal_item_irt.json":
            zout.writestr(info, (art / "universal_item_irt.json").read_bytes())
        else:
            zout.writestr(info, zin.read(info.filename))
    zout.write(art / "cached_features.npz", "data/expanded_features.npz")
print("wrote", out, out.stat().st_size // 1024, "KB")
