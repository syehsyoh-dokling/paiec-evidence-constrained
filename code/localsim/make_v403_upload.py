"""Build the upload copy of v403 (= v402b + expanded priors) by dropping the simulator-only
data/expanded_features.npz, and verify model.py is byte-identical to v402b's."""
import hashlib, pathlib, sys, zipfile
V = pathlib.Path(r"c:\Users\Saifuddin\Documents\LOMBA\localsim\variants")
src = V / "v402b_expanded_priors.zip"
base = V / "v402b_irt_k3_ts2_tb2_w060.zip"
out = V / "v403_expanded_upload.zip"
with zipfile.ZipFile(src) as zs, zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zo:
    for info in zs.infolist():
        if info.filename.endswith("expanded_features.npz"):
            continue
        zo.writestr(info, zs.read(info))
def sha(z, name):
    with zipfile.ZipFile(z) as f:
        return hashlib.sha256(f.read(name)).hexdigest()
same = sha(out, "model.py") == sha(base, "model.py")
with zipfile.ZipFile(out) as f, zipfile.ZipFile(base) as g:
    a = {i.filename: hashlib.sha256(f.read(i)).hexdigest() for i in f.infolist()}
    b = {i.filename: hashlib.sha256(g.read(i)).hexdigest() for i in g.infolist()}
print("wrote", out.name, out.stat().st_size // 1024, "KB; model.py identical to v402b:", same)
print("entries only in v403:", sorted(set(a) - set(b)), "only in v402b:", sorted(set(b) - set(a)))
print("changed entries:", sorted(k for k in a if k in b and a[k] != b[k]))
sys.exit(0 if same else 1)
