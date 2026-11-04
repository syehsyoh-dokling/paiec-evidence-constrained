"""Build credential-free ZIP variants of v153 that change only the budget-0 branch.

Each variant replaces the B=0 block in model.py with a data-driven prior:
    p = 0.5 + SHRINK * (subject_prior_mean - 0.5) + EASE * d_target_2pl
and removes the hard [0.005, 0.495] cap. api_keys are stripped from
submission_config.json in every output ZIP.
"""
import json
import pathlib
import zipfile

ROOT = pathlib.Path(r"c:\Users\Saifuddin\Documents\LOMBA")
SRC_ZIP = ROOT / "submission_paiec_v153_cb449_wp184_deb265_damp34.zip"
OUT_DIR = ROOT / "localsim" / "variants"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OLD_BLOCK = '''    if not labeled:
        base_b0 = p0
        features_str = str(item.get("item_features") or "").lower()
        if any(
            k in features_str
            for k in ("hard", "tier=frontier", "difficult", "level=5", "olympiad")
        ):
            base_b0 = float(max(base_b0 - 0.038, 0.395))
        return float(min(max(base_b0 + 0.238 * d_target_2pl, 0.005), 0.495))
'''

NEW_BLOCK = '''    if not labeled:
        # LOBO-selected budget-0 prior (localsim/b0_eval_driver.py): no hard 0.495 cap,
        # subject prior mean shrunk toward the center, full-weight zero-shot item ease.
        _info = _lookup_db_subject_info(subject)
        _m = float(_info.get("mean", 0.50)) if _info else 0.50
        base_b0 = {center} + {shrink} * (_m - 0.50)
        return float(min(max(base_b0 + {ease} * d_target_2pl, 0.02), 0.98))
'''

# (name, center, shrink, ease): final candidates after the LOBO grid
GRID = [("b0fix_c050_s050_e100", 0.50, 0.5, 1.0), ("b0fix_c055_s050_e100", 0.55, 0.5, 1.0)]


def strip_keys(cfg_bytes: bytes) -> bytes:
    cfg = json.loads(cfg_bytes.decode("utf-8"))
    cfg["api_keys"] = {}
    return json.dumps(cfg, indent=2).encode("utf-8")


def build(name: str, model_text: str | None):
    out = OUT_DIR / f"{name}.zip"
    with zipfile.ZipFile(SRC_ZIP) as zin, zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zout:
        for info in zin.infolist():
            data = zin.read(info.filename)
            if info.filename == "submission_config.json":
                data = strip_keys(data)
            elif info.filename == "model.py" and model_text is not None:
                data = model_text.encode("utf-8")
            zout.writestr(info, data)
    print("wrote", out.name, out.stat().st_size)


with zipfile.ZipFile(SRC_ZIP) as z:
    names = z.namelist()
    base_model = z.read("model.py").decode("utf-8")
print("zip entries:", names)
assert OLD_BLOCK in base_model, "B0 block not found in v153 model.py"

build("v153_nokeys", None)
for tag, center, shrink, ease in GRID:
    build(tag, base_model.replace(OLD_BLOCK, NEW_BLOCK.format(center=center, shrink=shrink, ease=ease)))
