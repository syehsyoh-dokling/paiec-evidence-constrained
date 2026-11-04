"""Control for the agent experiment: candidates sampled on the SAME slot ranges without a
language model (Latin-hypercube style stratified sampling, fixed seed). Same base ZIP,
same packaging, same QA gate. Slots: B0 (8), LOW_BUDGET_SHRINK (4), PEER_SHIFT (4).
"""
import json
import pathlib
import random
import zipfile

ROOT = pathlib.Path(r"c:\Users\Saifuddin\Documents\LOMBA")
SRC = ROOT / "localsim" / "variants" / "v153_nokeys_clean.zip"
OUT = ROOT / "random_slot_batch_20261104"
OUT.mkdir(exist_ok=True)
rng = random.Random(20261104)

with zipfile.ZipFile(SRC) as z:
    base = z.read("model.py").decode("utf-8")

OLD_B0 = '''    if not labeled:
        base_b0 = p0
        features_str = str(item.get("item_features") or "").lower()
        if any(
            k in features_str
            for k in ("hard", "tier=frontier", "difficult", "level=5", "olympiad")
        ):
            base_b0 = float(max(base_b0 - 0.038, 0.395))
        return float(min(max(base_b0 + 0.238 * d_target_2pl, 0.005), 0.495))
'''
NEW_B0 = '''    if not labeled:
        _info = _lookup_db_subject_info(subject)
        _m = float(_info.get("mean", 0.50)) if _info else 0.50
        base_b0 = {c} + {s} * (_m - 0.50)
        return float(min(max(base_b0 + {e} * d_target_2pl, 0.02), 0.98))
'''


def lhs(n, lo, hi):
    """n stratified samples in [lo, hi]."""
    cells = list(range(n)); rng.shuffle(cells)
    return [lo + (hi - lo) * (k + rng.random()) / n for k in cells]


def replace_once(t, old, new):
    assert t.count(old) == 1, (old[:50], t.count(old))
    return t.replace(old, new, 1)


def build(name, text, meta):
    with zipfile.ZipFile(SRC) as zin, zipfile.ZipFile(OUT / f"{name}.zip", "w", zipfile.ZIP_DEFLATED) as zout:
        for info in zin.infolist():
            data = zin.read(info.filename)
            if info.filename == "model.py":
                data = text.encode("utf-8")
            zout.writestr(info, data)
    json.dump(meta, open(OUT / f"{name}.meta.json", "w"), indent=1)
    print(name, meta["params"])


# B0: 8 candidates
cs, ss, es = lhs(8, 0.48, 0.56), lhs(8, 0.0, 1.0), lhs(8, 0.0, 1.2)
for i in range(8):
    c, s, e = round(cs[i], 3), round(ss[i], 3), round(es[i], 3)
    t = replace_once(base, OLD_B0, NEW_B0.format(c=c, s=s, e=e))
    build(f"rnd_{i+1:03d}_b0", t, {"slot": "B0", "params": {"center": c, "shrink": s, "ease": e}, "sampler": "lhs"})
# LOW_BUDGET_SHRINK: 4 candidates
b1s, b3s, dbs = lhs(4, 0.20, 0.60), lhs(4, 0.05, 0.35), lhs(4, 0.20, 0.32)
for i in range(4):
    b1, b3, db = round(b1s[i], 3), round(b3s[i], 3), round(dbs[i], 3)
    t = base
    t = replace_once(t, "(1.0 - 0.340)", f"(1.0 - {b1:.3f})"); t = replace_once(t, ") * p_base + 0.340 * (0.50", f") * p_base + {b1:.3f} * (0.50")
    t = replace_once(t, "(1.0 - 0.100)", f"(1.0 - {b3:.3f})"); t = replace_once(t, ") * p_base + 0.100 * (0.50", f") * p_base + {b3:.3f} * (0.50")
    t = replace_once(t, "0.265 * count * mean_sampled_d", f"{db:.3f} * count * mean_sampled_d")
    build(f"rnd_{i+9:03d}_low_budget_s", t, {"slot": "LOW_BUDGET_SHRINK", "params": {"blend_count1": b1, "blend_count_le3": b3, "debias": db}, "sampler": "lhs"})
# PEER_SHIFT: 4 candidates
gs, cps, ns = lhs(4, 0.25, 0.70), lhs(4, 0.35, 0.45), lhs(4, 1.0, 6.0)
for i in range(4):
    g, cp, n = round(gs[i], 3), round(cps[i], 4), round(ns[i], 2)
    t = base
    t = replace_once(t, "                0.45\n                * (peer_bm_mean - 0.386)\n                * (total_peer_n / (total_peer_n + 2.0))",
                     f"                {g:.3f}\n                * (peer_bm_mean - {cp:.4f})\n                * (total_peer_n / (total_peer_n + {n:.2f}))")
    build(f"rnd_{i+13:03d}_peer_shift", t, {"slot": "PEER_SHIFT", "params": {"gain": g, "center": cp, "shrink_n": n}, "sampler": "lhs"})
print("built", len(list(OUT.glob("*.zip"))), "candidates in", OUT)
