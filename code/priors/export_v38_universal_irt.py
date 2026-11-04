"""Export upgraded v38 data/universal_item_irt.json using Full-Matrix De-Confounded 2PL IRT."""

import json
import pathlib
import numpy as np

ROOT = pathlib.Path(r"c:\Users\Saifuddin\Documents\LOMBA")
DATA_DIR = ROOT / "paiec_v31_ble_hybrid" / "data"
CACHE_PATH = ROOT / "cached_dual_encoder_1600.npz"
BENCHMARKS = [
    "afrimedqa",
    "agentdojo",
    "ai2d_test",
    "hle",
    "matharena",
    "mathvista_mini",
    "mmbench_v11",
    "mmlupro",
]

npz = np.load(CACHE_PATH, allow_pickle=True)
X_dual_all = npz["X_dual"]
bm_arr = npz["bm_arr"]
delta_gt_all = npz["delta_gt"]
disc_gt_all = npz["disc_gt"]

Xc_list = []
centroids = {}
for bm in BENCHMARKS:
    mask = bm_arr == bm
    X_b = X_dual_all[mask]
    Xc_list.append(X_b - X_b.mean(axis=0, keepdims=True))
    # MiniLM 384-d centroid (unscaled)
    emb_m_mean = X_b[:, :384].mean(axis=0)
    emb_m_mean /= np.linalg.norm(emb_m_mean) + 1e-12
    centroids[bm] = [round(float(v), 5) for v in emb_m_mean]

X_tr = np.vstack(Xc_list)
w_ease = np.linalg.solve(X_tr.T @ X_tr + 35.0 * np.eye(X_tr.shape[1], dtype=np.float32), X_tr.T @ delta_gt_all)
w_disc = np.linalg.solve(X_tr.T @ X_tr + 55.0 * np.eye(X_tr.shape[1], dtype=np.float32), X_tr.T @ disc_gt_all)
feat_mean = X_dual_all.mean(axis=0)

out = {
    "w_ease": [round(float(v), 6) for v in w_ease],
    "w_disc": [round(float(v), 6) for v in w_disc],
    "feat_mean": [round(float(v), 6) for v in feat_mean],
    "centroids": centroids,
}

out_path = DATA_DIR / "universal_item_irt.json"
out_path.write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")
print(f"Exported v38 {out_path} ({out_path.stat().st_size:,} bytes)")
print(f"In-sample corr(w_ease): {np.corrcoef(X_tr @ w_ease, delta_gt_all)[0, 1]:+.4f}")
print(f"In-sample corr(w_disc): {np.corrcoef(X_tr @ w_disc, disc_gt_all)[0, 1]:+.4f}")
