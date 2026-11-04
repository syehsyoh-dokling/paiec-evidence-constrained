"""Rebuild data/subject_priors.json and data/universal_item_irt.json from a LARGER measurement-db
extract (many benchmarks, all items), in exactly the format model.py expects.

Input layouts supported (auto-detected):
  A) official payload layout:  <root>/<benchmark>/{subjects,items,response}.parquet
  B) long-form layout:         <root>/<benchmark>.parquet (subject_id,item_id,response,trial) +
                               <root>/items.parquet (item_id,benchmark_id,content) + <root>/subjects.parquet
Steps (mirrors generate_v38_subject_priors_and_simulate_alc.py and export_v38_universal_irt.py):
  1. per benchmark: alternating item/subject fixed effects on the binary response matrix
     -> item ease (ease_ib), subject item-adjusted ability (theta_sb), item discrimination proxy
  2. two-factor subject priors (general / frontier) with the same offsets as v38, keyed by
     canonical model name and by raw name
  3. item features for every text item (MiniLM + bge-small + 13 structural, same scaling as
     model.py) and ridge regression to ease / discrimination -> w_ease, w_disc, feat_mean, centroids
Outputs go to --out (default localsim/artifacts_expanded/). Benchmarks whose item content is an
identifier (e.g. hle) are used for subject priors only, never for the regressor.
Usage: python build_priors_expanded.py --root <dir> --out <dir> [--max-items-per-benchmark 4000] [--frontier matharena hle ...]
"""
import argparse
import json
import math
import pathlib
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # LOMBA root for the v38 helpers

ROOT = pathlib.Path(os.environ.get("PAIEC_ROOT", r"c:\Users\Saifuddin\Documents\LOMBA"))
MODEL_PY = ROOT / "paiec_v153_baseline" / "model.py"


def load_model_helpers():
    """Import the feature functions from model.py itself so scaling stays identical."""
    import importlib.util, os
    os.environ["PAIEC_DISABLE_BLE_NETWORK"] = "1"; os.environ["PREDICTIVE_EVAL_LOCAL_SMOKE_TEST"] = "1"
    sys.path.insert(0, str(MODEL_PY.parent)); sys.path.insert(0, str(MODEL_PY.parent / "src"))
    spec = importlib.util.spec_from_file_location("v153model", str(MODEL_PY))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def load_benchmarks(root: pathlib.Path):
    """Yield (name, responses_df[subject_id,item_id,y,trial], items_df[item_id,content], subjects_df[subject_id,normalized_name])."""
    root = pathlib.Path(root)
    if (root / "items.parquet").exists() and (root / "subjects.parquet").exists():
        items_all = pd.read_parquet(root / "items.parquet")
        subs_all = pd.read_parquet(root / "subjects.parquet")
        name_col = "normalized_name" if "normalized_name" in subs_all.columns else "display_name"
        subs_all = subs_all.rename(columns={name_col: "normalized_name"})[["subject_id", "normalized_name"]]
        for p in sorted(root.glob("*.parquet")):
            if p.stem in ("items", "subjects", "benchmarks"):
                continue
            r = pd.read_parquet(p)
            if "response" not in r.columns:
                continue
            r = r[["subject_id", "item_id", "response"] + (["trial"] if "trial" in r.columns else [])].copy()
            it = items_all[items_all.benchmark_id == p.stem][["item_id", "content"]] if "benchmark_id" in items_all.columns else items_all[["item_id", "content"]]
            yield p.stem, r, it, subs_all
    else:
        for d in sorted(root.iterdir()):
            if (d / "response.parquet").exists():
                r = pd.read_parquet(d / "response.parquet"); it = pd.read_parquet(d / "items.parquet")[["item_id", "content"]]
                s = pd.read_parquet(d / "subjects.parquet")[["subject_id", "normalized_name"]]
                yield d.name, r, it, s


def fixed_effects(Y, M, iters=40):
    mu = float(np.nanmean(np.where(M, Y, np.nan)))
    theta = np.zeros(Y.shape[0]); ease = np.zeros(Y.shape[1])
    n_s = M.sum(1); n_i = M.sum(0)
    for _ in range(iters):
        res_i = np.where(M, Y - mu - theta[:, None], 0.0)
        ease = res_i.sum(0) / np.maximum(n_i + 3.0, 1.0); ease -= ease.mean()
        res_s = np.where(M, Y - mu - ease[None, :], 0.0)
        reg = np.where(n_s >= 35, 2.0, 14.0)
        theta = res_s.sum(1) / np.maximum(n_s + reg, 1.0); theta -= theta.mean()
    # discrimination proxy: slope of item outcome on subject ability
    disc = np.zeros(Y.shape[1])
    tv = np.where(M, theta[:, None], 0.0); tc = tv - np.where(M, tv.sum(0) / np.maximum(n_i, 1), 0.0)
    num = np.where(M, (Y - mu - ease[None, :]) * tc, 0.0).sum(0); den = np.where(M, tc ** 2, 0.0).sum(0) + 0.5
    disc = num / den
    return mu, theta, ease, disc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True); ap.add_argument("--out", default=str(ROOT / "localsim" / "artifacts_expanded"))
    ap.add_argument("--max-items-per-benchmark", type=int, default=4000)
    ap.add_argument("--frontier", nargs="*", default=["matharena", "hle", "livecodebench", "swebench", "cybench"])
    ap.add_argument("--min-items-per-subject", type=int, default=20)
    a = ap.parse_args()
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    m = load_model_helpers()
    sys.path.insert(0, str(ROOT))
    from generate_v38_subject_priors_and_simulate_alc import canonical_model_key, heuristic_architecture_prior, is_thinking_model  # noqa

    rows = []          # per (benchmark, subject) rows for priors
    feat_rows = []     # (benchmark, item vector, ease, disc) for the regressor
    centroids = {}
    rng = np.random.default_rng(0)
    for name, r, it, subs in load_benchmarks(a.root):
        r = r[pd.to_numeric(r["response"], errors="coerce").notna()].copy()
        r["y"] = (pd.to_numeric(r["response"]) >= 0.5).astype(np.float32)
        cell = r.groupby(["subject_id", "item_id"])["y"].mean().reset_index()
        piv = cell.pivot(index="subject_id", columns="item_id", values="y")
        piv = piv.loc[piv.notna().sum(1) >= a.min_items_per_subject]
        if piv.shape[0] < 5 or piv.shape[1] < 20:
            print(f"skip {name}: {piv.shape}"); continue
        Y = piv.values.astype(np.float64); M = ~np.isnan(Y); Y = np.nan_to_num(Y)
        mu, theta, ease, disc = fixed_effects(Y, M)
        smap = subs.set_index("subject_id")["normalized_name"].to_dict()
        n_items_s = M.sum(1); raw_means = (Y * M).sum(1) / np.maximum(n_items_s, 1)
        for k, sid in enumerate(piv.index):
            rname = str(smap.get(sid, sid)).strip()
            rows.append({"benchmark": name, "raw_name": rname.lower(), "canon_key": canonical_model_key(rname), "is_thinking": is_thinking_model(rname),
                         "n_items": int(n_items_s[k]), "raw_mean": float(raw_means[k]), "item_adj_delta": float(theta[k]), "arch_prior": heuristic_architecture_prior(rname)})
        # item features (text benchmarks only)
        content = it.set_index("item_id")["content"].astype(str).to_dict()
        texts = [content.get(i, "") for i in piv.columns]
        id_like = np.mean([bool(re.match(r"^[a-z0-9_]+:[0-9a-f]{8,}$", t)) or len(t) < 15 for t in texts])
        print(f"{name:16s} subjects={piv.shape[0]:4d} items={piv.shape[1]:6d} responses={int(M.sum()):8d} mean={mu:.3f} text={'no' if id_like > 0.5 else 'yes'}", flush=True)
        if id_like > 0.5:
            continue
        idx = np.arange(len(texts))
        if len(idx) > a.max_items_per_benchmark:
            idx = np.sort(rng.choice(idx, a.max_items_per_benchmark, replace=False))
        vecs = []
        for j in idx:
            v = m._get_item_feature_vec(texts[j])
            if v is None:
                v = np.zeros(781, dtype=np.float32)
            vecs.append(v)
        X = np.vstack(vecs).astype(np.float32)
        feat_rows.append((name, X, ease[idx].astype(np.float32), disc[idx].astype(np.float32)))
        cm = X[:, :384].mean(0) / 0.7071; cm /= (np.linalg.norm(cm) + 1e-12)
        centroids[name] = [round(float(v), 5) for v in cm]

    # ---- subject priors (same two-factor rule as v38, generalised frontier set)
    df = pd.DataFrame(rows)
    front = set(a.frontier)
    gen_vals, front_vals = [], []
    for r in df.itertuples():
        if r.benchmark == "matharena":
            gv, fv = r.item_adj_delta + (0.06 if r.is_thinking else 0.48), r.item_adj_delta
        elif r.benchmark in front:
            gv, fv = r.item_adj_delta + 0.04, r.item_adj_delta
        else:
            gv, fv = r.item_adj_delta, 0.65 * r.item_adj_delta + (0.05 if r.is_thinking else -0.03)
        gen_vals.append(gv); front_vals.append(fv)
    df["gen"] = gen_vals; df["front"] = front_vals
    priors = {}
    for key_col in ("canon_key", "raw_name"):
        for key, sub in df.groupby(key_col):
            w = np.minimum(sub["n_items"].values / 25.0, 1.0); arch = float(sub["arch_prior"].mean())
            tg = float((np.sum(w * sub["gen"].values) + 0.08 * arch) / (np.sum(w) + 0.08))
            tf = float((np.sum(w * sub["front"].values) + 0.08 * arch) / (np.sum(w) + 0.08))
            glob = 0.78 * tg + 0.22 * tf
            priors[key] = {"delta": round(float(np.clip(glob, -0.42, 0.42)), 4), "delta_general": round(float(np.clip(tg, -0.42, 0.42)), 4),
                           "delta_frontier": round(float(np.clip(tf, -0.42, 0.42)), 4), "mean": round(float(np.clip(0.5 + glob, 0.08, 0.92)), 4),
                           "n_bms": int(sub["benchmark"].nunique()), "bm_deltas": {b: round(float(v), 4) for b, v in zip(sub["benchmark"], sub["gen"])}}
    (out / "subject_priors.json").write_text(json.dumps(priors, separators=(",", ":")), encoding="utf-8")

    # ---- universal item regressor
    Xc = np.vstack([X - X.mean(0, keepdims=True) for _, X, _, _ in feat_rows]); ease_all = np.concatenate([e for _, _, e, _ in feat_rows]); disc_all = np.concatenate([d for _, _, _, d in feat_rows])
    I = np.eye(Xc.shape[1], dtype=np.float32)
    lam_e, lam_d = 35.0 * len(ease_all) / 1600.0, 55.0 * len(ease_all) / 1600.0  # keep the ridge strength per sample as in v38
    w_ease = np.linalg.solve(Xc.T @ Xc + lam_e * I, Xc.T @ ease_all); w_disc = np.linalg.solve(Xc.T @ Xc + lam_d * I, Xc.T @ disc_all)
    feat_mean = np.vstack([X for _, X, _, _ in feat_rows]).mean(0)
    (out / "universal_item_irt.json").write_text(json.dumps({"w_ease": [round(float(v), 6) for v in w_ease], "w_disc": [round(float(v), 6) for v in w_disc],
                                                             "feat_mean": [round(float(v), 6) for v in feat_mean], "centroids": centroids}, separators=(",", ":")), encoding="utf-8")
    np.savez(out / "cached_features.npz", X=np.vstack([X for _, X, _, _ in feat_rows]), bm=np.array([n for n, X, _, _ in feat_rows for _ in range(len(X))]), ease=ease_all, disc=disc_all)
    print(f"subjects (keys)={len(priors)} benchmarks={df.benchmark.nunique()} regressor rows={len(ease_all)} in-sample corr(ease)={np.corrcoef(Xc @ w_ease, ease_all)[0,1]:+.3f}")
    print("wrote", out / "subject_priors.json", out / "universal_item_irt.json")


if __name__ == "__main__":
    main()
