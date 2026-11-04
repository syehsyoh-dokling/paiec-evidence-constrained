"""Leave-One-Benchmark-Out local simulator for PAIEC submissions.

For each held-out training benchmark (fold):
  * patch the submission's fitted artifacts so they contain NO information
    from the held-out benchmark (subject_priors bm_deltas, universal IRT ridge
    weights / centroids / feature mean, raw parquet dir)
  * anonymise the benchmark id (as on Codabench)
  * build draws of 1-3 pairs (peers share the benchmark) with fixed 50/50
    acquisition/evaluation splits; identical across variants -> paired diffs
Usage:
  python sim_local.py --zips a.zip b.zip --pairs-per-fold 24 --seed 0 --tag run1
"""
import argparse
import hashlib
import json
import os
import pathlib
import random
import shutil
import subprocess
import sys
import uuid
import zipfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

ROOT = pathlib.Path(r"c:\Users\Saifuddin\Documents\LOMBA")
DATA = ROOT / "paiec_v31_ble_hybrid" / "data"
NPZ = ROOT / "cached_dual_encoder_1600.npz"
SIM = ROOT / "localsim"
BMS = ["afrimedqa", "agentdojo", "ai2d_test", "hle", "matharena", "mathvista_mini", "mmbench_v11", "mmlupro"]
GROUP_SIZES = [1, 2, 3, 2, 1, 2]


def h(s):
    return int(hashlib.sha256(s.encode()).hexdigest()[:12], 16)


def anon(bm):
    return "benchmark_" + str(h("anon-" + bm) % 900000 + 100000)


_BM_CACHE = {}


def load_bm(bm):
    if bm not in _BM_CACHE:
        d = DATA / bm
        resp = pd.read_parquet(d / "response.parquet")
        items = pd.read_parquet(d / "items.parquet").set_index("item_id")
        subs = pd.read_parquet(d / "subjects.parquet").set_index("subject_id")
        resp = resp[resp["response"].notna()].copy()
        resp["y"] = (resp["response"].astype(float) >= 0.5).astype(int)
        _BM_CACHE[bm] = (resp, items, subs)
    return _BM_CACHE[bm]


def make_pair(bm, sid, seed, score=True):
    """One subject-benchmark pair with a fixed 50/50 acquisition/evaluation split."""
    resp, items, subs = load_bm(bm)
    bid = anon(bm)

    def item_dict(iid):
        return {"item_content": str(items.loc[iid, "content"] or ""), "item_features": "",
                "interactors": "", "benchmark_id": bid}

    r = resp[resp.subject_id == sid]
    row = subs.loc[sid] if sid in subs.index else None
    subject = {"normalized_name": str(row["normalized_name"]) if row is not None else sid,
               "provider": "", "release_date": str(row["release_date"] or "") if row is not None else "",
               "access_date": "", "harness": "", "harness_version": "", "reasoning_effort": "",
               "subject_features_extra": ""}
    g = r.groupby("item_id")["y"].apply(list)
    iids = sorted(g.index.tolist(), key=lambda i: h(f"split-{sid}-{i}"))
    half = len(iids) // 2
    acq_ids, ev_ids = iids[:half], iids[half:]
    acq_ids = sorted(acq_ids, key=lambda i: h(f"order-{seed}-{sid}-{i}"))
    acq = []
    for i in acq_ids:
        labs = g[i]
        acq.append({"item": item_dict(i), "label": labs[h(f"trial-{seed}-{sid}-{i}") % len(labs)]})
    ev = [{"item": item_dict(i), "labels": g[i]} for i in ev_ids] if score else []
    return {"pair_id": f"{bm}|{sid}", "subject_id": "subject_" + str(h(sid) % 900000), "subject": subject,
            "acq": acq, "eval": ev, "score": score}


def build_spec(bm, pairs_per_fold, seed, context_pairs=0):
    """Fold spec. With context_pairs > 0, each draw also carries up to that many unscored
    pairs of the same subjects on OTHER public benchmarks (their labels enter `labeled`,
    as cross-benchmark pairs do on Codabench), so predictors that pool a subject's labels
    across benchmarks can be evaluated. hle is never used as a context benchmark."""
    resp, items, subs = load_bm(bm)
    cnt = resp.groupby("subject_id").item_id.nunique()
    elig = sorted(cnt[cnt >= 80].index.tolist())
    rng = random.Random(h(f"{bm}-{seed}"))
    rng.shuffle(elig)
    elig = elig[:pairs_per_fold]
    pairs = [make_pair(bm, sid, seed) for sid in elig]
    draws, i, k = [], 0, 0
    while i < len(pairs):
        gs = GROUP_SIZES[k % len(GROUP_SIZES)]
        group = pairs[i:i + gs]
        ctx = []
        if context_pairs:
            for p in group:
                sid = p["pair_id"].split("|", 1)[1]
                others = [b for b in BMS if b not in (bm, "hle")]
                others = sorted(others, key=lambda b: h(f"ctx-{seed}-{sid}-{b}"))
                added = 0
                for b2 in others:
                    r2, _, _ = load_bm(b2)
                    if r2[r2.subject_id == sid].item_id.nunique() >= 80:
                        ctx.append(make_pair(b2, sid, seed, score=False))
                        added += 1
                    if added >= context_pairs:
                        break
        draws.append({"pairs": group + ctx})
        i += gs
        k += 1
    return {"fold": bm, "draws": draws}


def patch_artifacts(sub_dir, bm):
    dd = sub_dir / "data"
    if (dd / bm).exists():
        shutil.rmtree(dd / bm)
    sp = dd / "subject_priors.json"
    if sp.exists():
        pri = json.loads(sp.read_text(encoding="utf-8"))
        out = {}
        for k, info in pri.items():
            bd = dict(info.get("bm_deltas") or {})
            if bm in bd:
                dbm = bd.pop(bm)
                n = int(info.get("n_bms", len(bd) + 1))
                if n <= 1 or not bd:
                    continue  # subject only known from held-out benchmark -> unseen
                old = float(info.get("delta", 0.0))
                new = (old * n - dbm) / (n - 1)
                shift = new - old
                info = dict(info)
                info["delta"] = round(new, 4)
                for f in ("delta_general", "delta_frontier"):
                    if f in info:
                        info[f] = round(float(info[f]) + shift, 4)
                if "mean" in info:
                    info["mean"] = round(float(info["mean"]) + shift, 4)
                info["n_bms"] = n - 1
                info["bm_deltas"] = bd
            out[k] = info
        sp.write_text(json.dumps(out), encoding="utf-8")
    up = dd / "universal_item_irt.json"
    if up.exists():
        u = json.loads(up.read_text(encoding="utf-8"))
        exp = dd / "expanded_features.npz"
        if exp.exists():
            # expanded training set shipped with the submission: refit without the held-out benchmark
            z = np.load(exp, allow_pickle=True)
            X, bma, dg, sg = z["X"], z["bm"], z["ease"], z["disc"]
            keep = sorted({b for b in bma.tolist() if b != bm})
        else:
            z = np.load(NPZ, allow_pickle=True)
            X, bma, dg, sg = z["X_dual"], z["bm_arr"], z["delta_gt"], z["disc_gt"]
            keep = [b for b in BMS if b != bm]
        Xc = np.vstack([X[bma == b] - X[bma == b].mean(0, keepdims=True) for b in keep])
        m = np.isin(bma, keep)
        # rows ordered by `keep`, so reorder targets identically
        dgk = np.concatenate([dg[bma == b] for b in keep])
        sgk = np.concatenate([sg[bma == b] for b in keep])
        I = np.eye(Xc.shape[1], dtype=np.float32)
        u["w_ease"] = np.linalg.solve(Xc.T @ Xc + 35.0 * I, Xc.T @ dgk).round(6).tolist()
        u["w_disc"] = np.linalg.solve(Xc.T @ Xc + 55.0 * I, Xc.T @ sgk).round(6).tolist()
        u["feat_mean"] = X[m].mean(0).round(6).tolist()
        if "centroids" in u:
            u["centroids"].pop(bm, None)
        up.write_text(json.dumps(u, separators=(",", ":")), encoding="utf-8")


def run_variant_fold(zip_path, bm, spec_path, tag):
    name = pathlib.Path(zip_path).stem
    out = SIM / "results" / tag / f"{name}__{bm}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        r = json.load(open(out))
        print(f"  skip {name:55s} {bm:15s} (cached, pairs={len(r['results'])})", flush=True)
        return r
    # unique work dir per process: never rmtree a dir a stale worker may still hold
    wd = SIM / "work" / tag / f"{name}__{bm}__{os.getpid()}_{uuid.uuid4().hex[:6]}"
    wd.mkdir(parents=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(wd)
    patch_artifacts(wd, bm)
    env = dict(os.environ, PAIEC_DISABLE_BLE_NETWORK="1", PYTHONIOENCODING="utf-8")
    try:
        p = subprocess.run([sys.executable, str(SIM / "sim_worker.py"), str(wd), str(spec_path), str(out)],
                           capture_output=True, text=True, encoding="utf-8", errors="replace", env=env,
                           timeout=int(os.environ.get("SIM_WORKER_TIMEOUT", "2400")))
    except subprocess.TimeoutExpired:
        print(f"FAIL {name} {bm}: worker exceeded {os.environ.get('SIM_WORKER_TIMEOUT', '2400')} s and was killed", flush=True)
        return None
    if p.returncode != 0:
        print(f"FAIL {name} {bm}\n{p.stderr[-3000:]}", flush=True)
        return None
    r = json.load(open(out))
    print(f"  done {name:55s} {bm:15s} pairs={len(r['results']):3d} secs={r['secs']:.0f}", flush=True)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zips", nargs="+", required=True)
    ap.add_argument("--pairs-per-fold", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--folds", nargs="*", default=BMS)
    ap.add_argument("--tag", default="run")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--context-pairs", type=int, default=0, help="unscored same-subject pairs from other benchmarks per scored pair")
    a = ap.parse_args()
    specs = {}
    for bm in a.folds:
        suffix = f"_c{a.context_pairs}" if a.context_pairs else ""
        sp = SIM / "specs" / f"{bm}_s{a.seed}_p{a.pairs_per_fold}{suffix}.json"
        sp.parent.mkdir(parents=True, exist_ok=True)
        if not sp.exists():
            sp.write_text(json.dumps(build_spec(bm, a.pairs_per_fold, a.seed, a.context_pairs)), encoding="utf-8")
        specs[bm] = sp
    jobs = [(z, bm) for z in a.zips for bm in a.folds]
    with ThreadPoolExecutor(a.jobs) as ex:
        futs = {(z, bm): ex.submit(run_variant_fold, z, bm, specs[bm], a.tag) for z, bm in jobs}
    res = {}
    for (z, bm), f in futs.items():
        r = f.result()
        if r:
            for row in r["results"]:
                res.setdefault(pathlib.Path(z).stem, {})[row["pair"]] = row
    json.dump(res, open(SIM / "results" / a.tag / "combined.json", "w"))
    report(res)


def report(res):
    names = list(res)
    common = sorted(set.intersection(*[set(v) for v in res.values()]))
    base = names[0]
    print(f"\nCommon pairs: {len(common)}  (baseline = {base})")
    print(f"{'variant':55s} {'meanALC':>8s} {'diff':>8s} {'SE':>7s} {'wins':>9s}  B0     B1     B3     B7     B15    B31")
    for n in names:
        a = np.array([res[n][p]["alc"] for p in common])
        b = np.array([res[base][p]["alc"] for p in common])
        d = a - b
        se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else 0
        bud = np.mean([res[n][p]["briers"] for p in common], axis=0)
        print(f"{n[:55]:55s} {a.mean():8.4f} {d.mean():+8.4f} {se:7.4f} {int((d < 0).sum()):4d}/{len(d):<4d} " +
              " ".join(f"{x:.4f}" for x in bud))
    print("\nPer-fold mean ALC:")
    folds = sorted({p.split("|")[0] for p in common})
    print(f"{'variant':40s} " + " ".join(f"{f[:9]:>9s}" for f in folds))
    for n in names:
        print(f"{n[:40]:40s} " + " ".join(
            f"{np.mean([res[n][p]['alc'] for p in common if p.startswith(f + '|')]):9.4f}" for f in folds))


if __name__ == "__main__":
    main()
