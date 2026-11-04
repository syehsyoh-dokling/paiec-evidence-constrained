"""Budget-0 LOBO evaluation for one held-out fold.

Budget-0 predictions never see labels, so the whole grid of alternative B0
formulas can be scored from two quantities per (subject, item):
    m = subject prior mean from data/subject_priors.json (patched, LOBO)
    d = zero-shot 2PL item ease from the patched universal ridge
Alternative:  p = clip(c + s * (m - 0.5) + e * d, 0.02, 0.98)
Also scores v153's own B0 output, constant 0.5, and the per-pair oracle floor.
Usage: python b0_eval_fold.py <fold> <out.json> [candidate.zip]
With a candidate ZIP, the "v153" entry holds that candidate's own B0 output instead.
"""
import copy
import importlib.util
import json
import os
import pathlib
import shutil
import sys
import zipfile

import numpy as np

ROOT = pathlib.Path(os.environ.get("PAIEC_ROOT", r"c:\Users\Saifuddin\Documents\LOMBA"))
SIM = ROOT / "localsim"
sys.path.insert(0, str(SIM))
import sim_local  # noqa: E402

SRC_ZIP = SIM / "variants" / "v153_nokeys.zip"
C_GRID = [0.45, 0.50, 0.55]
S_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]
E_GRID = [0.0, 0.238, 0.5, 0.75, 1.0]


def main():
    bm, out_path = sys.argv[1], str(pathlib.Path(sys.argv[2]).resolve())
    src_zip = pathlib.Path(sys.argv[3]) if len(sys.argv) > 3 else SRC_ZIP
    wd = SIM / "work" / "b0eval" / f"{src_zip.stem}__{bm}__{os.getpid()}"
    wd.mkdir(parents=True)
    with zipfile.ZipFile(src_zip) as zf:
        zf.extractall(wd)
    sim_local.patch_artifacts(wd, bm)

    os.environ["PAIEC_DISABLE_BLE_NETWORK"] = "1"
    os.environ["PREDICTIVE_EVAL_LOCAL_SMOKE_TEST"] = "1"
    sys.path.insert(0, str(wd))
    os.chdir(wd)
    spec = importlib.util.spec_from_file_location("model", str(wd / "model.py"))
    model = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model)

    fold = sim_local.build_spec(bm, 10**6, 0)
    pairs = [p for d in fold["draws"] for p in d["pairs"]]
    rows = []  # per pair: dict of arrays
    for p in pairs:
        subj = p["subject"]
        ys, pv, ms, ds, ns = [], [], [], [], []
        for ev in p["eval"]:
            inp = [copy.deepcopy(subj), copy.deepcopy(ev["item"])]
            p_v153 = float(model.predict(inp, []))
            bid = ev["item"]["benchmark_id"]
            text = ev["item"]["item_content"]
            info = model._lookup_db_subject_info(subj)
            m = float(info.get("mean", 0.5)) if info else 0.5
            theta = model._get_subject_domain_theta(subj, bm_id=bid, est_p_bench=None)
            d = float(model._predict_2pl_item_ease(text, theta, bm_id=bid))
            labs = ev["labels"]
            ys.append(float(np.mean(labs)))
            ns.append(len(labs))
            pv.append(p_v153)
            ms.append(m)
            ds.append(d)
        rows.append({"pair": p["pair_id"], "known": bool(info), "y": ys, "n": ns, "pv": pv, "m": ms, "d": ds})

    def brier(pred, y, n):
        # mean over all responses: with k trials of mean y, sum (p-y_t)^2 = k*[(p-y)^2 + y(1-y)]
        pred = np.asarray(pred); y = np.asarray(y); n = np.asarray(n)
        return float(np.sum(n * ((pred - y) ** 2 + y * (1 - y))) / np.sum(n))

    res = {"fold": bm, "n_pairs": len(rows), "known_frac": float(np.mean([r["known"] for r in rows])), "configs": {}}
    per_pair = {}
    for r in rows:
        y, n, m, d = (np.asarray(r[k]) for k in ("y", "n", "m", "d"))
        per_pair.setdefault("v153", []).append(brier(r["pv"], y, n))
        per_pair.setdefault("const_0.5", []).append(brier(np.full_like(y, 0.5), y, n))
        pbar = float(np.sum(n * y) / np.sum(n))
        per_pair.setdefault("oracle_pair_const", []).append(brier(np.full_like(y, pbar), y, n))
        for c in C_GRID:
            for s in S_GRID:
                for e in E_GRID:
                    pred = np.clip(c + s * (m - 0.5) + e * d, 0.02, 0.98)
                    per_pair.setdefault(f"c{c:.2f}_s{s:.2f}_e{e:.3f}", []).append(brier(pred, y, n))
    for k, v in per_pair.items():
        res["configs"][k] = {"mean_b0": float(np.mean(v)), "per_pair": v}
    res["pairs"] = [r["pair"] for r in rows]
    json.dump(res, open(out_path, "w"))
    print(f"{bm}: pairs={len(rows)} known={res['known_frac']:.2f} v153={res['configs']['v153']['mean_b0']:.4f} "
          f"const0.5={res['configs']['const_0.5']['mean_b0']:.4f} oracle={res['configs']['oracle_pair_const']['mean_b0']:.4f}")


if __name__ == "__main__":
    main()
