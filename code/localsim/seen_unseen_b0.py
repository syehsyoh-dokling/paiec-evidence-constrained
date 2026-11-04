"""Within-benchmark generalisation at budget 0: for each of the eight training benchmarks,
compare the budget-0 Brier of a predictor on the 200 items its artifacts were fitted on
("seen") against items of the same benchmark that were never used to fit anything
("unseen", from the full measurement-db tables), for the same subjects.
Usage: python seen_unseen_b0.py <mirror_root> <out.json> <zip> [<zip> ...]
Per subject: up to 60 seen and 60 unseen items, interleaved, Brier at budget 0 (labeled=[]).
"""
import copy, hashlib, importlib.util, json, os, pathlib, random, shutil, sys, zipfile
import numpy as np, pandas as pd
ROOT = pathlib.Path(r"c:\Users\Saifuddin\Documents\LOMBA")
SIM = ROOT / "localsim"
sys.path.insert(0, str(SIM))
import sim_local  # noqa: E402

mirror, out_path = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]).resolve()
zips = [pathlib.Path(z).resolve() for z in sys.argv[3:]]
os.environ["PAIEC_DISABLE_BLE_NETWORK"] = "1"
os.environ["PREDICTIVE_EVAL_LOCAL_SMOKE_TEST"] = "1"
items_all = pd.read_parquet(mirror / "items.parquet")
subs_all = pd.read_parquet(mirror / "subjects.parquet").set_index("subject_id")
MAX_PER = 60


def load_model(zp):
    wd = SIM / "work" / "seen_unseen" / f"{zp.stem}__{os.getpid()}"
    if wd.exists():
        shutil.rmtree(wd)
    wd.mkdir(parents=True)
    with zipfile.ZipFile(zp) as zf:
        zf.extractall(wd)
    sys.path.insert(0, str(wd)); os.chdir(wd)
    spec = importlib.util.spec_from_file_location("model", str(wd / "model.py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m, wd


results = {}
for zp in zips:
    model, wd = load_model(zp)
    per_bm = {}
    for bm in sim_local.BMS:
        if bm == "hle":
            continue
        resp_seen, items_seen, subs_seen = sim_local.load_bm(bm)
        seen_ids = set(items_seen.index.astype(str))
        full = pd.read_parquet(mirror / f"{bm}.parquet")
        full = full[full["response"].notna()].copy()
        full["y"] = (full["response"].astype(float) >= 0.5).astype(int)
        full["item_id"] = full["item_id"].astype(str)
        it_bm = items_all[items_all["benchmark_id"].astype(str).str.lower().str.contains(bm.split("_")[0][:6])]
        content = dict(zip(it_bm["item_id"].astype(str), it_bm["content"].astype(str)))
        content.update({str(i): str(items_seen.loc[i, "content"] or "") for i in items_seen.index})
        bid = sim_local.anon(bm)
        rows = []
        subjects = [s for s in full["subject_id"].unique() if s in subs_seen.index]
        rng = random.Random(7)
        for sid in subjects:
            r = full[full.subject_id == sid]
            g = r.groupby("item_id")["y"].mean()
            seen = [i for i in g.index if i in seen_ids and i in content]
            unseen = [i for i in g.index if i not in seen_ids and i in content and content[i].strip()]
            if len(seen) < 20 or len(unseen) < 20:
                continue
            rng.shuffle(seen); rng.shuffle(unseen)
            seen, unseen = seen[:MAX_PER], unseen[:MAX_PER]
            row = subs_seen.loc[sid]
            subject = {"normalized_name": str(row["normalized_name"]), "provider": "", "release_date": str(row.get("release_date") or ""),
                       "access_date": "", "harness": "", "harness_version": "", "reasoning_effort": "", "subject_features_extra": ""}
            order = [("seen", i) for i in seen] + [("unseen", i) for i in unseen]
            rng.shuffle(order)
            bs = {"seen": [], "unseen": []}
            for kind, i in order:
                item = {"item_content": content[i], "item_features": "", "interactors": "", "benchmark_id": bid}
                p = float(model.predict([copy.deepcopy(subject), item], []))
                bs[kind].append((p - float(g[i])) ** 2)
            rows.append({"subject": sid, "n_seen": len(seen), "n_unseen": len(unseen), "b0_seen": float(np.mean(bs["seen"])), "b0_unseen": float(np.mean(bs["unseen"]))})
        if rows:
            ds = np.array([r["b0_seen"] for r in rows]); du = np.array([r["b0_unseen"] for r in rows])
            per_bm[bm] = {"subjects": len(rows), "b0_seen": float(ds.mean()), "b0_unseen": float(du.mean()),
                          "diff_unseen_minus_seen": float((du - ds).mean()), "se_diff": float((du - ds).std(ddof=1) / np.sqrt(len(rows))) if len(rows) > 1 else None, "rows": rows}
            print(f"{zp.stem:28s} {bm:16s} subjects={len(rows):3d} B0 seen={ds.mean():.4f} unseen={du.mean():.4f} diff={(du-ds).mean():+.4f} se={per_bm[bm]['se_diff']}", flush=True)
    results[zp.stem] = per_bm
    os.chdir(ROOT); shutil.rmtree(wd, ignore_errors=True)
    sys.path.pop(0)
    if "model" in sys.modules:
        del sys.modules["model"]
json.dump(results, open(out_path, "w"), indent=1)
for z, per in results.items():
    d = [v["diff_unseen_minus_seen"] for v in per.values()]
    print(f"SUMMARY {z}: benchmarks={len(d)} mean diff (unseen - seen) = {np.mean(d):+.4f}, SE over benchmarks = {np.std(d, ddof=1)/np.sqrt(len(d)):.4f}")
