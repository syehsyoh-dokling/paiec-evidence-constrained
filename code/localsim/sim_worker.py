"""Worker: run one submission directory on one held-out-benchmark fold.

Emulates Codabench `streaming_alc_v1`:
  * per pair: fixed 50/50 acquisition/evaluation item split (provided in spec)
  * candidates from the acquisition pool arrive in a fixed random order
  * acquisition_function decides (or uniform random if no labeling.py)
  * nested label sets at budgets 0,1,3,7,15,31; `labeled` contains
    [[subject, item], label] entries across all pairs of the draw
  * predict() called once per distinct eval input per budget
  * Brier per pair = mean over all eval responses; ALC weights 0.1/0.2x4/0.1
Usage: python sim_worker.py <submission_dir> <spec.json> <out.json>
"""
import copy
import hashlib
import importlib.util
import json
import os
import sys
import time

BUDGETS = [0, 1, 3, 7, 15, 31]
WEIGHTS = [0.1, 0.2, 0.2, 0.2, 0.2, 0.1]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    sub_dir, spec_path, out_path = sys.argv[1:4]
    os.environ["PAIEC_DISABLE_BLE_NETWORK"] = "1"
    os.environ["PREDICTIVE_EVAL_LOCAL_SMOKE_TEST"] = "1"
    sys.path.insert(0, sub_dir)
    os.chdir(sub_dir)
    model = load_module("model", os.path.join(sub_dir, "model.py"))
    acq_fn = None
    if os.path.exists(os.path.join(sub_dir, "labeling.py")):
        acq_fn = load_module("labeling", os.path.join(sub_dir, "labeling.py")).acquisition_function
    sys.path.insert(0, r"c:\Users\Saifuddin\Documents\LOMBA\paiec_baseline\tools")
    from streaming_ingestion import AcquisitionHook

    hook = AcquisitionHook(acq_fn)
    spec = json.load(open(spec_path, encoding="utf-8"))
    results = []
    t0 = time.time()
    for draw in spec["draws"]:
        pairs = draw["pairs"]
        labeled = []
        acq_index = []  # parallel to labeled: (pair_idx, k-th label of that pair)
        n_lab = [0] * len(pairs)
        streams = [list(p["acq"]) for p in pairs]
        pos = [0] * len(pairs)
        active = True
        while active:
            active = False
            for pi, p in enumerate(pairs):
                if n_lab[pi] >= 31 or pos[pi] >= len(streams[pi]):
                    continue
                active = True
                cand = streams[pi][pos[pi]]
                pos[pi] += 1
                inp = [p["subject"], cand["item"]]
                ctx = {
                    "subject_id": p["subject_id"],
                    "benchmark_id": cand["item"]["benchmark_id"],
                    "labels_acquired": n_lab[pi],
                    "labels_remaining": 31 - n_lab[pi],
                    "max_labels": 31,
                    "items_remaining": len(streams[pi]) - pos[pi] + 1,
                }
                if hook.mode == "stream":
                    prob = None
                    if hook.needs_prediction:
                        prob = float(model.predict(copy.deepcopy(inp), copy.deepcopy(labeled)))
                    query = hook(inp, prob, labeled, ctx)
                else:
                    key = json.dumps([ctx, inp], sort_keys=True, separators=(",", ":"))
                    u = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big") / 2**64
                    query = u < min(1, ctx["labels_remaining"] / ctx["items_remaining"])
                if query:
                    labeled.append([[p["subject"], cand["item"]], int(cand["label"])])
                    acq_index.append((pi, n_lab[pi]))
                    n_lab[pi] += 1
        for pi, p in enumerate(pairs):
            briers = []
            for B in BUDGETS:
                lab_B = [e for e, (pj, k) in zip(labeled, acq_index) if k < B]
                se, n = 0.0, 0
                for ev in p["eval"]:
                    pr = float(model.predict([copy.deepcopy(p["subject"]), copy.deepcopy(ev["item"])], copy.deepcopy(lab_B)))
                    pr = min(max(pr, 0.0), 1.0)
                    for y in ev["labels"]:
                        se += (pr - y) ** 2
                        n += 1
                briers.append(se / max(n, 1))
            alc = sum(w * b for w, b in zip(WEIGHTS, briers))
            results.append({"pair": p["pair_id"], "group_size": len(pairs), "briers": briers, "alc": alc})
    json.dump({"results": results, "secs": time.time() - t0}, open(out_path, "w"))


if __name__ == "__main__":
    main()
