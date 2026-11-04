"""Paired comparison of every variant in <results>/<tag>/combined.json against a reference variant.
Usage: python paired_vs.py <tag> <reference_variant> [variant ...]
Prints dALC, pair-level SE, fold-level SE, folds improved, and per-budget Brier deltas."""
import os
import json, math, pathlib, statistics, sys
from collections import defaultdict

tag, ref = sys.argv[1], sys.argv[2]
d = pathlib.Path(os.environ.get("PAIEC_ROOT", r"c:\Users\Saifuddin\Documents\LOMBA")) / "localsim" / "results" / tag
res = json.load(open(d / "combined.json"))
want = sys.argv[3:] or [v for v in res if v != ref]
BUD = ["0", "1", "3", "7", "15", "31"]


def budget_brier(row, b):
    if isinstance(row.get("briers"), list) and len(row["briers"]) == 6:
        return row["briers"][BUD.index(b)]
    for k in ("brier", "per_budget", "budget_brier"):
        if k in row and isinstance(row[k], dict) and b in row[k]:
            return row[k][b]
    return None


r0 = res[ref]
print(f"reference {ref}: pairs {len(r0)}  ALC {statistics.mean(x['alc'] for x in r0.values()):.4f}")
hdr = f"{'variant':34s} {'ALC':>7s} {'dALC':>8s} {'SEp':>7s} {'SEf':>7s} folds " + " ".join(f"{'dB'+b:>8s}" for b in BUD)
print(hdr)
for v in want:
    rv = res.get(v)
    if not rv:
        print(v, "missing"); continue
    common = [p for p in rv if p in r0]
    diffs = [rv[p]["alc"] - r0[p]["alc"] for p in common]
    byfold = defaultdict(list)
    for p in common:
        byfold[rv[p].get("fold") or rv[p].get("bm") or p.split("|")[0]].append(rv[p]["alc"] - r0[p]["alc"])
    fm = [statistics.mean(x) for x in byfold.values()]
    sep = statistics.stdev(diffs) / math.sqrt(len(diffs)) if len(diffs) > 1 else float("nan")
    sef = statistics.stdev(fm) / math.sqrt(len(fm)) if len(fm) > 1 else float("nan")
    dbs = []
    for b in BUD:
        xs = [(budget_brier(rv[p], b), budget_brier(r0[p], b)) for p in common]
        xs = [(a, c) for a, c in xs if a is not None and c is not None]
        dbs.append(statistics.mean(a - c for a, c in xs) if xs else float("nan"))
    print(f"{v:34s} {statistics.mean(rv[p]['alc'] for p in common):7.4f} {statistics.mean(diffs):+8.4f} {sep:7.4f} {sef:7.4f} {sum(1 for x in fm if x < 0)}/{len(fm)}  " + " ".join(f"{x:+8.4f}" for x in dbs))
