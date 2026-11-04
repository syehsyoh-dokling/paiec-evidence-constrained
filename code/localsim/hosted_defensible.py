"""What hosted score can be defended scientifically? Per model family: n, mean, median, SD,
standard error, 95% CI of the mean, best draw; plus a paired per-benchmark comparison of
today's uploads against the byte-identical controls using the permanent anonymous
benchmark IDs in the scoring logs."""
import os
import collections
import json
import math
import pathlib
import re
import statistics

ROOT = pathlib.Path(os.environ.get("PAIEC_ROOT", r"c:\Users\Saifuddin\Documents\LOMBA"))
res = json.load(open(ROOT / "localsim" / "results" / "hosted_scores.json"))
scored = {int(k): v for k, v in res.items() if v.get("score") is not None}


def family(v):
    f = (v.get("file") or "").lower(); b = v.get("batch") or ""
    if b == "pre_lobo":
        m = re.search(r"_v(\d+)", f); ver = int(m.group(1)) if m else 0
        return "pre-LOBO v146-v337 (v153 family)" if ver >= 146 else "pre-LOBO earlier generations"
    if "control" in f or (v.get("config") == "control"):
        return "control (byte-identical v153)"
    if b in ("quota_v402_20261104", "quota_day3_20261105"):
        return "v402b (logistic IRT)"
    if b == "quota_v401_20261104":
        return "v401 (b0fix + mixture)"
    if "b0" in f or "rnd" in f or "qwen" in b or "random" in b or "pending" in b or "quota_17" in b or "day2" in b:
        return "budget-0 revision family (student + sampler + grid)"
    return b


groups = collections.defaultdict(list)
for sid, v in scored.items():
    groups[family(v)].append(v["score"])
print(f"{'family':48s} {'n':>4s} {'mean':>7s} {'median':>7s} {'sd':>7s} {'se':>7s} {'95% CI of mean':>18s} {'best':>7s}")
for g, xs in sorted(groups.items(), key=lambda kv: -len(kv[1])):
    n = len(xs); m = statistics.mean(xs); sd = statistics.stdev(xs) if n > 1 else float("nan"); se = sd / math.sqrt(n) if n > 1 else float("nan")
    print(f"{g:48s} {n:4d} {m:7.4f} {statistics.median(xs):7.4f} {sd:7.4f} {se:7.4f} [{m-1.96*se:7.4f}, {m+1.96*se:7.4f}] {min(xs):7.4f}")

# paired per-benchmark comparison from scoring logs
row_re = re.compile(r"^(subject_\d+)\s+(benchmark_\d+)\s+(\d+)\s+([0-9.]+)\s+([0-9.]+)\s*$")
def pairs_of(sid):
    p = ROOT / "scoring_logs_all" / f"{sid}.txt"
    if not p.exists():
        return {}
    out = {}; sec = "alc"
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        m = re.match(r"^Label budget:\s*(\d+)", line)
        if m:
            sec = int(m.group(1)); continue
        m = row_re.match(line.strip())
        if m and sec == "alc":
            out[m.group(2)] = out.get(m.group(2), []) + [float(m.group(4))]
    return {b: statistics.mean(v) for b, v in out.items()}
bench = collections.defaultdict(lambda: collections.defaultdict(list))
for sid, v in scored.items():
    for b, alc in pairs_of(sid).items():
        bench[family(v)][b].append(alc)
fams = [f for f in groups if f in bench]
benches = sorted({b for f in fams for b in bench[f]})
if benches:
    print("\nPer hidden benchmark, mean pair ALC by family (n pairs):")
    print(f"{'benchmark':18s} " + " ".join(f"{f[:22]:>24s}" for f in fams))
    for b in benches:
        print(f"{b:18s} " + " ".join((f"{statistics.mean(bench[f][b]):.4f} ({len(bench[f][b]):3d})" if bench[f].get(b) else f"{'-':>12s}").rjust(24) for f in fams))
