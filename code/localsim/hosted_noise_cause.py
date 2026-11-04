"""Why is one hosted score 0.1276 while the family mean is 0.186? Decompose the per-pair ALC
variance into benchmark composition, subject composition and residual, using every scored log."""
import os
import collections, json, math, pathlib, re, statistics
ROOT = pathlib.Path(os.environ.get("PAIEC_ROOT", r"C:\Users\Saifuddin\Documents\LOMBA"))
res = json.load(open(ROOT / "localsim" / "results" / "hosted_scores.json"))
row_re = re.compile(r"^(subject_\d+)\s+(benchmark_\d+)\s+(\d+)\s+([0-9.]+)\s+([0-9.]+)\s*$")

def pairs_of(sid):
    p = ROOT / "scoring_logs_all" / f"{sid}.txt"
    if not p.exists():
        return []
    out = []; sec = "alc"
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        m = re.match(r"^Label budget:\s*(\d+)", line)
        if m:
            sec = int(m.group(1)); continue
        m = row_re.match(line.strip())
        if m and sec == "alc":
            out.append((m.group(1), m.group(2), int(m.group(3)), float(m.group(4))))
    return out

subs = {}
for sid, v in res.items():
    if v.get("score") is None:
        continue
    pr = pairs_of(sid)
    if pr:
        subs[int(sid)] = (v["score"], pr)
allp = [x for s in subs.values() for x in s[1]]
n_per = [len(s[1]) for s in subs.values()]
alc = [x[3] for x in allp]
print(f"submissions with logs {len(subs)}, pairs per submission mean {statistics.mean(n_per):.1f}")
sd_pair = statistics.pstdev(alc)
print(f"pair-level ALC: mean {statistics.mean(alc):.4f} SD {sd_pair:.4f} min {min(alc):.4f} max {max(alc):.4f}")
print(f"predicted SD of a {statistics.mean(n_per):.0f}-pair mean = {sd_pair/math.sqrt(statistics.mean(n_per)):.4f}; observed SD of submission scores = {statistics.pstdev([s[0] for s in subs.values()]):.4f}")
# variance decomposition
by_b = collections.defaultdict(list); by_pair = collections.defaultdict(list); by_s = collections.defaultdict(list)
for s, b, n, a in allp:
    by_b[b].append(a); by_pair[(s, b)].append(a); by_s[s].append(a)
gm = statistics.mean(alc)
var_tot = statistics.pvariance(alc)
var_b = sum(len(v) * (statistics.mean(v) - gm) ** 2 for v in by_b.values()) / len(alc)
var_pair = sum(len(v) * (statistics.mean(v) - gm) ** 2 for v in by_pair.values()) / len(alc)
print(f"variance of pair ALC {var_tot:.5f}: benchmark identity explains {var_b/var_tot:.0%}, (subject,benchmark) identity explains {var_pair/var_tot:.0%}, residual (same pair, different submissions/items) {1-var_pair/var_tot:.0%}")
print(f"distinct subjects {len(by_s)}, distinct pairs {len(by_pair)}, pairs seen >=3 times {sum(1 for v in by_pair.values() if len(v)>=3)}")
print("benchmark   n   mean   sd(within)")
for b, v in sorted(by_b.items(), key=lambda kv: statistics.mean(kv[1])):
    print(f"{b} {len(v):4d} {statistics.mean(v):.4f} {statistics.pstdev(v):.4f}")
# the 0.1276 draw
sid = 1020778
sc, pr = subs[sid]
print(f"\nsubmission {sid}: score {sc:.4f}")
print("subject          benchmark         n    ALC    bench mean  pair mean elsewhere (n)")
exp_b = []; exp_p = []
for s, b, n, a in pr:
    others = [x for x in by_pair[(s, b)]]
    # exclude this very observation
    others.remove(a)
    pm = statistics.mean(others) if others else float("nan")
    exp_b.append(statistics.mean(by_b[b])); exp_p.append(pm if others else statistics.mean(by_b[b]))
    print(f"{s} {b} {n:3d} {a:.4f}   {statistics.mean(by_b[b]):.4f}     {pm:.4f} ({len(others)})")
print(f"expected score from its benchmark composition alone: {statistics.mean(exp_b):.4f}")
print(f"expected score from its (subject,benchmark) composition (other submissions' results on the same pairs): {statistics.mean(exp_p):.4f}")
print(f"actual: {sc:.4f}; family mean 0.1859")
# how many submissions got an equally favourable draw? rank of this score
scores = sorted(s[0] for s in subs.values())
print(f"rank of {sc:.4f} among {len(scores)} scored submissions with logs: {scores.index(sc)+1}")
# what does the same draw's expected value look like for the 5 best scores
print("\n5 best hosted scores and their pair-composition expectation:")
for sid2, (sc2, pr2) in sorted(subs.items(), key=lambda kv: kv[1][0])[:5]:
    e = []
    for s, b, n, a in pr2:
        o = list(by_pair[(s, b)]); o.remove(a)
        e.append(statistics.mean(o) if o else statistics.mean(by_b[b]))
    print(f"  {sid2} {res[str(sid2)].get('file','')[:40]:40s} score {sc2:.4f} expected-from-draw {statistics.mean(e):.4f}")
