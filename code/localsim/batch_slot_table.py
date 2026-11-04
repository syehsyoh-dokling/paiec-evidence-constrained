"""Tabulate every autopilot batch: candidates and PASS_UPLOAD verdicts per edit slot, from the
AUTOPILOT_REPORT.md files under localsim/results/qa_*/. Also Fisher exact test of PASS rate,
student batches vs no-LLM sampler batches, per slot and pooled."""
import collections, pathlib, re
from math import comb
R = pathlib.Path(r"c:\Users\Saifuddin\Documents\LOMBA\localsim\results")
SLOT = [("control", "control"), ("_b0", "B0"), ("low_budget", "LOW_BUDGET_SHRINK"), ("peer_shift", "PEER_SHIFT"), ("mixture", "MIXTURE_PRIOR"), ("irt", "IRT_ENVELOPE")]
def slot_of(name):
    n = name.lower()
    for k, s in SLOT:
        if k in n:
            return s
    return "other"
tab = {}
for rep in sorted(R.glob("qa_*/AUTOPILOT_REPORT.md")):
    batch = rep.parent.name[3:]
    c = collections.defaultdict(lambda: [0, 0])
    for line in rep.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^\|\s*(\S+\.zip)\s*\|\s*\w+\s*\|\s*(\w+)\s*\|", line)
        if m:
            s = slot_of(m.group(1)); c[s][0] += 1; c[s][1] += m.group(2) == "PASS_UPLOAD"
    tab[batch] = dict(c)
slots = ["B0", "LOW_BUDGET_SHRINK", "PEER_SHIFT", "MIXTURE_PRIOR", "IRT_ENVELOPE", "control", "other"]
print(f"{'batch':16s} " + " ".join(f"{s[:10]:>10s}" for s in slots) + "   total")
for b, c in tab.items():
    tot = [sum(v[0] for v in c.values()), sum(v[1] for v in c.values())]
    print(f"{b:16s} " + " ".join(f"{c[s][1]}/{c[s][0]}".rjust(10) if s in c else " " * 10 for s in slots) + f"   {tot[1]}/{tot[0]}")
def fisher_two_sided(a, b, c, d):
    # table [[a,b],[c,d]]: a PASS of (a+b) in group 1, c PASS of (c+d) in group 2
    n = a + b + c + d; r1 = a + b; c1 = a + c
    def p(x): return comb(r1, x) * comb(n - r1, c1 - x) / comb(n, c1)
    p0 = p(a)
    return sum(p(x) for x in range(max(0, c1 - (n - r1)), min(r1, c1) + 1) if p(x) <= p0 + 1e-12)
student = [b for b in tab if "qwen" in b]; sampler = [b for b in tab if "random" in b]
print("\nstudent batches:", student, "| sampler batches:", sampler)
for s in slots[:5] + ["ALL non-control"]:
    def agg(bs):
        if s == "ALL non-control":
            return [sum(tab[b][k][i] for b in bs for k in tab[b] if k != "control") for i in (0, 1)]
        return [sum(tab[b].get(s, [0, 0])[i] for b in bs) for i in (0, 1)]
    (n1, p1), (n2, p2) = agg(student), agg(sampler)
    if n1 and n2:
        print(f"{s:18s} student {p1}/{n1} ({p1/n1:.0%})  sampler {p2}/{n2} ({p2/n2:.0%})  Fisher p={fisher_two_sided(p1, n1-p1, p2, n2-p2):.3f}")
