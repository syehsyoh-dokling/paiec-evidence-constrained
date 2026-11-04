"""Sensitivity of ALC to the slot parameters, from every gated candidate (student and sampler
batches): per slot, number of candidates, min / median / max paired dALC vs v153 and the
fraction of the slot's parameter range covered (from the autopilot reports' dALC column).
Also reads the slot ranges from qa_gate.ALLOWED_REGIONS when importable."""
import collections, pathlib, re, statistics
R = pathlib.Path(r"c:\Users\Saifuddin\Documents\LOMBA\localsim\results")
SLOT = [("control", "control"), ("_b0", "budget-0 branch (centre, slope, ease)"), ("low_budget", "low-budget shrinkage (blend weights)"),
        ("peer_shift", "peer shift (gain, centre, shrink)"), ("mixture", "mixture-prior pseudo-counts"), ("irt", "IRT envelope weight")]
def slot_of(name):
    n = name.lower()
    for k, s in SLOT:
        if k in n:
            return s
    return "other"
vals = collections.defaultdict(list)
for rep in sorted(R.glob("qa_*/AUTOPILOT_REPORT.md")):
    for line in rep.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^\|\s*(\S+\.zip)\s*\|\s*\w+\s*\|\s*(\w+)\s*\|\s*([+-]?[0-9.]+)\s*\|", line)
        if m and slot_of(m.group(1)) != "control":
            vals[slot_of(m.group(1))].append(float(m.group(3)))
print(f"{'slot':42s} {'n':>4s} {'min':>8s} {'median':>8s} {'max':>8s} {'n PASS':>7s}")
for s, xs in vals.items():
    print(f"{s:42s} {len(xs):4d} {min(xs):+8.4f} {statistics.median(xs):+8.4f} {max(xs):+8.4f} {sum(1 for x in xs if x <= -0.0005):7d}")
