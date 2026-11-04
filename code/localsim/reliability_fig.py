"""Reliability diagrams at budgets 0, 1, 3 and 31 for v153, v402b and v403, from the 10-bin
calibration histograms stored per pair by the simulator (calib[budget][bin] = [n, sum_pred, sum_y]).
Writes openreview_report/figures/reliability.pdf/.png and prints pooled ECE per variant/budget."""
import json, pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = pathlib.Path(r"c:\Users\Saifuddin\Documents\LOMBA")
D = ROOT / "localsim" / "results" / "calib16_ctx2"
FIG = ROOT / "openreview_report" / "figures"
VARS = [("v153_nokeys", "v153 (submitted)"), ("v402b_irt_k3_ts2_tb2_w060", "v402b (logistic IRT)"), ("v402b_expanded_priors", "v403 (+12-benchmark priors)")]
BUD = [0, 1, 3, 7, 15, 31]
SHOW = [0, 1, 3, 31]
hist = {}
for key, _ in VARS:
    agg = [[[0, 0.0, 0.0] for _ in range(10)] for _ in BUD]
    for f in sorted(D.glob(f"{key}__*.json")):
        for row in json.load(open(f))["results"]:
            c = row.get("calib")
            if not c:
                continue
            for bi in range(6):
                for k in range(10):
                    n, sp, sy = c[bi][k]
                    agg[bi][k][0] += n; agg[bi][k][1] += sp; agg[bi][k][2] += sy
    hist[key] = agg
fig, axes = plt.subplots(1, len(SHOW), figsize=(11, 3.0), sharey=True)
for ax, b in zip(axes, SHOW):
    bi = BUD.index(b)
    ax.plot([0, 1], [0, 1], color="0.7", lw=1, ls="--")
    for (key, label), m in zip(VARS, ["o", "s", "^"]):
        xs, ys, ns = [], [], []
        tot = sum(h[0] for h in hist[key][bi]); ece = 0.0
        for n, sp, sy in hist[key][bi]:
            if n >= 20:
                xs.append(sp / n); ys.append(sy / n); ns.append(n)
            if n:
                ece += n / tot * abs(sp / n - sy / n)
        ax.plot(xs, ys, marker=m, ms=4, lw=1.2, label=f"{label}, ECE {ece:.3f}")
        print(f"{key:28s} budget {b:2d} pooled ECE {ece:.4f} n={tot}")
    ax.set_title(f"budget {b}", fontsize=9); ax.set_xlabel("mean predicted probability", fontsize=8)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.tick_params(labelsize=7); ax.legend(fontsize=6, loc="upper left")
axes[0].set_ylabel("observed pass frequency", fontsize=8)
plt.tight_layout(); plt.savefig(FIG / "reliability.pdf"); plt.savefig(FIG / "reliability.png", dpi=160)
print("wrote", FIG / "reliability.pdf")
