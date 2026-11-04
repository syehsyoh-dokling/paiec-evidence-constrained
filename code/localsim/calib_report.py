"""Calibration report from simulator results that carry per-pair, per-budget 10-bin histograms.

For every variant and budget: mean ECE over pairs (10 equal-width bins, response-weighted, as the
organizers define it) and the Murphy decomposition of the Brier score pooled over pairs:
  Brier = reliability - resolution + uncertainty
  reliability = sum_k n_k (pbar_k - ybar_k)^2 / N ; resolution = sum_k n_k (ybar_k - ybar)^2 / N ;
  uncertainty = ybar (1 - ybar).
Usage: python calib_report.py <results_dir> [variant ...]
"""
import json
import pathlib
import statistics
import sys

BUDGETS = [0, 1, 3, 7, 15, 31]


def main():
    res_dir = pathlib.Path(sys.argv[1])
    res = json.load(open(res_dir / "combined.json"))
    names = sys.argv[2:] or list(res)
    print(f"{'variant':30s} budget   ECE(mean/pair)  reliability  resolution  uncertainty  Brier(pooled)")
    for name in names:
        rows = [r for r in res[name].values() if "calib" in r]
        if not rows:
            print(f"{name:30s} (no calibration histograms in these results)")
            continue
        for bi, B in enumerate(BUDGETS):
            eces = []
            pooled = [[0, 0.0, 0.0] for _ in range(10)]
            for r in rows:
                bins = r["calib"][bi]
                n_pair = sum(b[0] for b in bins)
                if n_pair:
                    eces.append(sum(b[0] * abs(b[1] / b[0] - b[2] / b[0]) for b in bins if b[0]) / n_pair)
                for k in range(10):
                    pooled[k][0] += bins[k][0]; pooled[k][1] += bins[k][1]; pooled[k][2] += bins[k][2]
            N = sum(b[0] for b in pooled)
            ybar = sum(b[2] for b in pooled) / N
            rel = sum(b[0] * (b[1] / b[0] - b[2] / b[0]) ** 2 for b in pooled if b[0]) / N
            reso = sum(b[0] * (b[2] / b[0] - ybar) ** 2 for b in pooled if b[0]) / N
            unc = ybar * (1 - ybar)
            print(f"{name:30s} {B:6d}   {statistics.mean(eces):.4f}          {rel:.4f}       {reso:.4f}      {unc:.4f}       {rel - reso + unc:.4f}")


if __name__ == "__main__":
    main()
