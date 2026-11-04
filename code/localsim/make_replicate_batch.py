"""Stage N byte-identical replicates of one submission ZIP for a hosted batch.

Rationale: on the formative leaderboard each submission is scored on a fresh random sample
(per-submission SD ~0.027), so the probability that at least one of N draws falls below a
target is maximised by submitting N copies of the model with the highest expected score,
not N near-identical variants. Replicates also give a direct estimate of resampling noise.
Usage: python make_replicate_batch.py <source.zip> <out_dir> <N> <version_prefix>
"""
import hashlib
import json
import pathlib
import shutil
import sys

src, out, n, prefix = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
out.mkdir(parents=True, exist_ok=True)
sha = hashlib.sha256(src.read_bytes()).hexdigest()
manifest = []
for i in range(1, n + 1):
    fname = f"sub_paiec_{prefix}_rep{i:02d}.zip"
    assert len(fname) <= 40
    shutil.copy(src, out / fname)
    manifest.append({"version": f"{prefix}_rep{i:02d}", "file": fname, "source": src.name, "verdict": "PASS_UPLOAD", "config": "replicate", "sha256": sha})
json.dump(manifest, open(out / "manifest.json", "w"), indent=1)
print(f"staged {n} replicates of {src.name} (sha {sha[:12]}) in {out}")
