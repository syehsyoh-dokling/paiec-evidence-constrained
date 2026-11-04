"""Rebuild <results>/<tag>/combined.json from all per-(variant, fold) result files, so that
variants from different runs under the same tag are analysed together.
Usage: python rebuild_combined.py <tag>"""
import glob
import json
import pathlib
import sys

tag = sys.argv[1]
d = pathlib.Path(r"c:\Users\Saifuddin\Documents\LOMBA\localsim\results") / tag
res = {}
for f in sorted(d.glob("*__*.json")):
    name = f.name.split("__")[0]
    for row in json.load(open(f))["results"]:
        res.setdefault(name, {})[row["pair"]] = row
json.dump(res, open(d / "combined.json", "w"))
print({k: len(v) for k, v in res.items()})
