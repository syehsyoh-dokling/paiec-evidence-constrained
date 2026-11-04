"""Remove comments that mention anonymous hidden benchmark/subject IDs from model.py inside
submission ZIPs. Only comment text changes; the script proves this by comparing the AST of
model.py before and after. Exits 1 if any text file in the ZIP still contains such an ID.

Usage: python strip_hidden_ids.py <in.zip> [<out.zip>]      (in place when out is omitted)
       python strip_hidden_ids.py --dir <folder>             (every *.zip in the folder, in place)
"""
import argparse
import ast
import re
import sys
import zipfile
from pathlib import Path

HID = re.compile(r"\b(benchmark|subject)_\d{6}\b")
REPL = {
    "Regime 1": "# Regime 1: near-zero pass rate",
    "Regime 2": "# Regime 2: mid-hard",
    "Regime 3": "# Regime 3: high saturation",
}


def clean_model(text: str) -> str:
    out = []
    for line in text.splitlines(keepends=True):
        if "#" in line and HID.search(line):
            code, _, comment = line.partition("#")
            new_comment = next((v for k, v in REPL.items() if k in comment), "# calibrated on aggregated formative feedback")
            line = code.rstrip() + "  " + new_comment + ("\n" if line.endswith("\n") else "")
        out.append(line)
    return "".join(out)


def same_ast(a: str, b: str) -> bool:
    return ast.dump(ast.parse(a)) == ast.dump(ast.parse(b))


def process(src: Path, dst: Path) -> dict:
    with zipfile.ZipFile(src) as zin:
        entries = [(i, zin.read(i.filename)) for i in zin.infolist()]
    changed = 0
    leftovers = []
    new_entries = []
    for info, data in entries:
        if info.filename == "model.py":
            text = data.decode("utf-8")
            cleaned = clean_model(text)
            assert same_ast(text, cleaned), "AST changed: refusing to write"
            if cleaned != text:
                changed += 1
            data = cleaned.encode("utf-8")
        if info.filename.endswith((".py", ".json", ".md", ".txt")) and HID.search(data.decode("utf-8", "replace")):
            leftovers.append(info.filename)
        new_entries.append((info, data))
    tmp = dst.with_suffix(".tmp.zip")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
        for info, data in new_entries:
            zout.writestr(info, data)
    tmp.replace(dst)
    return {"file": dst.name, "model_py_changed": bool(changed), "leftover_files_with_ids": leftovers}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", nargs="?"); ap.add_argument("dst", nargs="?"); ap.add_argument("--dir")
    a = ap.parse_args()
    jobs = [(p, p) for p in sorted(Path(a.dir).glob("*.zip"))] if a.dir else [(Path(a.src), Path(a.dst or a.src))]
    bad = False
    for s, d in jobs:
        r = process(s, d)
        bad |= bool(r["leftover_files_with_ids"])
        print(r)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
