"""Replace hardcoded development-machine paths in the released scripts with environment
overrides (PAIEC_ROOT for the project root, PAIEC_CODABENCH_TOKEN_FILE for the API token),
keeping the original path as the default so nothing changes on the development machine.
Compiles every edited file afterwards. Usage: python make_paths_portable.py [--dry]"""
import pathlib, py_compile, re, sys
ROOT = pathlib.Path(r"c:\Users\Saifuddin\Documents\LOMBA")
DRY = "--dry" in sys.argv
LOMBA = r"[cC]:\\Users\\Saifuddin\\Documents\\LOMBA"
files = sorted(set(list((ROOT / "localsim").glob("*.py")) + [ROOT / "generate_v38_subject_priors_and_simulate_alc.py", ROOT / "export_v38_universal_irt.py"]))
pat_path = re.compile(r'(pathlib\.Path|Path)\(r"(' + LOMBA + r')((?:\\[^"\\]+)*)"\)')
pat_raw = re.compile(r'r"(' + LOMBA + r')((?:\\[^"\\]+)*)"')
pat_tok = re.compile(r'(?:pathlib\.Path|Path)?\(?r"[cC]:[\\/]Users[\\/]Saifuddin[\\/]\.config[\\/]paiec[\\/]codabench-token"\)?')


def subpath(callable_name, root, tail):
    comps = [c for c in tail.split("\\") if c]
    out = f'{callable_name}(os.environ.get("PAIEC_ROOT", r"{root}"))'
    for c in comps:
        out += f' / "{c}"'
    return out


changed = []
for f in files:
    if f.name == "make_paths_portable.py" or not f.exists():
        continue
    src = f.read_text(encoding="utf-8")
    new = pat_path.sub(lambda m: subpath(m.group(1), m.group(2), m.group(3)), src)
    # token file: open(r"...codabench-token") or Path(r"...codabench-token")
    new = re.sub(r'open\(r"[cC]:[\\/]Users[\\/]Saifuddin[\\/]\.config[\\/]paiec[\\/]codabench-token"\)',
                 'open(os.environ.get("PAIEC_CODABENCH_TOKEN_FILE", r"C:/Users/Saifuddin/.config/paiec/codabench-token"))', new)
    new = re.sub(r'(pathlib\.Path|Path)\(r"[cC]:[\\/]Users[\\/]Saifuddin[\\/]\.config[\\/]paiec[\\/]codabench-token"\)',
                 lambda m: f'{m.group(1)}(os.environ.get("PAIEC_CODABENCH_TOKEN_FILE", r"C:/Users/Saifuddin/.config/paiec/codabench-token"))', new)
    # remaining raw strings under LOMBA not wrapped in Path(...): e.g. sys.path.insert(0, r"...\localsim")
    def raw_sub(m):
        comps = [c for c in m.group(2).split("\\") if c]
        expr = f'os.environ.get("PAIEC_ROOT", r"{m.group(1)}")'
        return f'os.path.join({expr}, {", ".join(repr(c) for c in comps)})' if comps else expr
    new = pat_raw.sub(raw_sub, new)
    if new != src:
        if "import os" not in new and "import os," not in new and not re.search(r"^\s*import\s+[^\n]*\bos\b", new, re.M):
            # insert after the module docstring / __future__ imports: before the first import line
            lines = new.split("\n")
            for i, ln in enumerate(lines):
                if ln.startswith("import ") or ln.startswith("from ") and "__future__" not in ln:
                    lines.insert(i, "import os")
                    break
            new = "\n".join(lines)
        changed.append(f)
        if not DRY:
            f.write_text(new, encoding="utf-8")
            py_compile.compile(str(f), doraise=True)
print(("would change" if DRY else "changed and compiled"), len(changed), "files:")
for f in changed:
    print("  ", f.relative_to(ROOT))
left = [f for f in files if f.exists() and re.search(LOMBA, f.read_text(encoding="utf-8")) and 'os.environ.get("PAIEC_ROOT"' not in f.read_text(encoding="utf-8")]
print("files still containing the path without override:", [str(f.relative_to(ROOT)) for f in left])
