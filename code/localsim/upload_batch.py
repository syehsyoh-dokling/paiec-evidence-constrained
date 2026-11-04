"""Upload every ZIP listed in <dir>/manifest.json to Codabench phase 29785, with an
incremental ledger (<dir>/upload_ledger.json) so a rerun never double-submits.
Stops cleanly when the daily quota is reached. Usage: python upload_batch.py <dir> [--max N]
"""
import argparse
import hashlib
import json
import sys
import time
import uuid
from pathlib import Path

import requests

TOKEN = Path(r"C:\Users\Saifuddin\.config\paiec\codabench-token").read_text().strip()
BASE = "https://www.codabench.org/api"
PHASE_ID = 29785
COMPETITION_ID = 17828
session = requests.Session()
session.headers.update({"Authorization": f"Token {TOKEN}"})


def req(method, url, **kw):
    last = None
    for attempt in range(6):
        try:
            r = session.request(method, url, timeout=35, **kw)
            if r.status_code == 429:
                time.sleep(15 + attempt * 8); continue
            if r.ok:
                return r
            last = f"HTTP {r.status_code}: {r.text[:160]}"
            if "maximum allowed submissions" in r.text:
                raise QuotaReached(last)
        except QuotaReached:
            raise
        except Exception as e:
            last = str(e)
        time.sleep(3 + attempt * 2)
    raise RuntimeError(f"failed {method} {url}: {last}")


class QuotaReached(Exception):
    pass


def put_file(url, path):
    for attempt in range(6):
        try:
            with open(path, "rb") as f:
                r = requests.put(url, data=f, headers={"Content-Type": "application/zip"}, timeout=90)
            if r.ok:
                return
        except Exception:
            pass
        time.sleep(3 + attempt * 2)
    raise RuntimeError(f"PUT failed {path}")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("dir", type=Path); ap.add_argument("--max", type=int, default=50)
    a = ap.parse_args()
    manifest = json.load(open(a.dir / "manifest.json"))
    ledger_path = a.dir / "upload_ledger.json"
    ledger = json.load(open(ledger_path)) if ledger_path.exists() else []
    done = {r["filename"] for r in ledger if r.get("sid")}
    sent = 0
    for m in manifest:
        if sent >= a.max:
            break
        zp = a.dir / m["file"]
        if zp.name in done:
            continue
        size = zp.stat().st_size
        print(f"uploading {zp.name} ({size:,} B) {m.get('verdict')} dALC={m.get('dalc')}", flush=True)
        try:
            ds = req("POST", f"{BASE}/datasets/", json={"name": f"submission-{uuid.uuid4().hex[:12]}", "type": "submission", "competition": COMPETITION_ID,
                                                         "is_public": False, "request_sassy_file_name": zp.name, "file_name": zp.name, "file_size": size}).json()
            put_file(ds["sassy_url"], zp)
            req("PUT", f"{BASE}/datasets/completed/{ds['key']}/")
            sub = req("POST", f"{BASE}/submissions/", json={"phase": PHASE_ID, "data": ds["key"]}).json()
            sid = sub.get("id")
            print(f"   -> submission id {sid}", flush=True)
            ledger.append({**m, "sid": sid, "filename": zp.name, "sha256": hashlib.sha256(zp.read_bytes()).hexdigest(), "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")})
            sent += 1
        except QuotaReached as e:
            print(f"   -> QUOTA REACHED, stopping: {e}", flush=True)
            json.dump(ledger, open(ledger_path, "w"), indent=2)
            sys.exit(2)
        except Exception as e:
            print(f"   -> FAILED {zp.name}: {e}", flush=True)
            ledger.append({**m, "sid": None, "filename": zp.name, "error": str(e), "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")})
        json.dump(ledger, open(ledger_path, "w"), indent=2)
        time.sleep(2.5)
    ok = [r for r in ledger if r.get("sid")]
    print(f"DONE: {len(ok)} submitted total; ledger {ledger_path}")


if __name__ == "__main__":
    main()
