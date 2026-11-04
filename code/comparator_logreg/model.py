"""Literature-style comparator: a logistic regression refit at every call on ALL revealed labels.

Features: one-hot subject, one-hot benchmark, one-hot (subject, benchmark) pair, and a
MiniLM sentence embedding of the item (384-d, scaled). L2 penalties: categorical 1.0,
embedding 5.0. Fitted by a few IRLS (Newton) steps in NumPy. No fitted artifacts, no priors:
budget 0 returns 0.5. This is the standard "learn a subject + benchmark + item-feature model
from the revealed labels" baseline that the performance-prediction literature would expect.
"""
from __future__ import annotations

import math
import os
import numpy as np

_TOK = None
_MOD = None
_CACHE: dict[str, np.ndarray] = {}
EMB_SCALE = 0.5
L2_CAT = 1.0
L2_EMB = 5.0
NEWTON_STEPS = 6


def _encoder():
    global _TOK, _MOD
    if _MOD is None:
        import torch
        from transformers import AutoModel, AutoTokenizer
        _TOK = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        _MOD = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        _MOD.to("cuda" if torch.cuda.is_available() else "cpu").eval()
    return _TOK, _MOD


def _embed(text: str) -> np.ndarray:
    key = (text or " ")[:800].strip() or " "
    if key in _CACHE:
        return _CACHE[key]
    try:
        import torch
        tok, mod = _encoder()
        inp = tok([key], padding=True, truncation=True, max_length=256, return_tensors="pt")
        inp = {k: v.to(mod.device) for k, v in inp.items()}
        with torch.no_grad():
            out = mod(**inp)
        mask = inp["attention_mask"].unsqueeze(-1)
        vec = ((out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9))[0].cpu().numpy().astype(np.float32)
        vec /= (np.linalg.norm(vec) + 1e-9)
    except Exception:
        vec = np.zeros(384, dtype=np.float32)
    _CACHE[key] = vec
    return vec


def _skey(s: dict) -> str:
    return "|".join(str(s.get(k) or "") for k in ("normalized_name", "harness", "harness_version", "reasoning_effort"))


_FIT_CACHE: dict = {}  # fingerprint of the labeled set -> (w, index maps); one fit per budget per pair


def _fit(rows, tkey, tb):
    import hashlib
    fp = hashlib.sha1(repr(sorted(rows)).encode() + f"|{tkey}|{tb}".encode()).hexdigest()
    if fp in _FIT_CACHE:
        return _FIT_CACHE[fp]
    subs = sorted({r[0] for r in rows} | {tkey}); bms = sorted({r[1] for r in rows} | {tb})
    pairs = sorted({(r[0], r[1]) for r in rows} | {(tkey, tb)})
    si = {k: i for i, k in enumerate(subs)}; bi = {k: i for i, k in enumerate(bms)}; pi = {k: i for i, k in enumerate(pairs)}
    n_cat = 1 + len(subs) + len(bms) + len(pairs)
    d = n_cat + 384

    def feat(skey, bkey, text):
        x = np.zeros(d, dtype=np.float64)
        x[0] = 1.0
        x[1 + si[skey]] = 1.0
        x[1 + len(subs) + bi[bkey]] = 1.0
        x[1 + len(subs) + len(bms) + pi[(skey, bkey)]] = 1.0
        x[n_cat:] = EMB_SCALE * _embed(text)
        return x

    X = np.vstack([feat(r[0], r[1], r[2]) for r in rows]); y = np.array([r[3] for r in rows])
    lam = np.full(d, L2_CAT); lam[0] = 0.01; lam[n_cat:] = L2_EMB
    w = np.zeros(d)
    for _ in range(NEWTON_STEPS):
        z = X @ w
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        g = X.T @ (p - y) + lam * w
        W = p * (1 - p)
        H = (X * W[:, None]).T @ X + np.diag(lam)
        try:
            w -= np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            break
    if len(_FIT_CACHE) > 512:
        _FIT_CACHE.clear()
    _FIT_CACHE[fp] = (w, feat)
    return _FIT_CACHE[fp]


def predict(input: list[dict], labeled: list | None = None) -> float:
    if not input or len(input) < 2:
        return 0.5
    subject, item = input
    rows = []
    for entry in labeled or []:
        try:
            (s, it), y = entry
        except Exception:
            continue
        if y not in (0, 1) or not isinstance(s, dict) or not isinstance(it, dict):
            continue
        rows.append((_skey(s), str(it.get("benchmark_id") or ""), str(it.get("item_content") or ""), float(y)))
    if not rows:
        return 0.5
    tkey, tb = _skey(subject), str(item.get("benchmark_id") or "")
    w, feat = _fit(rows, tkey, tb)
    xt = feat(tkey, tb, str(item.get("item_content") or ""))
    pr = 1.0 / (1.0 + math.exp(-float(np.clip(xt @ w, -30, 30))))
    return float(min(max(pr, 0.02), 0.98))
