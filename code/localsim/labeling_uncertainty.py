"""Uncertainty-weighted acquisition for PAIEC (candidate policy, evaluated by LOBO).

The platform offers each acquisition candidate once with the predictor's current estimate
`prediction`. We acquire with probability
    thresh = r * (1 - mix + mix * u / u_bar)
where r = labels_remaining / items_remaining (the budget-neutral base rate), u = 4 p (1 - p)
is the Bernoulli information of the current prediction (1 at p = 0.5, 0 at the extremes),
u_bar is a running mean of u over candidates seen for this pair, and mix in [0, 1] sets how
far from uniform the policy goes. Deterministic SHA-256 uniforms keep runs reproducible.
Rationale: for estimating a pass rate with a logit-additive model, items whose outcome is
most uncertain carry the most Fisher information; the predictor (v402) uses item ease, so
the selection bias of non-uniform sampling is absorbed by the model rather than the mean.
"""
from __future__ import annotations

import hashlib
import json

MIX = __MIX__
_STATE: dict[str, list] = {}  # pair key -> [sum_u, n]


def acquisition_function(input: list[dict], prediction: float | None = None, labeled: list | None = None, context: dict | None = None) -> bool:
    if context is None:
        return True
    labels_rem = int(context.get("labels_remaining", 0))
    items_rem = int(context.get("items_remaining", 1))
    if labels_rem <= 0:
        return False
    if items_rem <= labels_rem:
        return True
    r = float(labels_rem) / float(max(items_rem, 1))
    key = f"{context.get('subject_id')}|{context.get('benchmark_id')}"
    p = 0.5 if prediction is None else min(max(float(prediction), 0.0), 1.0)
    u = 4.0 * p * (1.0 - p)
    s = _STATE.setdefault(key, [0.0, 0])
    s[0] += u; s[1] += 1
    u_bar = max(s[0] / s[1], 1e-3)
    thresh = r * ((1.0 - MIX) + MIX * u / u_bar)
    thresh = min(max(thresh, 0.0), 1.0)
    seed = json.dumps([context, input], sort_keys=True, separators=(",", ":"))
    uniform = int.from_bytes(hashlib.sha256(seed.encode("utf-8")).digest()[:8], "big") / float(2 ** 64)
    return bool(uniform < thresh)
