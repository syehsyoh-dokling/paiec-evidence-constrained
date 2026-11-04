"""v402 = budget-0 fix + a logistic (1PL-with-ease) item-response estimator fitted at call time.

P(y_{s,i}=1) = sigmoid(theta_s + beta_b + kappa * d_i)
  theta_s ~ N(logit(p0_s), 1/tau_s)   (own prior from subject_priors; peers get their own prior centre)
  beta_b  ~ N(0, 1/tau_b)
  d_i     = zero-shot item ease from the universal regressor (probability-scale, scaled by kappa)
Fitted by damped Newton (coordinate-wise) on all revealed labels of the target benchmark (own + peers);
labels of the target subject on other benchmarks enter through a second set of beta_b' offsets.
Prediction = sigmoid(theta_s + beta_b + kappa * d_target), blended with p_base by budget weight.
"""
import pathlib
import zipfile

ROOT = pathlib.Path(r"c:\Users\Saifuddin\Documents\LOMBA")
SRC = ROOT / "localsim" / "variants" / "b0fix_clean_ref.zip"
OUT = ROOT / "localsim" / "variants"

with zipfile.ZipFile(SRC) as z:
    base = z.read("model.py").decode("utf-8")

IRT_FN = '''

def _logit(p: float) -> float:
    p = min(max(p, 1e-4), 1 - 1e-4)
    return math.log(p / (1 - p))


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x)) if x > -35 else 0.0


def _irt_estimate(subject: dict, benchmark: str, labeled: list, p0: float, d_target: float,
                  kappa: float, tau_s: float, tau_b: float, iters: int = 8):
    """MAP fit of theta_s (per subject), beta_b (per benchmark) under Gaussian priors on the logit
    scale, item ease fixed from the regressor. Returns (prediction, n_labels_used) or (None, 0)."""
    rows = []
    centres = {}
    tkey = _subject_key(subject)
    for entry in labeled:
        try:
            (obs_s, obs_i), label = entry
        except Exception:
            continue
        if label not in (0, 1) or not isinstance(obs_s, dict) or not isinstance(obs_i, dict):
            continue
        skey = _subject_key(obs_s)
        bkey = str(obs_i.get("benchmark_id") or "")
        if skey not in centres:
            info = _lookup_db_subject_info(obs_s)
            m = float(info.get("mean", 0.50)) if info else 0.50
            centres[skey] = _logit(0.50 + 0.5 * (m - 0.50))
        d = _predict_2pl_item_ease(str(obs_i.get("item_content") or ""), 0.0, bm_id=bkey)
        rows.append((skey, bkey, float(label), float(d)))
    if not rows:
        return None, 0
    if tkey not in centres:
        centres[tkey] = _logit(p0)
    theta = {s: c for s, c in centres.items()}
    beta = {b: 0.0 for _, b, _, _ in rows}
    for _ in range(iters):
        # subject step
        g = {s: 0.0 for s in theta}; h = {s: 0.0 for s in theta}
        for s, b, y, d in rows:
            p = _sigmoid(theta[s] + beta[b] + kappa * d)
            g[s] += (y - p); h[s] += p * (1 - p)
        for s in theta:
            g[s] -= tau_s * (theta[s] - centres[s]); h[s] += tau_s
            theta[s] += g[s] / max(h[s], 1e-6)
        # benchmark step
        g = {b: 0.0 for b in beta}; h = {b: 0.0 for b in beta}
        for s, b, y, d in rows:
            p = _sigmoid(theta[s] + beta[b] + kappa * d)
            g[b] += (y - p); h[b] += p * (1 - p)
        for b in beta:
            g[b] -= tau_b * beta[b]; h[b] += tau_b
            beta[b] += g[b] / max(h[b], 1e-6)
    tb = str(benchmark or "")
    n_used = sum(1 for s, b, _, _ in rows if s == tkey or b == tb)
    if n_used == 0:
        return None, 0
    pred = _sigmoid(theta[tkey] + beta.get(tb, 0.0) + kappa * d_target)
    return float(min(max(pred, 0.02), 0.98)), n_used


def _subject_key(subj: dict) -> str:
    return "|".join(str(subj.get(k) or "") for k in ("normalized_name", "harness", "harness_version", "reasoning_effort"))


def _irt_weight(count: int, profile) -> float:
    l1, l3, l7, linf = profile
    if count <= 1:
        return l1
    if count <= 3:
        return l3
    if count <= 7:
        return l7
    return linf
'''

ANCHOR_DEF = "\ndef predict(input: list[dict], labeled: list | None = None) -> float:"
OLD_AFTER_BASE = '''    # If multiple conflicting trials of the exact target item were observed for this subject,
    # return empirical trial mean shrunk lightly toward p_base
    if exact_self_labels:'''
NEW_AFTER_BASE = '''    # --- v402: logistic item-response estimate on all revealed labels (logit-additive) ---
    _p_irt, _n_irt = _irt_estimate(subject, benchmark, labeled, p0, d_target_2pl, *_V402_IRT)
    if _p_irt is not None and _n_irt > 0:
        _lam = _irt_weight(count, _V402_PROFILE)
        p_base = float(min(max((1.0 - _lam) * p_base + _lam * _p_irt, 0.005), 0.995))

    # If multiple conflicting trials of the exact target item were observed for this subject,
    # return empirical trial mean shrunk lightly toward p_base
    if exact_self_labels:'''

VARIANTS = {
    # name: (kappa, tau_s, tau_b, weight profile)
    "v402a_irt_k3_ts2_tb2_w040": (3.0, 2.0, 2.0, (0.40, 0.30, 0.20, 0.10)),
    "v402b_irt_k3_ts2_tb2_w060": (3.0, 2.0, 2.0, (0.60, 0.50, 0.30, 0.15)),
    "v402c_irt_k5_ts4_tb1_w040": (5.0, 4.0, 1.0, (0.40, 0.30, 0.20, 0.10)),
}
assert base.count(ANCHOR_DEF) == 1 and base.count(OLD_AFTER_BASE) == 1
for name, (k, ts, tb, prof) in VARIANTS.items():
    text = base.replace(ANCHOR_DEF, IRT_FN + f"\n_V402_IRT = ({k}, {ts}, {tb})\n_V402_PROFILE = {prof!r}\n" + ANCHOR_DEF, 1)
    text = text.replace(OLD_AFTER_BASE, NEW_AFTER_BASE, 1)
    compile(text, name, "exec")
    out = OUT / f"{name}.zip"
    with zipfile.ZipFile(SRC) as zin, zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zout:
        for info in zin.infolist():
            data = zin.read(info.filename)
            if info.filename == "model.py":
                data = text.encode("utf-8")
            zout.writestr(info, data)
    print("wrote", out.name)
