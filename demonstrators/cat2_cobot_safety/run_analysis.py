"""Cat. 2 — Cobot Safety + Brake Disc QC: Combined Instrumentation Analysis.

Produces:
- Console summary for both cases
- JSON evidence snippets with 7 signals (5 protocol + 2 Industry 5.0)
- Markdown report saved to analysis_history/cobot_brake_disc_analysis_YYYYMMDD.md

Failure modes:
  cobot_safety   → identity_contamination  (constraint_enforcement primary)
  brake_disc_qc  → brittle_autopass        (decision_robustness primary)

Industry 5.0 thresholds (approved 2026-03-26):
  sustainability FLAG if proxy_attribution_total > 0.25 (both cases)
  decision_robustness FLAG if flip_rate > 0.3 (brake_disc primary signal)
"""

import json
import sys
from datetime import date
from math import erfc, sqrt
from pathlib import Path

import numpy as np

# ── path setup ──────────────────────────────────────────────────────────────
THIS_DIR = Path(__file__).resolve().parent
DEMONSTRATORS_DIR = THIS_DIR.parent
sys.path.insert(0, str(DEMONSTRATORS_DIR))

from cat2_cobot_safety.generate_dataset import (
    generate_cobot_dataset,
    FEATURE_NAMES as CB_FEAT,
    G_DOMAIN as CB_GD,
    G_PROXY as CB_GP,
    CODED_EFFECTS as CB_CE,
)
from cat2_cobot_safety.generate_dataset_brake_disc import (
    generate_brake_disc_dataset,
    FEATURE_NAMES as BD_FEAT,
    G_DOMAIN as BD_GD,
    G_PROXY as BD_GP,
    CODED_EFFECTS as BD_CE,
    KNN_REFERENCE_COL as BD_KNN_COL,
    KNN_REFERENCE_VAL as BD_KNN_VAL,
    KNN_SUBSPACE_INDICES as BD_KNN_SUBSPACE,
)

ANALYSIS_HISTORY_DIR = THIS_DIR.parent.parent.parent / "analysis_history"


# ── minimal ML / stats primitives ───────────────────────────────────────────

class SimpleLR:
    def __init__(self):
        self.w = None
        self.b = 0.0

    def fit(self, X, y, lr=0.05, n_iter=500):
        n, d = X.shape
        self.w = np.zeros(d)
        self.b = 0.0
        for _ in range(n_iter):
            z = X @ self.w + self.b
            p = 1 / (1 + np.exp(-np.clip(z, -500, 500)))
            self.w -= lr * (X.T @ (p - y) / n)
            self.b -= lr * np.mean(p - y)

    def predict_proba(self, X):
        z = X @ self.w + self.b
        p1 = 1 / (1 + np.exp(-np.clip(z, -500, 500)))
        return np.column_stack([1 - p1, p1])


def mwu(x, y):
    """One-sided Mann-Whitney U. Returns (U, p)."""
    nx, ny = len(x), len(y)
    combined = np.concatenate([x, y])
    ranks = np.argsort(np.argsort(combined)) + 1.0
    U = np.sum(ranks[:nx]) - nx * (nx + 1) / 2
    mu = nx * ny / 2
    sigma = sqrt(nx * ny * (nx + ny + 1) / 12)
    z = (U - mu) / (sigma + 1e-10)
    return U, 0.5 * erfc(z / sqrt(2))


def attribution(model, x, bg, n_bg=60):
    bg_ = bg[:n_bg]
    base = model.predict_proba(x.reshape(1, -1))[0, 1]
    attr = np.zeros(len(x))
    for i in range(len(x)):
        Xp = np.tile(x, (len(bg_), 1))
        Xp[:, i] = bg_[:, i]
        attr[i] = base - np.mean(model.predict_proba(Xp)[:, 1])
    return attr


def flip_rate(model, x, feature_range, threshold=0.5, n=50, noise_scale=0.05):
    base = int(model.predict_proba(x.reshape(1, -1))[0, 1] >= threshold)
    flips = sum(
        1 for _ in range(n)
        if int(model.predict_proba(
            (x + np.random.normal(0, noise_scale, len(x)) * feature_range
             ).reshape(1, -1))[0, 1] >= threshold) != base
    )
    return flips / n


def knn_distance(x, X_ref, k=5, subspace=None):
    if subspace is not None:
        x = x[subspace]
        X_ref = X_ref[:, subspace]
    dists = np.sqrt(np.sum((X_ref - x) ** 2, axis=1))
    return float(np.mean(np.sort(dists)[:k]))


def attr_stability(model, x, bg, n_reps=10, sigma=0.01, rng=None):
    if rng is None:
        rng = np.random
    ranks = []
    for _ in range(n_reps):
        x_noisy = x + rng.normal(0, sigma, len(x))
        a = attribution(model, x_noisy, bg)
        rank = np.argsort(np.argsort(-np.abs(a))).astype(float)
        ranks.append(rank)
    return float(np.mean(np.std(np.array(ranks), axis=0)))


# ── Industry 5.0 signal builders ────────────────────────────────────────────

def sustainability_signal_cobot(proxy_total: float) -> dict:
    status = "FLAG" if proxy_total > 0.25 else "OK"
    return {
        "dimension": "worker_wellbeing",
        "at_risk_group": "temporary_workers",
        "indicator": "identity_driven_clearance_rate",
        "value": round(proxy_total, 4),
        "status": status,
    }


def human_ai_teaming_signal_cobot(fr_val: float, is_near: bool) -> dict:
    if fr_val > 0.3:
        level = "mandatory_review"
        status = "FLAG"
    elif is_near:
        level = "recommended_review"
        status = "OK"
    else:
        level = "nominal"
        status = "OK"
    return {
        "dimension": "supervisory_oversight",
        "escalation_level": level,
        "trigger": "near_threshold_with_identity_attribution",
        "human_override_available": True,
        "status": status,
    }


def sustainability_signal_brake_disc(proxy_total: float) -> dict:
    status = "FLAG" if proxy_total > 0.25 else "OK"
    return {
        "dimension": "product_safety",
        "indicator": "confounder_driven_autopass_rate",
        "value": round(proxy_total, 4),
        "note": "confounder-driven passes on safety parts create undetected recall risk",
        "status": status,
    }


def human_ai_teaming_signal_brake_disc(fr_val: float) -> dict:
    if fr_val > 0.3:
        level = "mandatory_review"
        status = "FLAG"
    elif fr_val > 0.1:
        level = "recommended_review"
        status = "OK"
    else:
        level = "nominal"
        status = "OK"
    return {
        "dimension": "quality_gate_oversight",
        "escalation_level": level,
        "trigger": "brittle_autopass_near_threshold",
        "human_override_available": True,
        "status": status,
    }


# ── core analysis function ───────────────────────────────────────────────────

def run_case_analysis(
    name: str,
    df,
    feat_names: list,
    g_d: list,
    g_p: list,
    coded: dict,
    failure_mode: str,
    demonstrator: str,
    sustainability_fn,
    teaming_fn,
    n_cases: int = 40,
    knn_bg_col: str = None,
    knn_bg_val: int = 0,
    knn_subspace: list = None,
):
    """Run full 7-signal instrumentation analysis for one case.

    Uses feat_names explicitly — never iterates over DataFrame columns.

    knn_bg_col / knn_bg_val: if provided, the k-NN prototype set is restricted
    to training rows where df[knn_bg_col] == knn_bg_val.

    knn_subspace: if provided, k-NN distance is computed only in these feature
    dimensions (indices into feat_names). Used when the envelope should measure
    out-of-distribution in a specific subspace (e.g. confounder dimensions only).
    """
    # ── data prep ──
    X = df[feat_names].values.astype(float)       # explicit FEATURE_NAMES
    y = df["label"].values.astype(float)
    mu, sd = X.mean(0), X.std(0) + 1e-10
    Xn = (X - mu) / sd

    np.random.seed(42)
    idx = np.random.permutation(len(X))
    sp = int(0.7 * len(X))
    Xtr, Xte = Xn[idx[:sp]], Xn[idx[sp:]]
    ytr, yte = y[idx[:sp]], y[idx[sp:]]

    model = SimpleLR()
    model.fit(Xtr, ytr)
    acc = float(np.mean((model.predict_proba(Xte)[:, 1] >= 0.5) == yte))
    scores = model.predict_proba(Xte)[:, 1]
    margins = np.abs(scores - 0.5)

    near_idx = np.where(margins < 0.15)[0]
    far_idx  = np.where(margins > 0.35)[0]
    feat_range = Xtr.max(0) - Xtr.min(0) + 1e-10

    # ── attribution/stability background (always full training sample) ──
    attr_bg = Xtr[:100]

    # ── k-NN prototype set (may be restricted to reference subset) ──
    if knn_bg_col is not None:
        col_tr = df[knn_bg_col].values[idx[:sp]]
        ref_mask = col_tr == knn_bg_val
        knn_bg = Xtr[ref_mask][:100]
    else:
        knn_bg = Xtr[:100]

    results = {
        "name": name,
        "demonstrator": demonstrator,
        "failure_mode": failure_mode,
        "annex_iii_category": "cat_2",
        "accuracy": acc,
        "n_near": int(len(near_idx)),
        "n_far": int(len(far_idx)),
        "near_cases": [],
        "far_cases": [],
    }

    for label, indices, is_near in [
        ("near", near_idx[:n_cases], True),
        ("far",  far_idx[:n_cases],  False),
    ]:
        for j in indices:
            x = Xte[j]
            sc = float(scores[j])
            attr = attribution(model, x, attr_bg)
            total = np.sum(np.abs(attr)) + 1e-10
            fr_val = flip_rate(model, x, feat_range)
            env_d = knn_distance(x, knn_bg, subspace=knn_subspace)
            _rng = np.random.RandomState(42 + int(j))
            astab = attr_stability(model, x, attr_bg, rng=_rng)

            legit_ratio = float(np.sum(np.abs(attr[g_d])) / total)
            proxy_attr = {feat_names[i]: float(np.abs(attr[i]) / total) for i in g_p}
            proxy_total = sum(proxy_attr.values())
            top4 = [feat_names[i] for i in np.argsort(-np.abs(attr))[:4]]

            # Verdict (non-compensatory)
            if astab > 0.3:
                verdict = "BLOCK"
            elif fr_val > 0.3 or legit_ratio < 0.6:
                verdict = "PROCEED WITH FLAG"
            else:
                verdict = "PROCEED"

            case = {
                # metadata
                "case_id": f"{demonstrator}-{label}-{int(j):04d}",
                "demonstrator": demonstrator,
                "failure_mode": failure_mode,
                "annex_iii_category": "cat_2",
                # signal 1: score + margin
                "output_score": round(sc, 4),
                "threshold": 0.50,
                "margin_to_threshold": round(abs(sc - 0.5), 4),
                # signal 2: constraint enforcement
                "constraint_enforcement": {
                    "legitimate_feature_ratio": round(legit_ratio, 4),
                    "proxy_attribution_total": round(proxy_total, 4),
                    "proxy_detail": {k: round(v, 4) for k, v in proxy_attr.items()},
                    "top_features": top4,
                    "status": "OK" if legit_ratio >= 0.6 else "CONTESTED",
                },
                # signal 3: envelope validity
                "envelope_validity": {
                    "distance_to_nearest_prototype": round(env_d, 4),
                    "status": "OK",
                },
                # signal 4: decision robustness
                "decision_robustness": {
                    "flip_rate": round(fr_val, 4),
                    "status": "CONTESTED" if fr_val > 0.3 else "OK",
                },
                # signal 5: record integrity
                "record_integrity": {
                    "attribution_stability_sigma": round(astab, 4),
                    "status": "BLOCK" if astab > 0.3 else "OK",
                },
                # Industry 5.0 — signal 6
                "sustainability_impact": sustainability_fn(proxy_total),
                # Industry 5.0 — signal 7
                "human_ai_teaming": teaming_fn(fr_val, is_near),
                "action_verdict": verdict,
            }
            results[f"{label}_cases"].append(case)

    # ── aggregate stats ──
    for label in ["near", "far"]:
        cases = results[f"{label}_cases"]
        if cases:
            results[f"{label}_flip_mean"]   = float(np.mean([c["decision_robustness"]["flip_rate"] for c in cases]))
            results[f"{label}_flip_std"]    = float(np.std( [c["decision_robustness"]["flip_rate"] for c in cases]))
            results[f"{label}_proxy_mean"]  = float(np.mean([c["constraint_enforcement"]["proxy_attribution_total"] for c in cases]))
            results[f"{label}_proxy_std"]   = float(np.std( [c["constraint_enforcement"]["proxy_attribution_total"] for c in cases]))
            results[f"{label}_legit_mean"]  = float(np.mean([c["constraint_enforcement"]["legitimate_feature_ratio"] for c in cases]))
            results[f"{label}_env_mean"]    = float(np.mean([c["envelope_validity"]["distance_to_nearest_prototype"] for c in cases]))
            results[f"{label}_env_std"]     = float(np.std( [c["envelope_validity"]["distance_to_nearest_prototype"] for c in cases]))
            results[f"{label}_astab_mean"]  = float(np.mean([c["record_integrity"]["attribution_stability_sigma"] for c in cases]))
            results[f"{label}_astab_std"]   = float(np.std( [c["record_integrity"]["attribution_stability_sigma"] for c in cases]))
            results[f"{label}_flagged_pct"] = float(np.mean([c["action_verdict"] != "PROCEED" for c in cases]))

    # ── Mann-Whitney U ──
    nc, fc = results["near_cases"], results["far_cases"]
    if len(nc) > 1 and len(fc) > 1:
        _, p_flip = mwu(
            np.array([c["decision_robustness"]["flip_rate"] for c in nc]),
            np.array([c["decision_robustness"]["flip_rate"] for c in fc]),
        )
        _, p_env = mwu(
            np.array([c["envelope_validity"]["distance_to_nearest_prototype"] for c in nc]),
            np.array([c["envelope_validity"]["distance_to_nearest_prototype"] for c in fc]),
        )
        _, p_astab = mwu(
            np.array([c["record_integrity"]["attribution_stability_sigma"] for c in nc]),
            np.array([c["record_integrity"]["attribution_stability_sigma"] for c in fc]),
        )
        results["flip_mwu_p"]       = p_flip
        results["flip_discriminates"]  = bool(p_flip < 0.01)
        results["env_mwu_p"]        = p_env
        results["env_discriminates"]   = bool(p_env < 0.01)
        results["astab_mwu_p"]      = p_astab
        results["astab_discriminates"] = bool(p_astab < 0.01)

    return results


# ── human_ai_teaming wrappers (uniform 2-arg signature) ─────────────────────

def _cobot_teaming(fr_val, is_near=False):
    return human_ai_teaming_signal_cobot(fr_val, is_near)


def _brake_teaming(fr_val, is_near=False):
    return human_ai_teaming_signal_brake_disc(fr_val)


# ── markdown report builder ──────────────────────────────────────────────────

def fmt_p(p: float) -> str:
    return f"{p:.2e}"


def build_markdown(cb: dict, bd: dict, today: str) -> str:
    cb_near = cb["near_cases"]
    cb_far  = cb["far_cases"]
    bd_near = bd["near_cases"]
    bd_far  = bd["far_cases"]

    cb_snip = cb_near[0] if cb_near else {}
    bd_snip = bd_near[0] if bd_near else {}

    lines = [
        f"# Cat. 2 — Cobot Safety + Brake Disc QC: Instrument Calibration Report",
        f"<!-- Generated: {today} | failure_modes: identity_contamination + brittle_autopass -->",
        "",
        "---",
        "",
        "## Caso A: Cobot Safety Clearance",
        "",
        "**Failure mode:** `identity_contamination` — worker identity drives safety clearance.",
        "**Primary signal:** `constraint_enforcement` (KernelSHAP attribution).",
        "**Dual-regulation:** same snippet answers Cat. 2 (safety) + Cat. 4 (discrimination).",
        "",
        "### Calibration results",
        "",
        f"| Metric | Near-threshold | Far-from-threshold |",
        f"|---|---|---|",
        f"| Cases analysed | {len(cb_near)} | {len(cb_far)} |",
        f"| Mean flip-rate | {cb.get('near_flip_mean',0):.3f} ± {cb.get('near_flip_std',0):.3f} | {cb.get('far_flip_mean',0):.3f} ± {cb.get('far_flip_std',0):.3f} |",
        f"| Mean proxy attribution | {cb.get('near_proxy_mean',0):.3f} ± {cb.get('near_proxy_std',0):.3f} | {cb.get('far_proxy_mean',0):.3f} ± {cb.get('far_proxy_std',0):.3f} |",
        f"| Mean legitimate ratio | {cb.get('near_legit_mean',0):.3f} | {cb.get('far_legit_mean',0):.3f} |",
        f"| Mean envelope distance | {cb.get('near_env_mean',0):.3f} ± {cb.get('near_env_std',0):.3f} | {cb.get('far_env_mean',0):.3f} ± {cb.get('far_env_std',0):.3f} |",
        f"| Mean attr. stability (σ rank) | {cb.get('near_astab_mean',0):.3f} ± {cb.get('near_astab_std',0):.3f} | {cb.get('far_astab_mean',0):.3f} ± {cb.get('far_astab_std',0):.3f} |",
        f"| Flagged (%) | {cb.get('near_flagged_pct',0):.1%} | {cb.get('far_flagged_pct',0):.1%} |",
        "",
        "**Discrimination tests (Mann-Whitney U):**",
        f"- Flip-rate near vs. far: p = {fmt_p(cb.get('flip_mwu_p',1))} → {'DISCRIMINATES' if cb.get('flip_discriminates') else 'does not discriminate'}",
        f"- Envelope distance near vs. far: p = {fmt_p(cb.get('env_mwu_p',1))} → {'DISCRIMINATES' if cb.get('env_discriminates') else 'does not discriminate (expected: identity contamination does not shift the input manifold)'}",
        f"- Attr. stability near vs. far: p = {fmt_p(cb.get('astab_mwu_p',1))} → {'DISCRIMINATES' if cb.get('astab_discriminates') else 'does not discriminate (expected: stability is a transversal gate, not a failure-mode detector)'}",
        "",
        "### Evidence snippet (Caso A)",
        "",
        "```json",
        json.dumps(cb_snip, indent=2, ensure_ascii=False),
        "```",
        "",
        "---",
        "",
        "## Caso B: Brake Disc QC",
        "",
        "**Failure mode:** `brittle_autopass` — environmental confounders produce fragile auto-passes.",
        "**Primary signal:** `decision_robustness` (flip-rate under bounded perturbation).",
        "",
        "### Calibration results",
        "",
        f"| Metric | Near-threshold | Far-from-threshold |",
        f"|---|---|---|",
        f"| Cases analysed | {len(bd_near)} | {len(bd_far)} |",
        f"| Mean flip-rate | {bd.get('near_flip_mean',0):.3f} ± {bd.get('near_flip_std',0):.3f} | {bd.get('far_flip_mean',0):.3f} ± {bd.get('far_flip_std',0):.3f} |",
        f"| Mean proxy attribution | {bd.get('near_proxy_mean',0):.3f} ± {bd.get('near_proxy_std',0):.3f} | {bd.get('far_proxy_mean',0):.3f} ± {bd.get('far_proxy_std',0):.3f} |",
        f"| Mean legitimate ratio | {bd.get('near_legit_mean',0):.3f} | {bd.get('far_legit_mean',0):.3f} |",
        f"| Mean envelope distance | {bd.get('near_env_mean',0):.3f} ± {bd.get('near_env_std',0):.3f} | {bd.get('far_env_mean',0):.3f} ± {bd.get('far_env_std',0):.3f} |",
        f"| Mean attr. stability (σ rank) | {bd.get('near_astab_mean',0):.3f} ± {bd.get('near_astab_std',0):.3f} | {bd.get('far_astab_mean',0):.3f} ± {bd.get('far_astab_std',0):.3f} |",
        f"| Flagged (%) | {bd.get('near_flagged_pct',0):.1%} | {bd.get('far_flagged_pct',0):.1%} |",
        "",
        "**Discrimination tests (Mann-Whitney U):**",
        f"- Flip-rate near vs. far: p = {fmt_p(bd.get('flip_mwu_p',1))} → {'DISCRIMINATES' if bd.get('flip_discriminates') else 'does not discriminate'}",
        f"- Envelope distance near vs. far: p = {fmt_p(bd.get('env_mwu_p',1))} → {'DISCRIMINATES' if bd.get('env_discriminates') else 'does not discriminate'}",
        f"- Attr. stability near vs. far: p = {fmt_p(bd.get('astab_mwu_p',1))} → {'DISCRIMINATES' if bd.get('astab_discriminates') else 'does not discriminate'}",
        "",
        "### Evidence snippet (Caso B)",
        "",
        "```json",
        json.dumps(bd_snip, indent=2, ensure_ascii=False),
        "```",
        "",
        "---",
        "",
        "## Síntesis Cat. 2",
        "",
        "| | Cobot Safety | Brake Disc QC |",
        "|---|---|---|",
        "| Failure mode | identity_contamination | brittle_autopass |",
        "| Primary signal | constraint_enforcement | decision_robustness |",
        "| Proxy type | worker identity (Cat. 4) | environmental confounder |",
        "| Envelope discriminates | No (expected) | See above |",
        "| Dual regulation | Cat. 2 + Cat. 4 | Cat. 2 only |",
        "",
        "One protocol, one snippet structure, two failure modes in the same Annex III",
        "category. The primary signal varies by domain: attribution decomposition detects",
        "identity contamination; flip-rate detects confounder-driven fragility.",
        "Non-compensatory evaluation: each signal is assessed on its own axis.",
    ]
    return "\n".join(lines)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    today = date.today().strftime("%Y%m%d")
    print("=" * 60)
    print("Cat. 2 Instrumentation Analysis — Cobot + Brake Disc")
    print("=" * 60)

    print("\n[1/2] Cobot Safety Clearance ...")
    df_cb = generate_cobot_dataset()
    cb = run_case_analysis(
        name="Cobot Safety Clearance",
        df=df_cb,
        feat_names=CB_FEAT,
        g_d=CB_GD,
        g_p=CB_GP,
        coded=CB_CE,
        failure_mode="identity_contamination",
        demonstrator="cobot_safety",
        sustainability_fn=sustainability_signal_cobot,
        teaming_fn=_cobot_teaming,
    )
    print(f"  Accuracy: {cb['accuracy']:.3f} | near: {cb['n_near']} | far: {cb['n_far']}")
    print(f"  Flip-rate p={cb.get('flip_mwu_p',1):.2e}  "
          f"Env p={cb.get('env_mwu_p',1):.2e}  "
          f"Stab p={cb.get('astab_mwu_p',1):.2e}")

    print("\n[2/2] Brake Disc QC ...")
    df_bd = generate_brake_disc_dataset()
    bd = run_case_analysis(
        name="Brake Disc QC",
        df=df_bd,
        feat_names=BD_FEAT,
        g_d=BD_GD,
        g_p=BD_GP,
        coded=BD_CE,
        failure_mode="brittle_autopass",
        demonstrator="brake_disc_qc",
        sustainability_fn=sustainability_signal_brake_disc,
        teaming_fn=_brake_teaming,
        knn_bg_col=BD_KNN_COL,       # "shift"
        knn_bg_val=BD_KNN_VAL,       # 0 = day shift
        knn_subspace=BD_KNN_SUBSPACE, # confounder dims [9,10,11,12]
    )
    print(f"  Accuracy: {bd['accuracy']:.3f} | near: {bd['n_near']} | far: {bd['n_far']}")
    print(f"  Flip-rate p={bd.get('flip_mwu_p',1):.2e}  "
          f"Env p={bd.get('env_mwu_p',1):.2e}  "
          f"Stab p={bd.get('astab_mwu_p',1):.2e}")

    # ── save markdown ──
    ANALYSIS_HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    out_path = ANALYSIS_HISTORY_DIR / f"cobot_brake_disc_analysis_{today}.md"
    md = build_markdown(cb, bd, today)
    out_path.write_text(md, encoding="utf-8")
    print(f"\nReport saved → {out_path}")

    # ── print example snippets ──
    print("\n── Example snippet: Cobot (near-threshold) ──")
    if cb["near_cases"]:
        print(json.dumps(cb["near_cases"][0], indent=2))
    print("\n── Example snippet: Brake Disc (near-threshold) ──")
    if bd["near_cases"]:
        print(json.dumps(bd["near_cases"][0], indent=2))


if __name__ == "__main__":
    main()
