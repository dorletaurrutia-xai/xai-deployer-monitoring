"""Cat. 6 — Law Enforcement: COMPAS Pretrial Non-Criminal-History Contamination.

Produces:
- Console calibration summary
- JSON evidence snippets with 7 signals (5 protocol + 2 Industry 5.0)
- Markdown report saved to analysis_history/compas_pretrial_analysis_YYYYMMDD.md

Failure mode: non_criminal_history_contamination — protected demographic
attributes (race, age, sex) contaminate the pretrial risk score. Due process
requires risk assessment to be based on criminal history only; the instrument
recovers how much each score traces to non-criminal-history features.

Primary signal:   constraint_enforcement (race + age + sex attribution)
Secondary signal: decision_robustness    (flip-rate near vs. far)

Industry 5.0 thresholds:
  sustainability FLAG  if race attribution > 0.10 (coded: 11-17%)
  human_ai_teaming:
    mandatory_review  always (score is advisory only — judicial decision must not substitute)
"""

import json
import sys
from datetime import date
from math import erfc, sqrt
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

# ── path setup ────────────────────────────────────────────────────────────────
THIS_DIR = Path(__file__).resolve().parent
DEMONSTRATORS_DIR = THIS_DIR.parent
REPO_DIR = DEMONSTRATORS_DIR.parent
sys.path.insert(0, str(DEMONSTRATORS_DIR))
sys.path.insert(0, str(REPO_DIR))

from cat6_compas_pretrial.generate_dataset import (
    generate_compas_dataset,
    FEATURE_NAMES,
    G_DOMAIN,
    G_PROXY,
    CODED_EFFECTS,
)
from instrumentation.protocol import InstrumentationProtocol

ANALYSIS_HISTORY_DIR = REPO_DIR.parent / "xai-analysis-history" / "cat6_compas_pretrial"


# ── stats primitives ──────────────────────────────────────────────────────────

def mwu(x, y):
    """One-sided Mann-Whitney U (tests x > y). Returns (U, p)."""
    nx, ny = len(x), len(y)
    combined = np.concatenate([x, y])
    ranks = np.argsort(np.argsort(combined)) + 1.0
    U = np.sum(ranks[:nx]) - nx * (nx + 1) / 2
    mu = nx * ny / 2
    sigma = sqrt(nx * ny * (nx + ny + 1) / 12)
    z = (U - mu) / (sigma + 1e-10)
    return U, 0.5 * erfc(z / sqrt(2))


# ── Industry 5.0 signal builders ──────────────────────────────────────────────

def sustainability_signal(proxy_attr: dict) -> dict:
    """FLAG if race attribution > 0.10 (coded: 11-17%)."""
    race_val = proxy_attr.get("race", 0.0)
    return {
        "dimension": "criminal_justice_equity",
        "at_risk_group": "racial_minorities",
        "indicator": "race_pretrial_attribution",
        "value": round(race_val, 4),
        "status": "FLAG" if race_val > 0.10 else "OK",
    }


def human_ai_teaming_signal(fr_val: float, is_near: bool) -> dict:
    # Pretrial risk score is advisory only: judicial review is always mandatory
    if fr_val > 0.3:
        level, status = "mandatory_review", "FLAG"
    else:
        level, status = "mandatory_review", "OK"
    return {
        "dimension": "judicial_oversight",
        "escalation_level": level,
        "trigger": "near_threshold_non_criminal_history",
        "human_override_available": True,
        "note": "COMPAS: score is advisory only — judicial decision must not substitute algorithmic output",
        "status": status,
    }


# ── core analysis ──────────────────────────────────────────────────────────────

def run_analysis(n_cases: int = 40) -> dict:
    df = generate_compas_dataset()

    X = df[FEATURE_NAMES].values.astype(float)
    y = df["label"].values.astype(float)
    mu, sd = X.mean(0), X.std(0) + 1e-10
    Xn = (X - mu) / sd

    np.random.seed(42)
    idx = np.random.permutation(len(X))
    sp = int(0.7 * len(X))
    Xtr, Xte = Xn[idx[:sp]], Xn[idx[sp:]]
    ytr, yte = y[idx[:sp]], y[idx[sp:]]

    model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    model.fit(Xtr, ytr)
    acc = float(np.mean((model.predict_proba(Xte)[:, 1] >= 0.5) == yte))
    scores = model.predict_proba(Xte)[:, 1]
    margins = np.abs(scores - 0.5)

    near_idx = np.where(margins < 0.15)[0]
    far_idx  = np.where(margins > 0.35)[0]

    protocol = InstrumentationProtocol(model, Xtr, FEATURE_NAMES, G_DOMAIN, G_PROXY)

    results = {
        "demonstrator": "compas_pretrial",
        "failure_mode": "non_criminal_history_contamination",
        "annex_iii_category": "cat_6",
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
            snippet = protocol.generate_snippet(
                x,
                case_id=f"compas_pretrial-{label}-{int(j):04d}",
                tau_stab=0.3,
            )

            ce = snippet.constraint_enforcement
            proxy_attr = ce.get("proxy_attribution", {})
            proxy_total = round(sum(proxy_attr.values()), 4)
            top4 = list(ce.get("top_features", {}).keys())
            legit_ratio = round(ce.get("legitimate_feature_ratio", 0.0), 4)

            dr = snippet.decision_robustness
            fr_val = dr["flip_rate"]

            ev = snippet.envelope_validity

            ri = snippet.record_integrity
            astab = ri.get("attribution_stability_sigma") or 0.0
            ri_status = "BLOCK" if astab > 0.3 else "OK"

            case = {
                "case_id": snippet.case_id,
                "demonstrator": "compas_pretrial",
                "failure_mode": "non_criminal_history_contamination",
                "annex_iii_category": "cat_6",
                "output_score": round(snippet.output_score, 4),
                "threshold": snippet.threshold,
                "margin_to_threshold": round(snippet.margin_to_threshold, 4),
                "constraint_enforcement": {
                    "legitimate_feature_ratio": legit_ratio,
                    "proxy_attribution_total": proxy_total,
                    "proxy_detail": {k: round(v, 4) for k, v in proxy_attr.items()},
                    "top_features": top4,
                    "status": ce.get("status", "OK"),
                },
                "envelope_validity": {
                    "distance_to_nearest_prototype": round(
                        ev.get("distance_to_nearest_prototype", 0), 4
                    ),
                    "status": ev.get("status", "OK"),
                },
                "decision_robustness": {
                    "flip_rate": round(fr_val, 4),
                    "status": dr.get("status", "OK"),
                },
                "record_integrity": {
                    "attribution_stability_sigma": round(astab, 4),
                    "status": ri_status,
                },
                "sustainability_impact": sustainability_signal(proxy_attr),
                "human_ai_teaming": human_ai_teaming_signal(fr_val, is_near),
                "action_verdict": snippet.action_verdict,
            }
            results[f"{label}_cases"].append(case)

    # ── aggregate stats ──
    for lbl in ["near", "far"]:
        cases = results[f"{lbl}_cases"]
        if cases:
            results[f"{lbl}_flip_mean"]   = float(np.mean([c["decision_robustness"]["flip_rate"] for c in cases]))
            results[f"{lbl}_flip_std"]    = float(np.std( [c["decision_robustness"]["flip_rate"] for c in cases]))
            results[f"{lbl}_proxy_mean"]  = float(np.mean([c["constraint_enforcement"]["proxy_attribution_total"] for c in cases]))
            results[f"{lbl}_proxy_std"]   = float(np.std( [c["constraint_enforcement"]["proxy_attribution_total"] for c in cases]))
            results[f"{lbl}_legit_mean"]  = float(np.mean([c["constraint_enforcement"]["legitimate_feature_ratio"] for c in cases]))
            results[f"{lbl}_env_mean"]    = float(np.mean([c["envelope_validity"]["distance_to_nearest_prototype"] for c in cases]))
            results[f"{lbl}_env_std"]     = float(np.std( [c["envelope_validity"]["distance_to_nearest_prototype"] for c in cases]))
            results[f"{lbl}_astab_mean"]  = float(np.mean([c["record_integrity"]["attribution_stability_sigma"] for c in cases]))
            results[f"{lbl}_astab_std"]   = float(np.std( [c["record_integrity"]["attribution_stability_sigma"] for c in cases]))
            results[f"{lbl}_flagged_pct"] = float(np.mean([c["action_verdict"] != "PROCEED" for c in cases]))
            # race attribution specifically
            results[f"{lbl}_race_mean"]   = float(np.mean([
                c["constraint_enforcement"]["proxy_detail"].get("race", 0)
                for c in cases
            ]))
            results[f"{lbl}_race_std"]    = float(np.std([
                c["constraint_enforcement"]["proxy_detail"].get("race", 0)
                for c in cases
            ]))

    # ── Mann-Whitney U ──
    nc, fc = results["near_cases"], results["far_cases"]
    if len(nc) > 1 and len(fc) > 1:
        for metric_key, get_val in [
            ("flip",  lambda c: c["decision_robustness"]["flip_rate"]),
            ("env",   lambda c: c["envelope_validity"]["distance_to_nearest_prototype"]),
            ("astab", lambda c: c["record_integrity"]["attribution_stability_sigma"]),
        ]:
            _, p = mwu(
                np.array([get_val(c) for c in nc]),
                np.array([get_val(c) for c in fc]),
            )
            results[f"{metric_key}_mwu_p"] = p
            results[f"{metric_key}_discriminates"] = bool(p < 0.01)

    return results


# ── report builder ─────────────────────────────────────────────────────────────

def fmt_p(p: float) -> str:
    return f"{p:.2e}"


def build_markdown(res: dict, today: str) -> str:
    near = res["near_cases"]
    far  = res["far_cases"]
    flag_cases = [c for c in near if c.get("sustainability_impact", {}).get("status") == "FLAG"]
    snip = flag_cases[0] if flag_cases else (near[0] if near else {})

    def disc(key):
        if res.get(key):
            return "DISCRIMINATES (p < 0.01)"
        return "does not discriminate"

    lines = [
        "# Cat. 6 — Law Enforcement: COMPAS Pretrial Non-Criminal-History Contamination",
        f"<!-- Generated: {today} | failure_mode: non_criminal_history_contamination -->",
        "",
        "---",
        "",
        "**Failure mode:** `non_criminal_history_contamination` — protected demographic "
        "attributes (race, age, sex) contaminate the pretrial risk score. Due process "
        "requires risk assessment based on criminal history only. Inspired by ProPublica's "
        "COMPAS analysis (Angwin et al., 2016): race attribution 11-17% despite being "
        "excluded from the score's stated inputs.",
        "",
        "**Primary signal:** `constraint_enforcement` (KernelSHAP attribution).",
        "**Regulation:** EU AI Act Annex III Category 6 (law enforcement).",
        "",
        "### Calibration results",
        "",
        "| Metric | Near-threshold | Far-from-threshold |",
        "|---|---|---|",
        f"| Cases analysed | {len(near)} | {len(far)} |",
        f"| Mean flip-rate | {res.get('near_flip_mean',0):.3f} ± {res.get('near_flip_std',0):.3f} | {res.get('far_flip_mean',0):.3f} ± {res.get('far_flip_std',0):.3f} |",
        f"| Mean proxy attribution | {res.get('near_proxy_mean',0):.3f} ± {res.get('near_proxy_std',0):.3f} | {res.get('far_proxy_mean',0):.3f} ± {res.get('far_proxy_std',0):.3f} |",
        f"| Mean race attribution | {res.get('near_race_mean',0):.3f} ± {res.get('near_race_std',0):.3f} | {res.get('far_race_mean',0):.3f} ± {res.get('far_race_std',0):.3f} |",
        f"| Mean legitimate ratio | {res.get('near_legit_mean',0):.3f} | {res.get('far_legit_mean',0):.3f} |",
        f"| Mean envelope distance | {res.get('near_env_mean',0):.3f} ± {res.get('near_env_std',0):.3f} | {res.get('far_env_mean',0):.3f} ± {res.get('far_env_std',0):.3f} |",
        f"| Mean attr. stability (σ rank) | {res.get('near_astab_mean',0):.3f} ± {res.get('near_astab_std',0):.3f} | {res.get('far_astab_mean',0):.3f} ± {res.get('far_astab_std',0):.3f} |",
        f"| Flagged (%) | {res.get('near_flagged_pct',0):.1%} | {res.get('far_flagged_pct',0):.1%} |",
        "",
        "**Discrimination tests (Mann-Whitney U):**",
        f"- Flip-rate near vs. far: p = {fmt_p(res.get('flip_mwu_p',1))} → {disc('flip_discriminates')}",
        f"- Envelope distance near vs. far: p = {fmt_p(res.get('env_mwu_p',1))} → {disc('env_discriminates')}",
        f"- Attr. stability near vs. far: p = {fmt_p(res.get('astab_mwu_p',1))} → {disc('astab_discriminates')}",
        "",
        "### Evidence snippet (near-threshold, sustainability FLAG preferred)",
        "",
        "```json",
        json.dumps(snip, indent=2, ensure_ascii=False),
        "```",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    today = date.today().strftime("%Y%m%d")
    print("=" * 60)
    print("Cat. 6 Instrumentation Analysis — COMPAS Pretrial")
    print("=" * 60)

    res = run_analysis(n_cases=40)

    print(f"  Accuracy: {res['accuracy']:.3f} | near: {res['n_near']} | far: {res['n_far']}")
    print(f"  Flip p={res.get('flip_mwu_p',1):.2e}  "
          f"Env p={res.get('env_mwu_p',1):.2e}  "
          f"Stab p={res.get('astab_mwu_p',1):.2e}")
    print(f"  Proxy near={res.get('near_proxy_mean',0):.3f}±{res.get('near_proxy_std',0):.3f}  "
          f"far={res.get('far_proxy_mean',0):.3f}±{res.get('far_proxy_std',0):.3f}")
    print(f"  Race  near={res.get('near_race_mean',0):.3f}±{res.get('near_race_std',0):.3f}  "
          f"far={res.get('far_race_mean',0):.3f}±{res.get('far_race_std',0):.3f}")
    print(f"  Flagged near={res.get('near_flagged_pct',0):.1%}  "
          f"far={res.get('far_flagged_pct',0):.1%}")

    ANALYSIS_HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    out_path = ANALYSIS_HISTORY_DIR / f"compas_pretrial_analysis_{today}.md"
    out_path.write_text(build_markdown(res, today), encoding="utf-8")
    print(f"\nReport saved → {out_path}")

    print("\n── Example snippet (sustainability FLAG preferred) ──")
    near_cases = res["near_cases"]
    flag_cases = [c for c in near_cases if c.get("sustainability_impact", {}).get("status") == "FLAG"]
    snip = flag_cases[0] if flag_cases else (near_cases[0] if near_cases else None)
    if snip:
        print(json.dumps(snip, indent=2))
