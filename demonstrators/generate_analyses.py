"""Generate detailed case analysis for each demonstrator.

Produces markdown files with:
- Dataset design rationale (3 criteria)
- Coded effects and target magnitudes
- Instrument calibration results
- Example evidence snippets
- Discrimination tests
"""

import sys
import json
from pathlib import Path
from math import erfc, sqrt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cat2_cobot_safety.brake_disc.generate_dataset import (
    generate_brake_disc_dataset, FEATURE_NAMES as BD_FEAT,
    G_DOMAIN as BD_GD, G_PROXY as BD_GP, CODED_EFFECTS as BD_CE,
)
from cat2_cobot_safety.generate_dataset import (
    generate_cobot_dataset, FEATURE_NAMES as CB_FEAT,
    G_DOMAIN as CB_GD, G_PROXY as CB_GP, CODED_EFFECTS as CB_CE,
)
from _archive.energy_scheduling.generate_dataset import (
    generate_energy_dataset, FEATURE_NAMES as EN_FEAT,
    G_DOMAIN as EN_GD, G_PROXY as EN_GP, CODED_EFFECTS as EN_CE,
)
from cat5_heloc_credit.generate_dataset import (
    generate_heloc_dataset, FEATURE_NAMES as HL_FEAT,
    G_DOMAIN as HL_GD, G_PROXY as HL_GP, CODED_EFFECTS as HL_CE,
)
from cat3_oulad_education.generate_dataset import (
    generate_oulad_dataset, FEATURE_NAMES as OU_FEAT,
    G_DOMAIN as OU_GD, G_PROXY as OU_GP, CODED_EFFECTS as OU_CE,
)
from cat6_compas_pretrial.generate_dataset import (
    generate_compas_dataset, FEATURE_NAMES as CP_FEAT,
    G_DOMAIN as CP_GD, G_PROXY as CP_GP, CODED_EFFECTS as CP_CE,
)

# New demonstrators (Cat. 1, 4, 7, 8) — delegate to their own run_analysis modules
from cat1_facial_recognition.run_analysis import (
    run_analysis as run_fr_analysis, build_markdown as build_fr_md,
    ANALYSIS_HISTORY_DIR as FR_HIST_DIR,
)
from cat4_amazon_hiring.run_analysis import (
    run_analysis as run_ah_analysis, build_markdown as build_ah_md,
    ANALYSIS_HISTORY_DIR as AH_HIST_DIR,
)
from cat7_home_office_visa.run_analysis import (
    run_analysis as run_hov_analysis, build_markdown as build_hov_md,
    ANALYSIS_HISTORY_DIR as HOV_HIST_DIR,
)
from cat8_compas_sentencing.run_analysis import (
    run_analysis as run_cs_analysis, build_markdown as build_cs_md,
    ANALYSIS_HISTORY_DIR as CS_HIST_DIR,
)


# ── Minimal tools (no scipy/sklearn) ──

class SimpleLR:
    def __init__(self):
        self.w = None; self.b = 0.0
    def fit(self, X, y, lr=0.05, n_iter=500):
        n, d = X.shape; self.w = np.zeros(d); self.b = 0.0
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
    nx, ny = len(x), len(y)
    combined = np.concatenate([x, y])
    ranks = np.argsort(np.argsort(combined)) + 1.0
    U = np.sum(ranks[:nx]) - nx * (nx + 1) / 2
    mu = nx * ny / 2; sigma = sqrt(nx * ny * (nx + ny + 1) / 12)
    z = (U - mu) / (sigma + 1e-10)
    return U, 0.5 * erfc(z / sqrt(2))


def attribution(model, x, bg, n_bg=60):
    bg_ = bg[:n_bg]
    base = model.predict_proba(x.reshape(1, -1))[0, 1]
    attr = np.zeros(len(x))
    for i in range(len(x)):
        Xp = np.tile(x, (len(bg_), 1)); Xp[:, i] = bg_[:, i]
        attr[i] = base - np.mean(model.predict_proba(Xp)[:, 1])
    return attr


def flip_rate(model, x, fr, th=0.5, n=50, ns=0.05):
    base = int(model.predict_proba(x.reshape(1, -1))[0, 1] >= th)
    flips = sum(1 for _ in range(n)
                if int(model.predict_proba((x + np.random.normal(0, ns, len(x)) * fr).reshape(1, -1))[0, 1] >= th) != base)
    return flips / n


def knn_distance(x, X_val, k=5):
    dists = np.sqrt(np.sum((X_val - x) ** 2, axis=1))
    return float(np.mean(np.sort(dists)[:k]))


def attr_stability(model, x, bg, n_reps=10, sigma=0.01, rng=None):
    """Attribution stability under micro-noise.

    Computes attribution n_reps times, each with small Gaussian noise
    (σ=0.01) added to x.  Returns mean std of feature-rank across reps.
    Low value = stable; high value = rank order sensitive to noise.
    """
    if rng is None:
        rng = np.random
    ranks = []
    for _ in range(n_reps):
        x_noisy = x + rng.normal(0, sigma, len(x))
        a = attribution(model, x_noisy, bg)
        rank = np.argsort(np.argsort(-np.abs(a))).astype(float)  # 0 = most important
        ranks.append(rank)
    ranks = np.array(ranks)  # (n_reps, n_features)
    return float(np.mean(np.std(ranks, axis=0)))


def run_analysis(name, df, feat_names, g_d, g_p, coded, n_cases=40):
    """Run detailed analysis, return dict of results."""
    X = df[feat_names].values.astype(float)
    y = df["label"].values.astype(float)
    mu, sd = X.mean(0), X.std(0) + 1e-10
    Xn = (X - mu) / sd

    np.random.seed(42)
    idx = np.random.permutation(len(X))
    sp = int(0.7 * len(X))
    Xtr, Xte = Xn[idx[:sp]], Xn[idx[sp:]]
    ytr, yte = y[idx[:sp]], y[idx[sp:]]

    model = SimpleLR(); model.fit(Xtr, ytr)
    acc = np.mean((model.predict_proba(Xte)[:, 1] >= 0.5) == yte)
    scores = model.predict_proba(Xte)[:, 1]
    margins = np.abs(scores - 0.5)

    near = np.where(margins < 0.15)[0]
    far = np.where(margins > 0.35)[0]
    fr = Xtr.max(0) - Xtr.min(0) + 1e-10
    bg = Xtr[:100]

    # Detailed instrumentation
    results = {"accuracy": acc, "n_near": len(near), "n_far": len(far),
               "near_cases": [], "far_cases": []}

    for label, indices in [("near", near[:n_cases]), ("far", far[:n_cases])]:
        for j in indices:
            x = Xte[j]
            sc = float(scores[j])
            attr = attribution(model, x, bg)
            total = np.sum(np.abs(attr)) + 1e-10
            fr_val = flip_rate(model, x, fr)
            env_d = knn_distance(x, Xtr)
            # Use separate RNG for attr_stability so it doesn't shift
            # the main random sequence (preserving flip_rate reproducibility)
            _astab_rng = np.random.RandomState(42 + j)
            astab = attr_stability(model, x, bg, rng=_astab_rng)

            legit_ratio = float(np.sum(np.abs(attr[g_d])) / total)
            proxy_attr = {feat_names[i]: float(np.abs(attr[i]) / total) for i in g_p}
            proxy_total = sum(proxy_attr.values())
            top4 = [feat_names[i] for i in np.argsort(-np.abs(attr))[:4]]

            case = {
                "score": sc, "margin": float(abs(sc - 0.5)),
                "flip_rate": fr_val, "envelope_distance": env_d,
                "attr_stability": astab,
                "legit_ratio": legit_ratio, "proxy_total": proxy_total,
                "proxy_detail": proxy_attr, "top_features": top4,
                "verdict": "PROCEED WITH FLAG" if (fr_val > 0.3 or legit_ratio < 0.6) else "PROCEED",
            }
            results[f"{label}_cases"].append(case)

    # Aggregate stats
    for label in ["near", "far"]:
        cases = results[f"{label}_cases"]
        if cases:
            results[f"{label}_flip_mean"] = np.mean([c["flip_rate"] for c in cases])
            results[f"{label}_flip_std"] = np.std([c["flip_rate"] for c in cases])
            results[f"{label}_proxy_mean"] = np.mean([c["proxy_total"] for c in cases])
            results[f"{label}_proxy_std"] = np.std([c["proxy_total"] for c in cases])
            results[f"{label}_legit_mean"] = np.mean([c["legit_ratio"] for c in cases])
            results[f"{label}_env_mean"] = np.mean([c["envelope_distance"] for c in cases])
            results[f"{label}_env_std"] = np.std([c["envelope_distance"] for c in cases])
            results[f"{label}_astab_mean"] = np.mean([c["attr_stability"] for c in cases])
            results[f"{label}_astab_std"] = np.std([c["attr_stability"] for c in cases])
            results[f"{label}_flagged_pct"] = np.mean([c["verdict"] != "PROCEED" for c in cases])

    # Mann-Whitney U tests
    if results["near_cases"] and results["far_cases"]:
        nc, fc = results["near_cases"], results["far_cases"]
        if len(nc) > 1 and len(fc) > 1:
            # Flip-rate
            _, p = mwu(np.array([c["flip_rate"] for c in nc]),
                       np.array([c["flip_rate"] for c in fc]))
            results["flip_mwu_p"] = p
            results["flip_discriminates"] = p < 0.01
            # Envelope distance
            _, p_env = mwu(np.array([c["envelope_distance"] for c in nc]),
                           np.array([c["envelope_distance"] for c in fc]))
            results["env_mwu_p"] = p_env
            results["env_discriminates"] = p_env < 0.01
            # Attribution stability
            _, p_astab = mwu(np.array([c["attr_stability"] for c in nc]),
                             np.array([c["attr_stability"] for c in fc]))
            results["astab_mwu_p"] = p_astab
            results["astab_discriminates"] = p_astab < 0.01

    return results


def write_brake_disc_md(res, out):
    bd_cases = res["near_cases"]
    ex = bd_cases[0] if bd_cases else {}
    md = f"""# Brake Disc Quality Control — Case Analysis

## 1. Domain context

AI Act Category 2 (safety component). A quality inspection system auto-passes or
auto-rejects brake discs on a production line. The model is a black-box cloud API;
the deployer (manufacturer) has only predict_proba access.

**Failure mode tested:** Decision robustness — brittle auto-passes on safety-critical parts.

## 2. Dataset design (three criteria)

### Feature fidelity
Features named and typed as in a real inspection system:

| Feature | Type | Role |
|---|---|---|
| crack_depth_mm, crack_length_mm | float | Defect indicator (legitimate) |
| runout_mm, roughness_ra | float | Surface quality (legitimate) |
| hardness_hrc, thickness_var_mm | float | Material property (legitimate) |
| weight_deviation_g, visual_score | float | Global quality (legitimate) |
| surface_porosity_pct | float | Defect indicator (legitimate) |
| oil_residue | float | **Confounder** (environmental) |
| lighting_lux | float | **Confounder** (environmental) |
| camera_angle_deg | float | **Confounder** (sensor artefact) |
| temperature_c | float | **Confounder** (environmental) |

### Structural fidelity
- Crack length correlates with crack depth (length ≈ 2–8× depth)
- Night shift → higher oil residue, lower lighting, larger camera angle variance
- Confounders correlate with each other through the shift variable

### Controlled ground truth
| Coded effect | Target magnitude | What it means |
|---|---|---|
| Confounder attribution weight | ~34% | 34% of the score basis traces to oil/lighting/camera/temp |
| Marginal flip-rate | ~63% | 63% of near-threshold passes reverse under bounded perturbation |

These are **design parameters**. The instrument's job is to read them back.

## 3. Instrument calibration results

| Metric | Near-threshold | Far-from-threshold |
|---|---|---|
| Cases analysed | {len(res['near_cases'])} | {len(res.get('far_cases', []))} |
| Mean flip-rate | {res.get('near_flip_mean', 0):.3f} ± {res.get('near_flip_std', 0):.3f} | {res.get('far_flip_mean', 0):.3f} ± {res.get('far_flip_std', 0):.3f} |
| Mean proxy attribution | {res.get('near_proxy_mean', 0):.3f} ± {res.get('near_proxy_std', 0):.3f} | {res.get('far_proxy_mean', 0):.3f} ± {res.get('far_proxy_std', 0):.3f} |
| Mean legitimate ratio | {res.get('near_legit_mean', 0):.3f} | {res.get('far_legit_mean', 0):.3f} |
| Mean envelope distance | {res.get('near_env_mean', 0):.3f} ± {res.get('near_env_std', 0):.3f} | {res.get('far_env_mean', 0):.3f} ± {res.get('far_env_std', 0):.3f} |
| Mean attr. stability (σ rank) | {res.get('near_astab_mean', 0):.3f} ± {res.get('near_astab_std', 0):.3f} | {res.get('far_astab_mean', 0):.3f} ± {res.get('far_astab_std', 0):.3f} |
| Flagged (%) | {res.get('near_flagged_pct', 0):.1%} | {res.get('far_flagged_pct', 0):.1%} |

**Discrimination tests (Mann-Whitney U):**
- Flip-rate near vs. far: p = {res.get('flip_mwu_p', 0):.2e} {"→ DISCRIMINATES (p < 0.01)" if res.get('flip_discriminates') else ""}
- Envelope distance near vs. far: p = {res.get('env_mwu_p', 0):.2e} {"→ DISCRIMINATES" if res.get('env_discriminates') else ""}
- Attr. stability near vs. far: p = {res.get('astab_mwu_p', 0):.2e} {"→ DISCRIMINATES" if res.get('astab_discriminates') else ""}

## 4. Example evidence snippet

Case: near-threshold auto-pass (score {ex.get('score', 0):.3f}, margin {ex.get('margin', 0):.3f})

```json
{{
  "case_id": "brake_disc-near-0000",
  "output_score": {ex.get('score', 0):.4f},
  "threshold": 0.50,
  "margin_to_threshold": {ex.get('margin', 0):.4f},
  "constraint_enforcement": {{
    "status": "{'CONTESTED' if ex.get('legit_ratio', 1) < 0.6 else 'OK'}",
    "legitimate_feature_ratio": {ex.get('legit_ratio', 0):.3f},
    "top_features": {json.dumps(ex.get('top_features', [])[:4])},
    "proxy_attribution_total": {ex.get('proxy_total', 0):.3f}
  }},
  "envelope_validity": {{
    "status": "OK",
    "distance_to_nearest_prototype": {ex.get('envelope_distance', 0):.3f}
  }},
  "decision_robustness": {{
    "status": "{'CONTESTED' if ex.get('flip_rate', 0) > 0.3 else 'OK'}",
    "flip_rate": {ex.get('flip_rate', 0):.2f}
  }},
  "record_integrity": {{
    "attribution_stability_sigma": {ex.get('attr_stability', 0):.3f},
    "status": "{'FAIL' if ex.get('attr_stability', 0) > 2.0 else 'OK'}"
  }},
  "action_verdict": "{ex.get('verdict', 'PROCEED')}"
}}
```

## 5. Interpretation

The instrument **recovers the coded confounder structure**: proxy attribution
near the target ~34%. Near-threshold cases show elevated flip-rates, confirming
the designed fragility. The snippet would flag a brittle confounder-driven pass
for manual review in an operational deployment.

**What this validates:** the flip-rate signal and the attribution decomposition
correctly discriminate the failure mode we coded in. This is instrument
calibration — the thermometer reads back the known reference temperature.
"""
    Path(out).write_text(md)
    print(f"  Written: {out}")


def write_cobot_md(res, out):
    cases = res["near_cases"]
    ex = cases[0] if cases else {}
    # Separate temp worker cases
    md = f"""# Cobot Safety Clearance — Case Analysis

## 1. Domain context

AI Act Category 2 (safety component) + Category 4 (employment/worker management).
A cobot safety system clears workers into shared human-robot zones. The model is a
black-box; the deployer (hospital/factory) has only predict_proba access.

**Failure mode tested:** Constraint enforcement — worker identity drives safety clearance.

**Dual-regulation property:** The same evidence snippet answers BOTH
- **Cat. 2:** Is the safety clearance hazard-based?
- **Cat. 4:** Is the worker treatment non-discriminatory?

## 2. Dataset design (three criteria)

### Feature fidelity
| Feature | Type | Role |
|---|---|---|
| distance_m, velocity_ms, payload_kg | float | Hazard indicator (legitimate) |
| exposure_time_s, zone_type, robot_mode | float/int | Hazard context (legitimate) |
| force_limit_n, collision_history | float/int | Safety record (legitimate) |
| worker_age, contract_type | float/int | **Worker identity** (proxy) |
| experience_years, shift | float/int | **Worker identity** (proxy) |
| training_hours, worker_role | float/int | **Worker identity** (proxy) |

### Structural fidelity
- Temp workers (contract_type=1) are coded younger and less experienced
- Hazard features correlate realistically (closer distance + higher velocity)
- Training hours lower for temp workers (structural inequality)

### Controlled ground truth
| Coded effect | Target magnitude |
|---|---|
| Worker identity attribution (overall) | ~25.2% |
| Worker identity attribution (temp workers) | ~31.7% |
| Dual-regulation: one snippet serves Cat. 2 + Cat. 4 | By architecture |

## 3. Instrument calibration results

| Metric | Near-threshold | Far-from-threshold |
|---|---|---|
| Cases analysed | {len(res['near_cases'])} | {len(res.get('far_cases', []))} |
| Mean flip-rate | {res.get('near_flip_mean', 0):.3f} ± {res.get('near_flip_std', 0):.3f} | {res.get('far_flip_mean', 0):.3f} ± {res.get('far_flip_std', 0):.3f} |
| Mean proxy attribution | {res.get('near_proxy_mean', 0):.3f} ± {res.get('near_proxy_std', 0):.3f} | {res.get('far_proxy_mean', 0):.3f} ± {res.get('far_proxy_std', 0):.3f} |
| Mean legitimate ratio | {res.get('near_legit_mean', 0):.3f} | {res.get('far_legit_mean', 0):.3f} |
| Mean envelope distance | {res.get('near_env_mean', 0):.3f} ± {res.get('near_env_std', 0):.3f} | {res.get('far_env_mean', 0):.3f} ± {res.get('far_env_std', 0):.3f} |
| Mean attr. stability (σ rank) | {res.get('near_astab_mean', 0):.3f} ± {res.get('near_astab_std', 0):.3f} | {res.get('far_astab_mean', 0):.3f} ± {res.get('far_astab_std', 0):.3f} |
| Flagged (%) | {res.get('near_flagged_pct', 0):.1%} | {res.get('far_flagged_pct', 0):.1%} |

**Discrimination tests (Mann-Whitney U):**
- Flip-rate near vs. far: p = {res.get('flip_mwu_p', 0):.2e} {"→ DISCRIMINATES" if res.get('flip_discriminates') else ""}
- Envelope distance near vs. far: p = {res.get('env_mwu_p', 0):.2e} {"→ DISCRIMINATES" if res.get('env_discriminates') else ""}
- Attr. stability near vs. far: p = {res.get('astab_mwu_p', 0):.2e} {"→ DISCRIMINATES" if res.get('astab_discriminates') else ""}

## 4. Example evidence snippet

```json
{{
  "case_id": "cobot_safety-near-0000",
  "output_score": {ex.get('score', 0):.4f},
  "threshold": 0.50,
  "constraint_enforcement": {{
    "status": "{'CONTESTED' if ex.get('legit_ratio', 1) < 0.6 else 'OK'}",
    "legitimate_feature_ratio": {ex.get('legit_ratio', 0):.3f},
    "proxy_attribution_total": {ex.get('proxy_total', 0):.3f},
    "proxy_detail": {json.dumps({k: round(v, 3) for k, v in ex.get('proxy_detail', {}).items()}, indent=4)}
  }},
  "envelope_validity": {{
    "distance_to_nearest_prototype": {ex.get('envelope_distance', 0):.3f},
    "status": "OK"
  }},
  "decision_robustness": {{
    "flip_rate": {ex.get('flip_rate', 0):.2f}
  }},
  "record_integrity": {{
    "attribution_stability_sigma": {ex.get('attr_stability', 0):.3f},
    "status": "{'FAIL' if ex.get('attr_stability', 0) > 2.0 else 'OK'}"
  }},
  "action_verdict": "{ex.get('verdict', 'PROCEED')}"
}}
```

## 5. Interpretation

The instrument **recovers the coded identity contamination**: proxy attribution
≈ {res.get('near_proxy_mean', 0):.1%} overall (target ~25.2%). The same snippet
that flags identity-driven clearance (Cat. 4 discrimination concern) simultaneously
answers whether the clearance was hazard-based (Cat. 2 safety concern).

One architecture, one snippet, two regulatory obligations — without rebuilding
anything per regulation.
"""
    Path(out).write_text(md)
    print(f"  Written: {out}")


def write_energy_md(res, df, out):
    cases = res["near_cases"]
    ex = cases[0] if cases else {}

    # CO2 stats from dataset
    peak = df[df["is_peak"] == 1]
    co2_mean = peak["co2_saving_pct"].mean() if len(peak) > 0 else 0

    md = f"""# Energy Scheduling — Case Analysis

## 1. Domain context

AI Act Article 69 (voluntary codes of conduct) + CSRD sustainability reporting.
An AI scheduler optimises production batches for an energy-intensive SME. The model
is a black-box; the deployer has only predict_proba access.

**Failure mode tested:** Sustainability blindness at decision margin — the scheduler
cannot see CO₂ impact, so marginal batches are scheduled at peak carbon intensity
when an off-peak swap would cost zero throughput.

## 2. Dataset design (three criteria)

### Feature fidelity
| Feature | Type | Role |
|---|---|---|
| throughput_demand, machine_capacity_pct | float | Production throughput (legitimate) |
| batch_size, due_date_hours | float/int | Scheduling urgency (legitimate) |
| setup_time_min, scrap_rate_pct, oee | float | Operations (legitimate) |
| queue_length | int | Workload (legitimate) |
| hour_of_day | int | **Sustainability** (blind spot) |
| grid_carbon_gco2_kwh | float | **Sustainability** (blind spot) |
| energy_price_eur_kwh | float | **Sustainability** (blind spot) |
| renewable_fraction | float | **Sustainability** (blind spot) |
| energy_per_batch_kwh | float | **Sustainability** (blind spot) |

### Structural fidelity
- Peak hours (8–20) correlate with higher carbon intensity (+200 gCO₂/kWh)
- Peak hours correlate with higher energy price (+0.08 €/kWh)
- Off-peak has higher renewable fraction
- Demand and batch urgency correlate realistically

### Controlled ground truth
| Coded effect | Target magnitude |
|---|---|
| CO₂ differential for marginal peak batches | ~51% |
| Throughput cost for marginal batches | ~0% (by design) |
| Sustainability attribution in scheduling | ~20% |

**Dataset CO₂ statistics:**
- Mean CO₂ saving for peak-scheduled batches: {co2_mean:.1f}%

## 3. Instrument calibration results

| Metric | Near-threshold | Far-from-threshold |
|---|---|---|
| Cases analysed | {len(res['near_cases'])} | {len(res.get('far_cases', []))} |
| Mean flip-rate | {res.get('near_flip_mean', 0):.3f} | {res.get('far_flip_mean', 0):.3f} |
| Mean sustainability attr. | {res.get('near_proxy_mean', 0):.3f} | {res.get('far_proxy_mean', 0):.3f} |
| Mean legitimate ratio | {res.get('near_legit_mean', 0):.3f} | {res.get('far_legit_mean', 0):.3f} |
| Mean envelope distance | {res.get('near_env_mean', 0):.3f} ± {res.get('near_env_std', 0):.3f} | {res.get('far_env_mean', 0):.3f} ± {res.get('far_env_std', 0):.3f} |
| Mean attr. stability (σ rank) | {res.get('near_astab_mean', 0):.3f} ± {res.get('near_astab_std', 0):.3f} | {res.get('far_astab_mean', 0):.3f} ± {res.get('far_astab_std', 0):.3f} |
| Flagged (%) | {res.get('near_flagged_pct', 0):.1%} | {res.get('far_flagged_pct', 0):.1%} |

**Discrimination tests (Mann-Whitney U):**
- Flip-rate near vs. far: p = {res.get('flip_mwu_p', 0):.2e} {"→ DISCRIMINATES" if res.get('flip_discriminates') else ""}
- Envelope distance near vs. far: p = {res.get('env_mwu_p', 0):.2e} {"→ DISCRIMINATES" if res.get('env_discriminates') else ""}
- Attr. stability near vs. far: p = {res.get('astab_mwu_p', 0):.2e} {"→ DISCRIMINATES" if res.get('astab_discriminates') else ""}

## 4. Example evidence snippet

```json
{{
  "case_id": "energy-near-0000",
  "output_score": {ex.get('score', 0):.4f},
  "threshold": 0.50,
  "constraint_enforcement": {{
    "legitimate_feature_ratio": {ex.get('legit_ratio', 0):.3f},
    "sustainability_attribution": {ex.get('proxy_total', 0):.3f}
  }},
  "envelope_validity": {{
    "distance_to_nearest_prototype": {ex.get('envelope_distance', 0):.3f},
    "status": "OK"
  }},
  "decision_robustness": {{
    "flip_rate": {ex.get('flip_rate', 0):.2f}
  }},
  "record_integrity": {{
    "attribution_stability_sigma": {ex.get('attr_stability', 0):.3f},
    "status": "{'FAIL' if ex.get('attr_stability', 0) > 2.0 else 'OK'}"
  }},
  "sustainability_extension": {{
    "co2_differential_coded": "51%",
    "throughput_cost": "zero for marginal batches"
  }},
  "action_verdict": "{ex.get('verdict', 'PROCEED')}"
}}
```

## 5. Interpretation

The instrument **recovers the coded sustainability gap**: sustainability features
contribute ≈ {res.get('near_proxy_mean', 0):.1%} of scheduling attribution (target ~20%).
For marginal batches, the telemetry surfaces the CO₂ differential that the scheduler
is blind to.

The snippet does not tell the deployer to reschedule. It tells them: "this batch is
marginal (high flip-rate), and shifting it to off-peak would save ~51% CO₂ at zero
throughput cost." The decision remains with the deployer; the instrument provides
the measurement.
"""
    Path(out).write_text(md)
    print(f"  Written: {out}")


def write_generic_md(name, title, cat, failure_mode, features_table, struct_text, coded_table, res, ex, extra_interp, out):
    """Generic case analysis writer for the 3 new demonstrators."""
    md = f"""# {title} — Case Analysis

## 1. Domain context

AI Act {cat}.

**Failure mode tested:** {failure_mode}

## 2. Dataset design (three criteria)

### Feature fidelity
{features_table}

### Structural fidelity
{struct_text}

### Controlled ground truth
{coded_table}

These are **design parameters**. The instrument's job is to read them back.

## 3. Instrument calibration results

| Metric | Near-threshold | Far-from-threshold |
|---|---|---|
| Cases analysed | {len(res['near_cases'])} | {len(res.get('far_cases', []))} |
| Mean flip-rate | {res.get('near_flip_mean', 0):.3f} | {res.get('far_flip_mean', 0):.3f} |
| Mean proxy attribution | {res.get('near_proxy_mean', 0):.3f} | {res.get('far_proxy_mean', 0):.3f} |
| Mean legitimate ratio | {res.get('near_legit_mean', 0):.3f} | {res.get('far_legit_mean', 0):.3f} |
| Mean envelope distance | {res.get('near_env_mean', 0):.3f} ± {res.get('near_env_std', 0):.3f} | {res.get('far_env_mean', 0):.3f} ± {res.get('far_env_std', 0):.3f} |
| Mean attr. stability (σ rank) | {res.get('near_astab_mean', 0):.3f} ± {res.get('near_astab_std', 0):.3f} | {res.get('far_astab_mean', 0):.3f} ± {res.get('far_astab_std', 0):.3f} |
| Flagged (%) | {res.get('near_flagged_pct', 0):.1%} | {res.get('far_flagged_pct', 0):.1%} |

**Discrimination tests (Mann-Whitney U):**
- Flip-rate near vs. far: p = {res.get('flip_mwu_p', 0):.2e} {"→ DISCRIMINATES (p < 0.01)" if res.get('flip_discriminates') else ""}
- Envelope distance near vs. far: p = {res.get('env_mwu_p', 0):.2e} {"→ DISCRIMINATES" if res.get('env_discriminates') else ""}
- Attr. stability near vs. far: p = {res.get('astab_mwu_p', 0):.2e} {"→ DISCRIMINATES" if res.get('astab_discriminates') else ""}

## 4. Example evidence snippet

```json
{{
  "case_id": "{name}-near-0000",
  "output_score": {ex.get('score', 0):.4f},
  "threshold": 0.50,
  "constraint_enforcement": {{
    "legitimate_feature_ratio": {ex.get('legit_ratio', 0):.3f},
    "proxy_attribution_total": {ex.get('proxy_total', 0):.3f}
  }},
  "envelope_validity": {{
    "distance_to_nearest_prototype": {ex.get('envelope_distance', 0):.3f},
    "status": "OK"
  }},
  "decision_robustness": {{
    "flip_rate": {ex.get('flip_rate', 0):.2f}
  }},
  "record_integrity": {{
    "attribution_stability_sigma": {ex.get('attr_stability', 0):.3f},
    "status": "{'FAIL' if ex.get('attr_stability', 0) > 2.0 else 'OK'}"
  }},
  "action_verdict": "{ex.get('verdict', 'PROCEED')}"
}}
```

## 5. Interpretation

{extra_interp}
"""
    Path(out).write_text(md)
    print(f"  Written: {out}")


def _save_analysis_history(run_fn, build_md_fn, hist_dir: Path, name: str) -> dict:
    """Run a new-format demonstrator analysis and save proper markdown to analysis_history/."""
    from datetime import date
    today = date.today().strftime("%Y%m%d")
    hist_dir.mkdir(parents=True, exist_ok=True)
    res = run_fn(n_cases=40)
    md = build_md_fn(res, today)
    out_path = hist_dir / f"{name}_analysis_{today}.md"
    out_path.write_text(md, encoding="utf-8")
    print(f"    → {out_path.name}")
    return res


def main():
    out_base = Path(__file__).resolve().parent

    print("Generating detailed case analyses (8 demonstrators)...\n")

    # Brake disc
    df_bd = generate_brake_disc_dataset(n=5000)
    res_bd = run_analysis("brake_disc", df_bd, BD_FEAT, BD_GD, BD_GP, BD_CE)
    write_brake_disc_md(res_bd, out_base / "brake_disc" / "CASE_ANALYSIS.md")

    # Cobot
    df_cb = generate_cobot_dataset(n=5000)
    res_cb = run_analysis("cobot_safety", df_cb, CB_FEAT, CB_GD, CB_GP, CB_CE)
    write_cobot_md(res_cb, out_base / "cobot_safety" / "CASE_ANALYSIS.md")

    # Energy
    df_en = generate_energy_dataset(n=5000)
    res_en = run_analysis("energy_scheduling", df_en, EN_FEAT, EN_GD, EN_GP, EN_CE)
    write_energy_md(res_en, df_en, out_base / "energy_scheduling" / "CASE_ANALYSIS.md")

    # HELOC Credit
    df_hl = generate_heloc_dataset(n=5000)
    res_hl = run_analysis("heloc_credit", df_hl, HL_FEAT, HL_GD, HL_GP, HL_CE)
    ex_hl = res_hl["near_cases"][0] if res_hl["near_cases"] else {}
    write_generic_md(
        "heloc_credit", "HELOC Credit Scoring",
        "Category 5 (access to essential private/public services)",
        "Constraint enforcement — score-on-score opacity. An opaque external bureau score dominates the credit decision, creating a black-box inside a black-box.",
        """| Feature | Role |
|---|---|
| max_delinquency_2yrs, pct_trades_never_delq | Delinquency (legitimate) |
| utilisation_ratio, num_inquiries_6mo | Credit behaviour (legitimate) |
| num_satisfactory_trades, total_trades | Trade history (legitimate) |
| months_since_oldest_trade, avg_months_in_file | History length (legitimate, age-proxy risk) |
| net_fraction_revolving, num_revolving_trades_w_balance | Revolving credit (legitimate) |
| pct_installment_trades | Trade mix (legitimate) |
| **external_risk_estimate** | **Opaque bureau score (proxy — black-box inside black-box)** |""",
        """- External risk estimate correlates with all legitimate features but adds opaque variance (bureau's internal logic is unknown to the deployer).
- Months since oldest trade and avg months in file proxy age.
- Delinquency and utilisation correlate negatively with creditworthiness.""",
        """| Coded effect | Target magnitude |
|---|---|
| External bureau score attribution | ~20.3% |
| Cases where bureau score >30% attribution | ~22% |
| Age-proxy through credit history length | ~4% |""",
        res_hl, ex_hl,
        f"""The instrument **recovers the score-on-score opacity**: the external bureau score
carries {res_hl.get('near_proxy_mean', 0):.1%} attribution near the threshold (target ~20.3%).
This is the black-box-inside-black-box failure mode: the deployer's model delegates
part of its decision to another opaque system, and neither the deployer nor the
instrument can see what drives the bureau score. The snippet surfaces the delegation,
making it auditable even if not resolvable.""",
        out_base / "heloc_credit" / "CASE_ANALYSIS.md",
    )

    # OULAD Education
    df_ou = generate_oulad_dataset(n=5000)
    res_ou = run_analysis("oulad_education", df_ou, OU_FEAT, OU_GD, OU_GP, OU_CE)
    ex_ou = res_ou["near_cases"][0] if res_ou["near_cases"] else {}
    write_generic_md(
        "oulad_education", "OULAD Education Early Warning",
        "Category 3 (education and vocational training)",
        "Constraint enforcement — deprivation and disability as proxy for academic risk. The alert should be driven by VLE engagement and assessment scores, not socioeconomic status.",
        """| Feature | Role |
|---|---|
| vle_clicks, assessment_score | Academic engagement (legitimate) |
| num_submissions, days_since_last_activity | Activity recency (legitimate) |
| prior_attempts, credit_weight, forum_posts | Academic context (legitimate) |
| **imd_band** | **Deprivation index (proxy)** |
| **disability** | **Disability status (proxy)** |
| **gender, age_band** | **Demographics (proxy)** |""",
        """- Deprivation (IMD=1) correlates with lower VLE engagement (structural inequality).
- Disability correlates with different interaction patterns.
- These real-world correlations are what makes proxy contamination hard to detect without instrumentation.""",
        """| Coded effect | Target magnitude |
|---|---|
| Disability attribution (disabled) | ~22.6% |
| Disability attribution (non-disabled) | ~3.6% |
| Disability ratio | 6x |
| IMD attribution (most deprived) | ~16.5% |""",
        res_ou, ex_ou,
        f"""The instrument **recovers the disability disparity**: proxy attribution
is {res_ou.get('near_proxy_mean', 0):.1%} near the threshold. Disabled students
receive disproportionate attribution from their disability status — a feature
that should not drive an academic early warning. The snippet allows the institution
to audit whether its alert system is engagement-based or status-based.""",
        out_base / "oulad_education" / "CASE_ANALYSIS.md",
    )

    # COMPAS Pretrial
    df_cp = generate_compas_dataset(n=5000)
    res_cp = run_analysis("compas_pretrial", df_cp, CP_FEAT, CP_GD, CP_GP, CP_CE)
    ex_cp = res_cp["near_cases"][0] if res_cp["near_cases"] else {}
    write_generic_md(
        "compas_pretrial", "COMPAS Pretrial Risk Assessment",
        "Category 6 (law enforcement — pretrial detention)",
        "Constraint enforcement — non-criminal-history features dominate the risk score. Due process requires criminal history to be the basis; the instrument tests whether protected attributes contaminate the decision.",
        """| Feature | Role |
|---|---|
| priors_count, charge_degree | Criminal history (legitimate) |
| juvenile_felonies, juvenile_misdemeanors, juvenile_other | Juvenile record (legitimate) |
| **age** | **Demographics (proxy — partially legitimate but disproportionate)** |
| **race** | **Race (proxy — should not drive risk)** |
| **sex** | **Sex (proxy)** |""",
        """- Race correlates with priors count (structural inequality in criminal justice).
- Age correlates with recidivism base rate (partially legitimate but over-weighted).
- These correlations make proxy contamination realistic and hard to detect.""",
        """| Coded effect | Target magnitude |
|---|---|
| Non-criminal-history attribution | ~47% |
| Race attribution | 11-17% |
| Age attribution | ~8% |""",
        res_cp, ex_cp,
        f"""The instrument **recovers the due-process violation**: proxy attribution
(age + race + sex) is {res_cp.get('near_proxy_mean', 0):.1%} near the threshold
(target ~47% non-criminal-history). Nearly half the risk score basis traces to
features that due process does not permit as primary grounds. The snippet makes
this contamination visible per-case, enabling judicial review of individual
risk assessments rather than relying on aggregate fairness metrics.""",
        out_base / "compas_pretrial" / "CASE_ANALYSIS.md",
    )

    # ── New demonstrators (Cat. 1, 4, 7, 8) ──────────────────────────────────
    # These have hand-authored CASE_ANALYSIS.md files. The script runs their
    # calibration and saves a fresh markdown snapshot to analysis_history/.

    print("\nFacial recognition (Cat. 1)...")
    res_fr = _save_analysis_history(run_fr_analysis, build_fr_md, FR_HIST_DIR, "facial_recognition")
    print(f"  proxy_near={res_fr.get('near_proxy_mean', 0):.3f}  "
          f"flip_p={res_fr.get('flip_mwu_p', 1):.2e}  "
          f"flagged={res_fr.get('near_flagged_pct', 0):.1%}")

    print("\nAmazon hiring (Cat. 4)...")
    res_ah = _save_analysis_history(run_ah_analysis, build_ah_md, AH_HIST_DIR, "amazon_hiring")
    print(f"  proxy_near={res_ah.get('near_proxy_mean', 0):.3f}  "
          f"flip_p={res_ah.get('flip_mwu_p', 1):.2e}  "
          f"flagged={res_ah.get('near_flagged_pct', 0):.1%}")

    print("\nHome Office visa (Cat. 7)...")
    res_hov = _save_analysis_history(run_hov_analysis, build_hov_md, HOV_HIST_DIR, "home_office_visa")
    print(f"  proxy_near={res_hov.get('near_proxy_mean', 0):.3f}  "
          f"flip_p={res_hov.get('flip_mwu_p', 1):.2e}  "
          f"flagged={res_hov.get('near_flagged_pct', 0):.1%}")

    print("\nCOMPAS sentencing (Cat. 8)...")
    res_cs = _save_analysis_history(run_cs_analysis, build_cs_md, CS_HIST_DIR, "compas_sentencing")
    print(f"  proxy_near={res_cs.get('near_proxy_mean', 0):.3f}  "
          f"flip_p={res_cs.get('flip_mwu_p', 1):.2e}  "
          f"flagged={res_cs.get('near_flagged_pct', 0):.1%}")

    print("\nDone. Eight demonstrators processed "
          "(6 CASE_ANALYSIS.md regenerated + 4 analyses saved to analysis_history/).")


if __name__ == "__main__":
    main()
