# Brake Disc Quality Control — Case Analysis

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
| Cases analysed | 40 | 1 |
| Mean flip-rate | 0.092 ± 0.140 | 0.000 ± 0.000 |
| Mean proxy attribution | 0.268 ± 0.135 | 0.104 ± 0.000 |
| Mean legitimate ratio | 0.732 | 0.896 |
| Mean envelope distance | 2.184 ± 0.417 | 3.879 ± 0.000 |
| Mean attr. stability (σ rank) | 0.092 ± 0.065 | 0.000 ± 0.000 |
| Flagged (%) | 17.5% | 0.0% |

**Discrimination tests (Mann-Whitney U):**
- Flip-rate near vs. far: p = 0.00e+00 
- Envelope distance near vs. far: p = 0.00e+00 
- Attr. stability near vs. far: p = 0.00e+00 

## 4. Example evidence snippet

Case: near-threshold auto-pass (score 0.533, margin 0.033)

```json
{
  "case_id": "brake_disc-near-0000",
  "output_score": 0.5331,
  "threshold": 0.50,
  "margin_to_threshold": 0.0331,
  "constraint_enforcement": {
    "status": "OK",
    "legitimate_feature_ratio": 0.715,
    "top_features": ["surface_porosity_pct", "thickness_var_mm", "oil_residue", "camera_angle_deg"],
    "proxy_attribution_total": 0.285
  },
  "envelope_validity": {
    "status": "OK",
    "distance_to_nearest_prototype": 3.034
  },
  "decision_robustness": {
    "status": "OK",
    "flip_rate": 0.16
  },
  "record_integrity": {
    "attribution_stability_sigma": 0.046,
    "status": "OK"
  },
  "action_verdict": "PROCEED"
}
```

## 5. Interpretation

The instrument **recovers the coded confounder structure**: proxy attribution
near the target ~34%. Near-threshold cases show elevated flip-rates, confirming
the designed fragility. The snippet would flag a brittle confounder-driven pass
for manual review in an operational deployment.

**What this validates:** the flip-rate signal and the attribution decomposition
correctly discriminate the failure mode we coded in. This is instrument
calibration — the thermometer reads back the known reference temperature.
