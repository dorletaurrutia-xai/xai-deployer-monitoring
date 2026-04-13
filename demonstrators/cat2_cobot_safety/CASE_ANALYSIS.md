# Cat. 2 — Safety Component: Cobot Clearance + Brake Disc QC

AI Act Category 2 covers any AI system that is itself a safety component, or whose
output is used as a safety component, in critical infrastructure and industrial
machinery. This demonstrator covers two distinct failure modes within Category 2,
both instrumented with the same five-signal protocol.

---

## Caso A: Cobot Safety Clearance

### 1. Domain context

AI Act Category 2 (safety component) + Category 4 (employment/worker management).
A cobot safety system clears workers into shared human-robot zones. The model is a
black-box; the deployer (hospital/factory) has only `predict_proba` access.

**Failure mode:** `identity_contamination` — worker identity (contract type,
experience, shift) contaminates the safety clearance decision.

**Dual-regulation property:** The same evidence snippet answers BOTH
- **Cat. 2:** Is the safety clearance hazard-based?
- **Cat. 4:** Is the worker treatment non-discriminatory?

### 2. Dataset design (three criteria)

#### Feature fidelity
| Feature | Type | Role |
|---|---|---|
| distance_m, velocity_ms, payload_kg | float | Hazard indicator (legitimate) |
| exposure_time_s, zone_type, robot_mode | float/int | Hazard context (legitimate) |
| force_limit_n, collision_history | float/int | Safety record (legitimate) |
| worker_age, contract_type | float/int | **Worker identity** (proxy) |
| experience_years, shift | float/int | **Worker identity** (proxy) |
| training_hours, worker_role | float/int | **Worker identity** (proxy) |

#### Structural fidelity
- Temp workers (contract_type=1) are coded younger and less experienced.
- Hazard features correlate realistically (closer distance + higher velocity = more dangerous).
- Training hours lower for temp workers (structural inequality).

#### Controlled ground truth
| Coded effect | Target magnitude |
|---|---|
| Worker identity attribution (overall) | ~25.2% |
| Worker identity attribution (temp workers) | ~31.7% |
| Dual-regulation: one snippet serves Cat. 2 + Cat. 4 | By architecture |

These are **design parameters**. The instrument's job is to read them back.

### 3. Instrument calibration results

| Metric | Near-threshold | Far-from-threshold |
|---|---|---|
| Cases analysed | 40 | 40 |
| Mean flip-rate | 0.168 ± 0.168 | 0.000 ± 0.000 |
| Mean proxy attribution | 0.263 ± 0.065 | 0.220 ± 0.070 |
| Mean legitimate ratio | 0.737 | 0.780 |
| Mean envelope distance | 3.378 ± 0.533 | 3.881 ± 0.606 |
| Mean attr. stability (σ rank) | 0.091 ± 0.071 | 0.074 ± 0.055 |
| Flagged (%) | 22.5% | 0.0% |

**Discrimination tests (Mann-Whitney U):**
- Flip-rate near vs. far: p = 1.40e-13 → DISCRIMINATES (p < 0.01)
- Envelope distance near vs. far: p = 9.99e-01 → does not discriminate (expected: identity contamination does not shift the input manifold — near-threshold cases are the most typical mixed-signal cases, with lower k-NN distance than extreme-signal far cases; this confirms the envelope axis is not degraded)
- Attr. stability near vs. far: p = 2.04e-01 → does not discriminate (expected: stability is a transversal admissibility gate, not a detector of identity contamination)

### 4. Evidence snippet (Caso A — 7 signals)

```json
{
  "case_id": "cobot_safety-near-0001",
  "demonstrator": "cobot_safety",
  "failure_mode": "identity_contamination",
  "annex_iii_category": "cat_2",
  "output_score": 0.4729,
  "threshold": 0.5,
  "margin_to_threshold": 0.0271,
  "constraint_enforcement": {
    "legitimate_feature_ratio": 0.743,
    "proxy_attribution_total": 0.257,
    "proxy_detail": {
      "worker_age": 0.0083,
      "contract_type": 0.0951,
      "experience_years": 0.0251,
      "shift": 0.0021,
      "training_hours": 0.0415,
      "worker_role": 0.0848
    },
    "top_features": ["exposure_time_s", "payload_kg", "distance_m", "zone_type"],
    "status": "OK"
  },
  "envelope_validity": {
    "distance_to_nearest_prototype": 2.5424,
    "status": "OK"
  },
  "decision_robustness": {
    "flip_rate": 0.2,
    "status": "OK"
  },
  "record_integrity": {
    "attribution_stability_sigma": 0.1472,
    "status": "OK"
  },
  "sustainability_impact": {
    "dimension": "worker_wellbeing",
    "at_risk_group": "temporary_workers",
    "indicator": "identity_driven_clearance_rate",
    "value": 0.257,
    "status": "FLAG"
  },
  "human_ai_teaming": {
    "dimension": "supervisory_oversight",
    "escalation_level": "recommended_review",
    "trigger": "near_threshold_with_identity_attribution",
    "human_override_available": true,
    "status": "OK"
  },
  "action_verdict": "PROCEED"
}
```

`sustainability_impact` is FLAG (proxy > 0.25): 25.7% of this clearance decision
traces to worker identity. `human_ai_teaming` escalates to `recommended_review`:
a near-threshold decision with identity attribution warrants human confirmation
before acting on the clearance denial.

### 5. Interpretation

The instrument **recovers the coded identity contamination**: proxy attribution
≈ 26.3% near threshold (target ~25.2%). Flip-rate discriminates near vs. far
(p = 1.40e-13): near-threshold clearances are fragile.

The same snippet that flags identity-driven clearance (Cat. 4 discrimination
concern) simultaneously answers whether the clearance was hazard-based (Cat. 2
safety concern). One architecture, one snippet, two regulatory obligations.

---

## Caso B: Brake Disc QC

### 1. Domain context

AI Act Category 2 (safety component). A quality inspection system auto-passes or
auto-rejects brake discs on a production line. The model is a black-box cloud API;
the deployer (manufacturer) has only `predict_proba` access.

**Failure mode:** `brittle_autopass` — environmental confounders (oil residue,
lighting, camera angle, temperature) produce fragile near-threshold auto-passes
on safety-critical parts.

### 2. Dataset design (three criteria)

#### Feature fidelity
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

`shift` (day/evening/night) is a construction variable used to correlate
confounders with each other. It is excluded from FEATURE_NAMES and does not
enter the model.

#### Structural fidelity
- Crack length correlates with crack depth (length ≈ 2–8× depth).
- Night shift → higher oil residue, lower lighting, larger camera angle variance.
- Confounders correlate with each other through the shift variable.

#### Controlled ground truth
| Coded effect | Target magnitude |
|---|---|
| Confounder attribution weight | ~34% |
| Marginal auto-pass flip-rate | ~63% |

These are **design parameters**. The instrument's job is to read them back.

### 3. Instrument calibration results

| Metric | Near-threshold | Far-from-threshold |
|---|---|---|
| Cases analysed | 40 | 40 |
| Mean flip-rate | 0.193 ± 0.117 | 0.000 ± 0.000 |
| Mean proxy attribution | 0.330 ± 0.118 | 0.199 ± 0.184 |
| Mean legitimate ratio | 0.670 | 0.801 |
| Mean envelope distance | 1.548 ± 0.897 | 1.335 ± 1.023 |
| Mean attr. stability (σ rank) | 0.121 ± 0.084 | 0.055 ± 0.053 |
| Flagged (%) | 37.5% | 12.5% |

Envelope computed in confounder subspace (oil_residue, lighting_lux, camera_angle_deg,
temperature_c) against day-shift training prototypes. See note below.

**Discrimination tests (Mann-Whitney U):**
- Flip-rate near vs. far: p = 6.91e-15 → DISCRIMINATES (p < 0.01)
- Envelope distance near vs. far: p = 1.02e-01 → does not discriminate (expected: brittle_autopass is a within-manifold failure — confounder contamination biases the model's attribution without moving cases outside the operational input distribution; the envelope sensor correctly stays silent)
- Attr. stability near vs. far: p = 1.28e-04 → DISCRIMINATES (p < 0.01)

The 12.5% flagged rate in far cases reflects the high variance of confounder
attribution (σ = 0.184): some far-from-threshold cases still carry heavy confounder
weight, which the instrument correctly surfaces.

**Envelope sensor note:** The envelope is computed in the 4-dimensional confounder
subspace against day-shift training cases (the operational baseline defined in
`generate_dataset_brake_disc.py: KNN_SUBSPACE_INDICES, KNN_REFERENCE_COL`). This
design ensures that night-shift inputs with elevated confounders appear further from
the day-shift reference. Non-discrimination (p = 1.02e-01) reflects that both near-
and far-threshold cases span the shift distribution proportionally — the failure
mode does not selectively concentrate night-shift inputs near the decision boundary.
This is structurally analogous to Caso A: both failure modes operate within the
input manifold, so the envelope axis is not the primary detection surface.

### 4. Evidence snippet (Caso B — 7 signals)

```json
{
  "case_id": "brake_disc_qc-near-0000",
  "demonstrator": "brake_disc_qc",
  "failure_mode": "brittle_autopass",
  "annex_iii_category": "cat_2",
  "output_score": 0.3711,
  "threshold": 0.5,
  "margin_to_threshold": 0.1289,
  "constraint_enforcement": {
    "legitimate_feature_ratio": 0.8108,
    "proxy_attribution_total": 0.1892,
    "proxy_detail": {
      "oil_residue": 0.0734,
      "lighting_lux": 0.0828,
      "camera_angle_deg": 0.0075,
      "temperature_c": 0.0256
    },
    "top_features": ["crack_depth_mm", "crack_length_mm", "runout_mm", "lighting_lux"],
    "status": "OK"
  },
  "envelope_validity": {
    "distance_to_nearest_prototype": 1.7384,
    "status": "OK"
  },
  "decision_robustness": {
    "flip_rate": 0.06,
    "status": "OK"
  },
  "record_integrity": {
    "attribution_stability_sigma": 0.0,
    "status": "OK"
  },
  "sustainability_impact": {
    "dimension": "product_safety",
    "indicator": "confounder_driven_autopass_rate",
    "value": 0.1892,
    "note": "confounder-driven passes on safety parts create undetected recall risk",
    "status": "OK"
  },
  "human_ai_teaming": {
    "dimension": "quality_gate_oversight",
    "escalation_level": "nominal",
    "trigger": "brittle_autopass_near_threshold",
    "human_override_available": true,
    "status": "OK"
  },
  "action_verdict": "PROCEED"
}
```

### 5. Interpretation

The instrument **recovers the coded confounder structure**: proxy attribution
near-threshold ≈ 33% (target ~34%). Flip-rate discriminates (p = 6.91e-15):
near-threshold auto-passes are significantly more fragile than far-threshold ones.
Attribution stability also discriminates (p = 1.28e-04): near-threshold cases
show less stable rank orders, consistent with the confounder signal introducing
ambiguity at the decision boundary.

`sustainability_impact` is computed per-case from confounder attribution: a
confounder-driven pass on a brake disc is an undetected safety defect with
downstream recall risk. `human_ai_teaming` escalates when flip-rate > 0.3
(`mandatory_review`) or flip-rate > 0.1 (`recommended_review`).

---

## Síntesis Cat. 2

| | Cobot Safety (Caso A) | Brake Disc QC (Caso B) |
|---|---|---|
| Failure mode | `identity_contamination` | `brittle_autopass` |
| Primary signal | `constraint_enforcement` | `decision_robustness` |
| Secondary discriminating | flip-rate (p=1.40e-13) | flip-rate + attr.stability (p=1.28e-04) |
| Proxy type | worker identity (6 features) | environmental confounders (4 features) |
| Envelope discriminates | No (expected — identity does not shift manifold) | No (expected — confounder contamination is within-manifold) |
| Dual regulation | Cat. 2 + Cat. 4 | Cat. 2 only |
| Industry 5.0 sustainability | worker wellbeing / temp worker risk | product safety / recall risk |

One protocol, one snippet structure, two failure modes within the same Annex III
category. The primary signal varies by domain: attribution decomposition detects
who is driving the decision; flip-rate detects whether that decision is stable.
Non-compensatory evaluation — each signal is assessed on its own axis. A clean
attribution reading does not offset a fragile decision, and vice versa.

**Envelope non-discrimination is a diagnostic feature, not a deficiency.** The
envelope sensor activates only for genuine input-space distributional shift (e.g.,
sensor drift, adversarial inputs, domain transfer). Both Cat. 2 failure modes here
— identity contamination and confounder contamination — are attribution-space
failures that occur within the operational input manifold. The envelope's silence
confirms the inputs are in-distribution; the failure is internal to the model's
reasoning. This specificity is what makes the five-signal protocol
*non-compensatory*: each sensor is diagnostic for a different failure axis.
