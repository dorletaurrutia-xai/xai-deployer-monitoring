# Cat. 1 — Biometric Identification: Facial Recognition Demographic Proxy Alert

AI Act Category 1 covers AI systems intended to be used for the real-time and
post-remote biometric identification of natural persons. This demonstrator
instruments the failure mode of demographic proxy alert: a live facial
recognition (LFR) system activates alerts at disproportionate rates for
non-white subjects because it has learned to use the historical false-positive
rate for a demographic profile as a decision input — effectively encoding
documented system bias as a risk signal.

Inspired by the Metropolitan Police UK LFR deployments (2016–present) and
the Big Brother Watch (2018) audit, which documented a 98% false positive rate.
NIST FRVT (2019) confirmed that commercial FR systems exhibit significantly
higher error rates for darker-skinned faces; Buolamwini & Gebru (2018) showed
error rates up to 34.7% for darker-skinned women vs. 0.8% for lighter-skinned
men. The snippet this instrument produces is the per-case attribution that
documents *which features drove this specific alert*.

---

## 1. Domain context

AI Act Category 1 (remote biometric identification and categorisation).
An alert-activation model classifies matching decisions: given a candidate
match and a confidence_score from the FR system, should the alert be activated
(i.e., should the operator escalate to detain/question the subject)?
The deployer (police operator / automated LFR system) has only `predict_proba`
access to the alert model.

**Modelling decision:** the system does NOT classify images directly — it
classifies *matching decisions*. Each case is one candidate match event;
the FR algorithm's output (`confidence_score`) is a feature, not the model.

**Failure mode:** `demographic_proxy_alert` — three demographic-correlated
proxy features contaminate the alert threshold:
- `demographic_parity_ratio`: the historical false-positive rate for this
  demographic profile in prior deployments. A biased system creates biased
  history; that history is then fed back as a risk signal.
- `skin_tone_estimate`: normalised skin tone from image analysis. A correlate
  of demographic_group that adds a within-group skin-tone signal.
- `prior_watch_list_flag`: binary indicator of prior watch-list association.
  Structurally higher for non-white subjects through policing disparities
  and over-representation in historical intelligence databases.

**Big Brother Watch property:** the 2018 audit reported a 98% false positive
rate in aggregate but could not identify *per alert* what drove the decision.
The snippet this instrument produces is that per-alert breakdown.

## 2. Dataset design (three criteria)

### Feature fidelity

| Feature | Type | Role |
|---|---|---|
| confidence_score | float [0–1] | FR system matching confidence (legitimate) |
| image_quality_score | float [0–1] | Captured image quality (legitimate) |
| match_database_size | float [0–1] | Reference database size / FP pressure (legitimate) |
| time_since_enrollment | float [0–10] | Template age in years (legitimate) |
| lighting_conditions | float [0–1] | Illumination quality (legitimate) |
| **demographic_parity_ratio** | float [0–1] | **Proxy — historical FP rate for demographic profile** |
| **skin_tone_estimate** | float [0–1] | **Proxy — skin tone from image analysis** |
| **prior_watch_list_flag** | binary {0,1} | **Proxy — prior watch-list association** |

`match_genuineness` and `demographic_group` are latent construction variables
— not in FEATURE_NAMES and not used by the model.

### Structural fidelity

- A shared `match_genuineness` latent factor drives all five legitimate features
  with 70% weight (30% idiosyncratic noise), creating correlated match-quality
  profiles: strong genuine matches score high on confidence, image quality, and
  lighting simultaneously.
- `match_database_size` and `time_since_enrollment` contribute *negatively* to
  the alert signal (larger database = more false-positive pressure; older
  template = appearance drift). Both are partly correlated with
  `(1 - match_genuineness)`, so the model learns meaningful negative weights.
  This mixed-sign structure creates stronger near-threshold proxy amplification
  than all-positive legitimate designs: positive and negative legitimate
  components partially cancel near the decision boundary, increasing the
  relative proxy attribution share.
- `demographic_group` drives proxy features through structural inequalities:
  non-white subjects have higher `demographic_parity_ratio` (mean ≈ 0.55 vs.
  0.20 for white), higher `skin_tone_estimate` (mean ≈ 0.63 vs. 0.29), and
  higher `prior_watch_list_flag` rate (30% vs. 8%).
- Alert rate gap: white 44.5%, non-white 50.2% (~6pp gap driven by
  demographic proxy signal).

### Controlled ground truth

| Coded effect | Target magnitude |
|---|---|
| Proxy attribution coded weight | ~9.8% of combined signal |
| Proxy attribution near-threshold (SHAP) | ~20–25% |
| DPR attribution near-threshold | ~12–15% |
| Alert rate gap (non-white − white) | ~6pp |

These are **design parameters**. The instrument's job is to read them back.

## 3. Instrument calibration results

| Metric | Near-threshold | Far-from-threshold |
|---|---|---|
| Cases analysed | 40 | 40 |
| Mean flip-rate | 0.148 ± 0.158 | 0.000 ± 0.000 |
| Mean proxy attribution | 0.244 ± 0.113 | 0.066 ± 0.028 |
| Mean DPR attribution | 0.138 ± 0.091 | 0.037 ± 0.023 |
| Mean legitimate ratio | 0.756 | 0.934 |
| Mean envelope distance | 1.578 ± 0.318 | 2.087 ± 0.444 |
| Mean attr. stability (σ rank) | 0.093 ± 0.102 | 0.011 ± 0.032 |
| Flagged (%) | 25.0% | 0.0% |

**Discrimination tests (Mann-Whitney U):**
- Flip-rate near vs. far: p = 2.00e-13 → DISCRIMINATES (p < 0.01)
- Envelope distance near vs. far: p = 1.00e+00 → does not discriminate (expected: demographic proxy bias is a within-manifold failure — subjects of all demographics appear in the same 8-dimensional input space; the failure is internal to the model's learned weighting of the proxy features)
- Attr. stability near vs. far: p = 9.45e-11 → DISCRIMINATES (p < 0.01)

## 4. Evidence snippet (7 signals)

```json
{
  "case_id": "facial_recognition-near-0009",
  "demonstrator": "facial_recognition",
  "failure_mode": "demographic_proxy_alert",
  "annex_iii_category": "cat_1",
  "output_score": 0.4533,
  "threshold": 0.5,
  "margin_to_threshold": 0.0467,
  "constraint_enforcement": {
    "legitimate_feature_ratio": 0.636,
    "proxy_attribution_total": 0.364,
    "proxy_detail": {
      "demographic_parity_ratio": 0.2187,
      "skin_tone_estimate": 0.0244,
      "prior_watch_list_flag": 0.1209
    },
    "top_features": ["demographic_parity_ratio", "time_since_enrollment",
                     "confidence_score", "image_quality_score"],
    "status": "OK"
  },
  "envelope_validity": {
    "distance_to_nearest_prototype": 1.6188,
    "status": "OK"
  },
  "decision_robustness": {
    "flip_rate": 0.1,
    "status": "OK"
  },
  "record_integrity": {
    "attribution_stability_sigma": 0.0,
    "status": "OK"
  },
  "sustainability_impact": {
    "dimension": "civil_liberties",
    "at_risk_group": "non_white_subjects",
    "indicator": "demographic_proxy_alert_rate",
    "value": 0.2187,
    "status": "FLAG"
  },
  "human_ai_teaming": {
    "dimension": "officer_oversight",
    "escalation_level": "recommended_review",
    "trigger": "near_threshold_demographic_proxy_alert",
    "human_override_available": true,
    "note": "Big Brother Watch 2018: Metropolitan Police LFR — 98% false positive rate",
    "status": "OK"
  },
  "action_verdict": "PROCEED"
}
```

`constraint_enforcement` is OK (legitimate_ratio 63.6% ≥ 60%) but
`sustainability_impact` is FLAG: 21.9% of this alert decision traces to
`demographic_parity_ratio` — the historical false-positive rate for this
demographic profile. Crucially, `demographic_parity_ratio` appears as the
**top feature** in this case, ranked above `confidence_score`, the FR
system's primary technological output. The procedural verdict is PROCEED
because no single core signal crosses its threshold; the sustainability FLAG
is an independent axis that surfaces what the core check does not.
`human_ai_teaming` escalates to `recommended_review`: a near-threshold alert
with demographic proxy attribution above the civil liberties threshold warrants
officer verification before acting on the alert.

## 5. Interpretation

The instrument **recovers the coded demographic bias**: proxy attribution
near-threshold ≈ 24.4% (coded ~9.8%; amplification factor ~2.5×). DPR
attribution near-threshold ≈ 13.8% (target 12–15%).

**Near-threshold amplification is stronger than in all-positive designs.**
In amazon_hiring (Cat. 4), the amplification factor was ~1.4× (35.5% measured
vs. 25% coded). In facial_recognition, the factor is ~2.5× because the
legitimate signal contains mixed-sign components: `confidence_score` and
`lighting_conditions` contribute positively to the alert, while
`match_database_size` and `time_since_enrollment` contribute negatively.
Near the decision boundary, these positive and negative legitimate components
partially cancel each other, leaving more relative attribution space for the
proxy features. The instrument captures this structural amplification property
without requiring access to the model's weights.

**`demographic_parity_ratio` appears as the top feature.** In the snippet,
`demographic_parity_ratio` (a historical bias metric) ranks above
`confidence_score` (the FR algorithm's primary output). This is not a
coincidence: the biased system has trained on historically biased alert
outcomes, encoding the demographic FP rate as a predictive signal. The
deployer's alert model has learned that high historical FP rates for a
demographic profile correlate with positive labels in the training data
— because the training data itself reflected the biased prior system.

**The watch-list flag effect.** `prior_watch_list_flag` contributes 12.09%
in the snippet despite having only a 0.5% coded weight (0.005 × flag). This
occurs because the flag is binary: when flag = 1, the SHAP value equals
approximately the model weight × (1 − background mean). For a rare binary
feature with a low background mean (~19%), a single flag = 1 creates a
concentrated attribution spike. The instrument surfaces this even when the
population-level coded weight is small.

**Flip-rate discriminates** (p = 2.00e-13): near-threshold alerts are
significantly more fragile than far-from-threshold ones. A 10% flip-rate in
the snippet means that 5 of 50 bounded perturbations of this match event's
features would reverse the alert decision — fragility invisible to the
operator who receives only the point score.

**Attr. stability discriminates** (p = 9.45e-11): near-threshold cases show
less stable feature rank orders under repeated attribution, consistent with
demographic proxy introducing ambiguity at the decision boundary.

**Envelope does not discriminate** (expected): demographic proxy bias is a
within-manifold failure. Subjects of all demographics appear in the same
8-dimensional input space. The failure is not that non-white subjects are
out-of-distribution, but that the model has learned to weight demographic
history as a risk signal for in-distribution match events. The envelope's
silence confirms the inputs are operationally valid; the contamination is
internal to the model's learned representation.

**The non-compensatory gap.** The snippet's `action_verdict` is PROCEED:
no individual core signal crosses its flag threshold. But `sustainability_impact`
is independently FLAG. This is a design feature of the 7-signal non-compensatory
protocol — civil liberties impact and procedural correctness are evaluated on
separate axes. The Big Brother Watch audit found that 98% of Metropolitan Police
LFR alerts were false positives; the snippet documents the mechanism one alert
at a time: 21.9% of this specific alert decision traces to the demographic
history of the subject, not to the strength of the face match itself.
