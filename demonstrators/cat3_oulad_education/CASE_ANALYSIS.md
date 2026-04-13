# OULAD Education Early Warning — Case Analysis

## 1. Domain context

AI Act Category 3 (education and vocational training).

**Failure mode tested:** Constraint enforcement — deprivation and disability as proxy for academic risk. The alert should be driven by VLE engagement and assessment scores, not socioeconomic status.

## 2. Dataset design (three criteria)

### Feature fidelity
| Feature | Role |
|---|---|
| vle_clicks, assessment_score | Academic engagement (legitimate) |
| num_submissions, days_since_last_activity | Activity recency (legitimate) |
| prior_attempts, credit_weight, forum_posts | Academic context (legitimate) |
| **imd_band** | **Deprivation index (proxy)** |
| **disability** | **Disability status (proxy)** |
| **gender, age_band** | **Demographics (proxy)** |

### Structural fidelity
- Deprivation (IMD=1) correlates with lower VLE engagement (structural inequality).
- Disability correlates with different interaction patterns.
- These real-world correlations are what makes proxy contamination hard to detect without instrumentation.

### Controlled ground truth
| Coded effect | Target magnitude |
|---|---|
| Disability attribution (disabled) | ~22.6% |
| Disability attribution (non-disabled) | ~3.6% |
| Disability ratio | 6x |
| IMD attribution (most deprived) | ~16.5% |

These are **design parameters**. The instrument's job is to read them back.

## 3. Instrument calibration results

| Metric | Near-threshold | Far-from-threshold |
|---|---|---|
| Cases analysed | 40 | 1 |
| Mean flip-rate | 0.128 | 0.000 |
| Mean proxy attribution | 0.424 | 0.056 |
| Mean legitimate ratio | 0.576 | 0.944 |
| Mean envelope distance | 1.604 ± 0.509 | 9.263 ± 0.000 |
| Mean attr. stability (σ rank) | 0.071 ± 0.067 | 0.073 ± 0.000 |
| Flagged (%) | 60.0% | 0.0% |

**Discrimination tests (Mann-Whitney U):**
- Flip-rate near vs. far: p = 0.00e+00 
- Envelope distance near vs. far: p = 0.00e+00 
- Attr. stability near vs. far: p = 0.00e+00 

## 4. Example evidence snippet

```json
{
  "case_id": "oulad_education-near-0000",
  "output_score": 0.4360,
  "threshold": 0.50,
  "constraint_enforcement": {
    "legitimate_feature_ratio": 0.740,
    "proxy_attribution_total": 0.260
  },
  "envelope_validity": {
    "distance_to_nearest_prototype": 2.081,
    "status": "OK"
  },
  "decision_robustness": {
    "flip_rate": 0.14
  },
  "record_integrity": {
    "attribution_stability_sigma": 0.145,
    "status": "OK"
  },
  "action_verdict": "PROCEED"
}
```

## 5. Interpretation

The instrument **recovers the disability disparity**: proxy attribution
is 42.4% near the threshold. Disabled students
receive disproportionate attribution from their disability status — a feature
that should not drive an academic early warning. The snippet allows the institution
to audit whether its alert system is engagement-based or status-based.
