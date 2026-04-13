# COMPAS Pretrial Risk Assessment — Case Analysis

## 1. Domain context

AI Act Category 6 (law enforcement — pretrial detention).

**Failure mode tested:** Constraint enforcement — non-criminal-history features dominate the risk score. Due process requires criminal history to be the basis; the instrument tests whether protected attributes contaminate the decision.

## 2. Dataset design (three criteria)

### Feature fidelity
| Feature | Role |
|---|---|
| priors_count, charge_degree | Criminal history (legitimate) |
| juvenile_felonies, juvenile_misdemeanors, juvenile_other | Juvenile record (legitimate) |
| **age** | **Demographics (proxy — partially legitimate but disproportionate)** |
| **race** | **Race (proxy — should not drive risk)** |
| **sex** | **Sex (proxy)** |

### Structural fidelity
- Race correlates with priors count (structural inequality in criminal justice).
- Age correlates with recidivism base rate (partially legitimate but over-weighted).
- These correlations make proxy contamination realistic and hard to detect.

### Controlled ground truth
| Coded effect | Target magnitude |
|---|---|
| Non-criminal-history attribution | ~47% |
| Race attribution | 11-17% |
| Age attribution | ~8% |

These are **design parameters**. The instrument's job is to read them back.

## 3. Instrument calibration results

| Metric | Near-threshold | Far-from-threshold |
|---|---|---|
| Cases analysed | 40 | 0 |
| Mean flip-rate | 0.048 | 0.000 |
| Mean proxy attribution | 0.300 | 0.000 |
| Mean legitimate ratio | 0.700 | 0.000 |
| Mean envelope distance | 0.414 ± 0.479 | 0.000 ± 0.000 |
| Mean attr. stability (σ rank) | 0.011 ± 0.028 | 0.000 ± 0.000 |
| Flagged (%) | 27.5% | 0.0% |

**Discrimination tests (Mann-Whitney U):**
- Flip-rate near vs. far: p = 0.00e+00 
- Envelope distance near vs. far: p = 0.00e+00 
- Attr. stability near vs. far: p = 0.00e+00 

## 4. Example evidence snippet

```json
{
  "case_id": "compas_pretrial-near-0000",
  "output_score": 0.4062,
  "threshold": 0.50,
  "constraint_enforcement": {
    "legitimate_feature_ratio": 0.814,
    "proxy_attribution_total": 0.186
  },
  "envelope_validity": {
    "distance_to_nearest_prototype": 0.124,
    "status": "OK"
  },
  "decision_robustness": {
    "flip_rate": 0.00
  },
  "record_integrity": {
    "attribution_stability_sigma": 0.000,
    "status": "OK"
  },
  "action_verdict": "PROCEED"
}
```

## 5. Interpretation

The instrument **recovers the due-process violation**: proxy attribution
(age + race + sex) is 30.0% near the threshold
(target ~47% non-criminal-history). Nearly half the risk score basis traces to
features that due process does not permit as primary grounds. The snippet makes
this contamination visible per-case, enabling judicial review of individual
risk assessments rather than relying on aggregate fairness metrics.
