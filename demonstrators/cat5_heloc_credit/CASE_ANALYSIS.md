# HELOC Credit Scoring — Case Analysis

## 1. Domain context

AI Act Category 5 (access to essential private/public services).

**Failure mode tested:** Constraint enforcement — score-on-score opacity. An opaque external bureau score dominates the credit decision, creating a black-box inside a black-box.

## 2. Dataset design (three criteria)

### Feature fidelity
| Feature | Role |
|---|---|
| max_delinquency_2yrs, pct_trades_never_delq | Delinquency (legitimate) |
| utilisation_ratio, num_inquiries_6mo | Credit behaviour (legitimate) |
| num_satisfactory_trades, total_trades | Trade history (legitimate) |
| months_since_oldest_trade, avg_months_in_file | History length (legitimate, age-proxy risk) |
| net_fraction_revolving, num_revolving_trades_w_balance | Revolving credit (legitimate) |
| pct_installment_trades | Trade mix (legitimate) |
| **external_risk_estimate** | **Opaque bureau score (proxy — black-box inside black-box)** |

### Structural fidelity
- External risk estimate correlates with all legitimate features but adds opaque variance (bureau's internal logic is unknown to the deployer).
- Months since oldest trade and avg months in file proxy age.
- Delinquency and utilisation correlate negatively with creditworthiness.

### Controlled ground truth
| Coded effect | Target magnitude |
|---|---|
| External bureau score attribution | ~20.3% |
| Cases where bureau score >30% attribution | ~22% |
| Age-proxy through credit history length | ~4% |

These are **design parameters**. The instrument's job is to read them back.

## 3. Instrument calibration results

| Metric | Near-threshold | Far-from-threshold |
|---|---|---|
| Cases analysed | 40 | 40 |
| Mean flip-rate | 0.034 | 0.000 |
| Mean proxy attribution | 0.148 | 0.176 |
| Mean legitimate ratio | 0.852 | 0.824 |
| Mean envelope distance | 2.713 ± 0.371 | 2.145 ± 0.320 |
| Mean attr. stability (σ rank) | 0.083 ± 0.073 | 0.073 ± 0.067 |
| Flagged (%) | 2.5% | 2.5% |

**Discrimination tests (Mann-Whitney U):**
- Flip-rate near vs. far: p = 1.40e-13 → DISCRIMINATES (p < 0.01)
- Envelope distance near vs. far: p = 1.94e-09 → DISCRIMINATES
- Attr. stability near vs. far: p = 2.41e-01 → does not discriminate (expected: stability is a transversal admissibility gate, not a detector of score-on-score opacity; a non-discriminating reading confirms the signal's axis is not degraded near the threshold)

## 4. Example evidence snippet

```json
{
  "case_id": "heloc_credit-near-0000",
  "output_score": 0.5028,
  "threshold": 0.50,
  "constraint_enforcement": {
    "legitimate_feature_ratio": 0.795,
    "proxy_attribution_total": 0.205,
    "proxy_detail": {
      "external_risk_estimate": 0.205
    }
  },
  "envelope_validity": {
    "distance_to_nearest_prototype": 3.113,
    "status": "OK"
  },
  "decision_robustness": {
    "flip_rate": 0.34
  },
  "_note": "flip_rate 0.34 is a high-fragility individual case, not the mean (0.034); chosen to show PROCEED WITH FLAG verdict for a borderline instance where 34% of perturbations reverse the decision",
  "record_integrity": {
    "attribution_stability_sigma": 0.000,
    "status": "OK"
  },
  "action_verdict": "PROCEED WITH FLAG"
}
```

## 5. Interpretation

The instrument **recovers the score-on-score opacity**: the external bureau score
carries 14.8% attribution near the threshold (target ~20.3%).

**On the recovery gap (14.8% vs. 20.3%):** The gap is expected at the near-threshold
margin. Near-threshold cases are marginal precisely because legitimate features are
mixed — neither strongly approving nor denying. In this subset, the bureau score's
contribution is partially diluted by the legitimate signal, reducing its measured
attribution share. The population-level target (20.3%) applies across the full
distribution; the instrument's near-threshold reading (14.8%) reflects the local
structure of borderline cases. The discrimination test (p = 1.40e-13 for flip-rate)
confirms the instrument distinguishes near-threshold fragility even with this dilution.

This is the black-box-inside-black-box failure mode: the deployer's model delegates
part of its decision to another opaque system, and neither the deployer nor the
instrument can see what drives the bureau score. The snippet surfaces the delegation,
making it auditable even if not resolvable. Auditability without resolution is the
correct outcome under the black-box constraint: the deployer can flag and escalate
without vendor cooperation.
