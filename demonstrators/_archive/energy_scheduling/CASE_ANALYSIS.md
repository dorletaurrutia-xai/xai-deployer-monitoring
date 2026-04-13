# Energy Scheduling — Case Analysis

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
- Mean CO₂ saving for peak-scheduled batches: 57.8%

## 3. Instrument calibration results

| Metric | Near-threshold | Far-from-threshold |
|---|---|---|
| Cases analysed | 40 | 5 |
| Mean flip-rate | 0.086 | 0.000 |
| Mean sustainability attr. | 0.176 | 0.092 |
| Mean legitimate ratio | 0.824 | 0.908 |
| Mean envelope distance | 1.987 ± 0.410 | 2.328 ± 0.099 |
| Mean attr. stability (σ rank) | 0.091 ± 0.070 | 0.084 ± 0.058 |
| Flagged (%) | 12.5% | 0.0% |

**Discrimination tests (Mann-Whitney U):**
- Flip-rate near vs. far: p = 3.55e-02 
- Envelope distance near vs. far: p = 9.94e-01 
- Attr. stability near vs. far: p = 5.14e-01 

## 4. Example evidence snippet

```json
{
  "case_id": "energy-near-0000",
  "output_score": 0.4593,
  "threshold": 0.50,
  "constraint_enforcement": {
    "legitimate_feature_ratio": 0.922,
    "sustainability_attribution": 0.078
  },
  "envelope_validity": {
    "distance_to_nearest_prototype": 3.589,
    "status": "OK"
  },
  "decision_robustness": {
    "flip_rate": 0.14
  },
  "record_integrity": {
    "attribution_stability_sigma": 0.000,
    "status": "OK"
  },
  "sustainability_extension": {
    "co2_differential_coded": "51%",
    "throughput_cost": "zero for marginal batches"
  },
  "action_verdict": "PROCEED"
}
```

## 5. Interpretation

The instrument **recovers the coded sustainability gap**: sustainability features
contribute ≈ 17.6% of scheduling attribution (target ~20%).
For marginal batches, the telemetry surfaces the CO₂ differential that the scheduler
is blind to.

The snippet does not tell the deployer to reschedule. It tells them: "this batch is
marginal (high flip-rate), and shifting it to off-peak would save ~51% CO₂ at zero
throughput cost." The decision remains with the deployer; the instrument provides
the measurement.
