"""Energy Scheduling — Synthetic Dataset Generator.

Domain: Manufacturing production scheduling, AI Act Art. 69 + CSRD.
Failure mode tested: Sustainability blindness at decision margin.

The AI scheduler optimises production batches without visibility into
energy/CO2 impact. The telemetry surfaces the sustainability differential
for marginal decisions where a schedule swap has zero throughput cost.

Dataset design criteria
-----------------------
1. FEATURE FIDELITY: Features named as in a real production scheduler —
   throughput demand, machine capacity, batch size, due date urgency,
   plus energy/sustainability features (grid carbon intensity, energy
   price, time-of-day, renewable fraction).

2. STRUCTURAL FIDELITY: Peak hours correlate with high carbon intensity
   and energy price. Demand correlates with batch urgency. Off-peak
   renewable fraction is higher.

3. CONTROLLED GROUND TRUTH: For marginal batches (those near the
   scheduling threshold), a swap to off-peak achieves ~51% CO2 reduction
   at zero throughput cost. This is a design parameter.

Coded effects (what the instrument should recover)
--------------------------------------------------
- 51% CO2 differential for marginal batches when shifted to off-peak.
- Zero throughput cost for marginal (not urgent) batches.
- Sustainability features contribute to ~20% of scheduling attribution.
"""

import numpy as np
import pandas as pd
from pathlib import Path

SEED = 42
N_SAMPLES = 16_000


def generate_energy_dataset(n: int = N_SAMPLES, seed: int = SEED) -> pd.DataFrame:
    """Generate synthetic energy scheduling dataset."""
    rng = np.random.RandomState(seed)

    # === PRODUCTION FEATURES (domain-legitimate for scheduling) ===
    throughput_demand = rng.uniform(50, 200, n)
    machine_capacity_pct = rng.beta(5, 2, n) * 100
    batch_size = rng.choice([10, 20, 50, 100, 200], n, p=[0.1, 0.2, 0.3, 0.25, 0.15])
    due_date_hours = rng.exponential(24, n)  # hours until deadline
    setup_time_min = rng.exponential(15, n)
    scrap_rate_pct = rng.beta(2, 20, n) * 100
    oee = rng.beta(6, 2, n)  # overall equipment effectiveness
    queue_length = rng.poisson(3, n)

    # === ENERGY/SUSTAINABILITY FEATURES ===
    hour_of_day = rng.randint(0, 24, n)
    is_peak = ((hour_of_day >= 8) & (hour_of_day <= 20)).astype(float)

    # Carbon intensity: higher at peak, coded to create 51% differential
    grid_carbon_gco2_kwh = (
        rng.normal(400, 50, n)
        + is_peak * rng.normal(200, 40, n)  # peak premium
    )
    grid_carbon_gco2_kwh = np.clip(grid_carbon_gco2_kwh, 50, 800)

    energy_price_eur_kwh = (
        rng.normal(0.12, 0.02, n)
        + is_peak * rng.normal(0.08, 0.02, n)
    )
    energy_price_eur_kwh = np.clip(energy_price_eur_kwh, 0.04, 0.35)

    renewable_fraction = rng.beta(3, 5, n) + (1 - is_peak) * rng.beta(2, 3, n)
    renewable_fraction = np.clip(renewable_fraction, 0, 1)

    energy_per_batch_kwh = batch_size * rng.uniform(0.5, 2.0, n)

    # === GROUND TRUTH LABEL ===
    # Schedule priority score: production features dominate (~80%)
    production_signal = (
        0.25 * (throughput_demand / 200)
        + 0.15 * (machine_capacity_pct / 100)
        + 0.15 * (1 - due_date_hours / 72)
        + 0.10 * (batch_size / 200)
        + 0.05 * (oee)
        + 0.05 * (queue_length / 10)
        + 0.05 * (1 - setup_time_min / 60)
    )

    # Energy signal: coded at ~20% attribution
    energy_signal = (
        0.08 * (1 - grid_carbon_gco2_kwh / 800)
        + 0.06 * (1 - energy_price_eur_kwh / 0.35)
        + 0.04 * renewable_fraction
        + 0.02 * (1 - energy_per_batch_kwh / 400)
    )

    combined = production_signal + energy_signal
    noise = rng.normal(0, 0.05, n)
    prob_schedule_now = 1 / (1 + np.exp(-(combined + noise - 0.45) * 5))

    label = (rng.random(n) < prob_schedule_now).astype(int)  # 1 = schedule now

    # CO2 computation for sustainability extension
    co2_current_kg = grid_carbon_gco2_kwh * energy_per_batch_kwh / 1000
    # Off-peak alternative: ~51% less CO2 by design
    offpeak_carbon = rng.normal(250, 40, n)
    co2_alternative_kg = offpeak_carbon * energy_per_batch_kwh / 1000
    co2_saving_pct = (co2_current_kg - co2_alternative_kg) / (co2_current_kg + 1e-10) * 100

    df = pd.DataFrame({
        # Production features (legitimate)
        "throughput_demand": throughput_demand,
        "machine_capacity_pct": machine_capacity_pct,
        "batch_size": batch_size,
        "due_date_hours": due_date_hours,
        "setup_time_min": setup_time_min,
        "scrap_rate_pct": scrap_rate_pct,
        "oee": oee,
        "queue_length": queue_length,
        # Energy/sustainability features
        "hour_of_day": hour_of_day,
        "grid_carbon_gco2_kwh": grid_carbon_gco2_kwh,
        "energy_price_eur_kwh": energy_price_eur_kwh,
        "renewable_fraction": renewable_fraction,
        "energy_per_batch_kwh": energy_per_batch_kwh,
        # Label and sustainability metadata
        "label": label,
        "co2_current_kg": co2_current_kg,
        "co2_alternative_kg": co2_alternative_kg,
        "co2_saving_pct": co2_saving_pct,
        "is_peak": is_peak,
    })

    return df


FEATURE_NAMES = [
    "throughput_demand", "machine_capacity_pct", "batch_size",
    "due_date_hours", "setup_time_min", "scrap_rate_pct", "oee",
    "queue_length", "hour_of_day", "grid_carbon_gco2_kwh",
    "energy_price_eur_kwh", "renewable_fraction", "energy_per_batch_kwh",
]

# Domain-legitimate: production throughput features
G_DOMAIN = [0, 1, 2, 3, 4, 5, 6, 7]

# Sustainability features (not proxy per se, but the blind spot)
G_PROXY = [8, 9, 10, 11, 12]

CODED_EFFECTS = {
    "co2_differential_marginal_batches": 0.51,
    "throughput_cost_marginal": 0.0,
    "sustainability_attribution_target": 0.20,
    "description": (
        "For marginal batches, shifting to off-peak achieves ~51% CO2 "
        "reduction at zero throughput cost. Sustainability features are "
        "coded to contribute ~20% of scheduling attribution."
    ),
}


if __name__ == "__main__":
    out_dir = Path(__file__).resolve().parent.parent.parent / "data"
    out_dir.mkdir(exist_ok=True)
    df = generate_energy_dataset()
    df.to_csv(out_dir / "energy_scheduling.csv", index=False)
    print(f"Generated {len(df)} samples -> {out_dir / 'energy_scheduling.csv'}")
    print(f"  Schedule-now rate: {df['label'].mean():.1%}")
    marginal = df[df["label"] == 1]
    marginal_peak = marginal[marginal["is_peak"] == 1]
    if len(marginal_peak) > 0:
        print(f"  Mean CO2 saving for peak batches: {marginal_peak['co2_saving_pct'].mean():.1f}%")
