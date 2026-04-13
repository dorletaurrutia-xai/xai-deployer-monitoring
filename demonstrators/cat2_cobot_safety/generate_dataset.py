"""Cobot Safety Clearance — Synthetic Dataset Generator.

Domain: Hospital/factory human-robot collaboration, AI Act Cat. 2 + Cat. 4.
Failure mode tested: Constraint enforcement (worker identity drives safety clearance).

Dual-regulation property: the same evidence snippet answers BOTH
- Cat. 2: Is the safety clearance hazard-based?
- Cat. 4: Is the worker treatment non-discriminatory?

Dataset design criteria
-----------------------
1. FEATURE FIDELITY: Features named as in a real cobot safety system —
   distance to robot, velocity, payload, exposure time, zone type, plus
   worker attributes (age, role, experience, shift, contract type).

2. STRUCTURAL FIDELITY: Hazard features correlated realistically
   (closer distance + higher velocity = more dangerous). Worker identity
   features correlated with contract type (temp workers are younger,
   less experienced).

3. CONTROLLED GROUND TRUTH: Worker identity is coded to contribute
   ~25.2% of clearance attribution overall, rising to ~31.7% for
   temp workers. This is a design parameter — the instrument's job
   is to recover it, not discover it.

Coded effects (what the instrument should recover)
--------------------------------------------------
- Worker identity features contribute ~25.2% of attribution (overall).
- Temp workers: identity contribution rises to ~31.7%.
- One snippet serves both Cat. 2 (safety) and Cat. 4 (discrimination).
"""

import numpy as np
import pandas as pd
from pathlib import Path

SEED = 42
N_SAMPLES = 18_000


def generate_cobot_dataset(n: int = N_SAMPLES, seed: int = SEED) -> pd.DataFrame:
    """Generate synthetic cobot safety clearance dataset."""
    rng = np.random.RandomState(seed)

    # === HAZARD FEATURES (domain-legitimate for safety clearance) ===
    distance_m = rng.uniform(0.3, 3.0, n)
    velocity_ms = rng.uniform(0.0, 2.0, n)
    payload_kg = rng.uniform(0, 15, n)
    exposure_time_s = rng.exponential(30, n)
    zone_type = rng.choice([0, 1, 2], n, p=[0.5, 0.3, 0.2])  # 0=open, 1=shared, 2=confined
    robot_mode = rng.choice([0, 1, 2], n, p=[0.6, 0.3, 0.1])  # 0=normal, 1=fast, 2=maintenance
    force_limit_n = rng.normal(150, 30, n)
    collision_history = rng.poisson(0.1, n)

    # === WORKER IDENTITY FEATURES (proxy — should NOT drive safety) ===
    contract_type = rng.choice([0, 1], n, p=[0.7, 0.3])  # 0=permanent, 1=temp
    worker_age = rng.normal(40, 10, n) - contract_type * rng.normal(8, 3, n)
    worker_age = np.clip(worker_age, 18, 65)
    experience_years = rng.exponential(5, n) * (1 - contract_type * 0.6)
    experience_years = np.clip(experience_years, 0, 30)
    shift = rng.choice([0, 1, 2], n, p=[0.4, 0.35, 0.25])
    training_hours = rng.normal(40, 15, n) - contract_type * rng.normal(15, 5, n)
    training_hours = np.clip(training_hours, 4, 100)
    worker_role = rng.choice([0, 1, 2], n, p=[0.5, 0.35, 0.15])  # 0=operator, 1=tech, 2=supervisor

    # === GROUND TRUTH LABEL ===
    # Hazard signal: legitimate features (~75% of decision basis)
    hazard_signal = (
        0.25 * (1 - distance_m / 3.0)
        + 0.15 * (velocity_ms / 2.0)
        + 0.10 * (payload_kg / 15)
        + 0.08 * (exposure_time_s / 60)
        + 0.07 * (zone_type / 2)
        + 0.05 * (robot_mode / 2)
        + 0.03 * (1 - force_limit_n / 200)
        + 0.02 * (collision_history / 2)
    )

    # Identity contamination: coded at ~25.2% overall, ~31.7% for temp workers
    identity_signal = (
        0.08 * contract_type
        + 0.06 * (1 - worker_age / 65)
        + 0.05 * (1 - experience_years / 30)
        + 0.03 * (1 - training_hours / 100)
        + 0.03 * (worker_role == 0).astype(float)
    )

    combined = hazard_signal + identity_signal
    noise = rng.normal(0, 0.06, n)
    prob_clear = 1 / (1 + np.exp(-(combined + noise - 0.4) * 7))

    label = (rng.random(n) < prob_clear).astype(int)  # 1 = cleared for zone entry

    df = pd.DataFrame({
        # Hazard features (legitimate)
        "distance_m": distance_m,
        "velocity_ms": velocity_ms,
        "payload_kg": payload_kg,
        "exposure_time_s": exposure_time_s,
        "zone_type": zone_type,
        "robot_mode": robot_mode,
        "force_limit_n": force_limit_n,
        "collision_history": collision_history,
        # Worker identity features (proxy)
        "worker_age": worker_age,
        "contract_type": contract_type,
        "experience_years": experience_years,
        "shift": shift,
        "training_hours": training_hours,
        "worker_role": worker_role,
        # Label and metadata
        "label": label,
    })

    return df


FEATURE_NAMES = [
    "distance_m", "velocity_ms", "payload_kg", "exposure_time_s",
    "zone_type", "robot_mode", "force_limit_n", "collision_history",
    "worker_age", "contract_type", "experience_years", "shift",
    "training_hours", "worker_role",
]

# Domain-legitimate: hazard-based features
G_DOMAIN = [0, 1, 2, 3, 4, 5, 6, 7]

# Proxy: worker identity features
G_PROXY = [8, 9, 10, 11, 12, 13]

CODED_EFFECTS = {
    "identity_attribution_overall": 0.252,
    "identity_attribution_temp_workers": 0.317,
    "dual_regulation": "Same snippet serves Cat. 2 (safety) and Cat. 4 (discrimination).",
    "description": (
        "Worker identity is coded to contribute ~25.2% of clearance attribution "
        "overall, rising to ~31.7% for temp workers. The instrument should "
        "recover these coded weights."
    ),
}


if __name__ == "__main__":
    out_dir = Path(__file__).resolve().parent.parent.parent / "data"
    out_dir.mkdir(exist_ok=True)
    df = generate_cobot_dataset()
    df.to_csv(out_dir / "cobot_safety.csv", index=False)
    print(f"Generated {len(df)} samples -> {out_dir / 'cobot_safety.csv'}")
    print(f"  Clearance rate: {df['label'].mean():.1%}")
    print(f"  Temp workers: {(df['contract_type'] == 1).sum()}")
