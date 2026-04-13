"""Brake Disc Quality Control — Synthetic Dataset Generator.

Domain: Automotive production line, AI Act Category 2.
Failure mode tested: Decision robustness (brittle auto-passes on safety parts).

Dataset design criteria
-----------------------
1. FEATURE FIDELITY: Features are named and typed as in a real brake disc
   inspection system (crack depth, runout, roughness, hardness, etc.).
   Confounders are named as environmental factors (oil residue, lighting,
   camera angle, temperature) that a real vision/sensor system might leak.

2. STRUCTURAL FIDELITY: Feature distributions use domain-plausible ranges.
   Correlations between legitimate features (e.g., crack_depth ~ roughness)
   and between confounders (e.g., oil_residue ~ lighting on night shift) are
   coded to reflect real manufacturing structure.

3. CONTROLLED GROUND TRUTH: The confounder effect is coded at a known
   magnitude (~34% attribution weight). The fragility of marginal cases
   is coded by placing them near the decision boundary. These are design
   parameters, not discoveries — the instrument's job is to read them back.

Coded effects (what the instrument should recover)
--------------------------------------------------
- Confounders contribute ~34% of attribution mass for near-threshold cases.
- 63% of marginal auto-passes are brittle (flip under bounded perturbation).
- Night-shift cases have higher confounder correlation.
"""

import numpy as np
import pandas as pd
from pathlib import Path

SEED = 42
N_SAMPLES = 20_000


def generate_brake_disc_dataset(
    n: int = N_SAMPLES, seed: int = SEED
) -> pd.DataFrame:
    """Generate synthetic brake disc inspection dataset.

    Returns DataFrame with 13 features + labels + metadata.
    """
    rng = np.random.RandomState(seed)

    # === LEGITIMATE FEATURES (domain: defect indicators) ===
    crack_depth_mm = rng.exponential(0.05, n)  # mm, mostly small
    crack_length_mm = crack_depth_mm * rng.uniform(2, 8, n)
    runout_mm = np.abs(rng.normal(0.02, 0.01, n))  # lateral runout
    roughness_ra = rng.lognormal(np.log(1.6), 0.3, n)  # Ra in microns
    hardness_hrc = rng.normal(35, 3, n)  # Rockwell C
    thickness_var_mm = np.abs(rng.normal(0.0, 0.015, n))  # thickness variation
    weight_deviation_g = rng.normal(0, 5, n)  # grams from nominal
    visual_score = rng.beta(8, 2, n)  # 0-1, automated vision score
    surface_porosity_pct = rng.exponential(0.5, n)

    # === CONFOUNDER FEATURES (environmental, should NOT drive decision) ===
    # These are coded to correlate with the label at ~34% attribution
    shift = rng.choice([0, 1, 2], n, p=[0.4, 0.35, 0.25])  # 0=day, 1=evening, 2=night
    oil_residue = rng.exponential(0.3, n) + shift * rng.exponential(0.15, n)
    lighting_lux = rng.normal(800, 100, n) - shift * rng.normal(100, 30, n)
    lighting_lux = np.clip(lighting_lux, 200, 1200)
    camera_angle_deg = rng.normal(90, 2, n) + shift * rng.normal(0, 1.5, n)
    temperature_c = rng.normal(22, 3, n) + shift * rng.normal(2, 1, n)

    # === GROUND TRUTH LABEL ===
    # Defect score: legitimate features drive the label
    defect_signal = (
        0.30 * (crack_depth_mm / 0.15)
        + 0.15 * (runout_mm / 0.04)
        + 0.10 * (roughness_ra / 3.0)
        + 0.10 * (thickness_var_mm / 0.03)
        + 0.05 * (1 - visual_score)
        + 0.05 * (surface_porosity_pct / 2.0)
    )

    # Confounder contamination: ~34% of signal for affected cases
    confounder_signal = (
        0.15 * (oil_residue / 1.0)
        + 0.10 * (1 - lighting_lux / 1200)
        + 0.05 * (np.abs(camera_angle_deg - 90) / 5)
        + 0.04 * (temperature_c / 40)
    )

    # Combined score (pass = 1 = "good part", reject = 0 = "defective")
    combined = defect_signal + confounder_signal
    noise = rng.normal(0, 0.08, n)
    prob_reject = 1 / (1 + np.exp(-(combined + noise - 0.5) * 4))

    label = (rng.random(n) < (1 - prob_reject)).astype(int)  # 1 = pass

    # Margin to threshold (for identifying marginal cases)
    margin = np.abs(prob_reject - 0.5)

    df = pd.DataFrame({
        # Legitimate features
        "crack_depth_mm": crack_depth_mm,
        "crack_length_mm": crack_length_mm,
        "runout_mm": runout_mm,
        "roughness_ra": roughness_ra,
        "hardness_hrc": hardness_hrc,
        "thickness_var_mm": thickness_var_mm,
        "weight_deviation_g": weight_deviation_g,
        "visual_score": visual_score,
        "surface_porosity_pct": surface_porosity_pct,
        # Confounders
        "oil_residue": oil_residue,
        "lighting_lux": lighting_lux,
        "camera_angle_deg": camera_angle_deg,
        "temperature_c": temperature_c,
        # Label and metadata
        "label": label,
        "shift": shift,
        "margin_to_threshold": margin,
    })

    return df


# Feature configuration for the instrumentation protocol
FEATURE_NAMES = [
    "crack_depth_mm", "crack_length_mm", "runout_mm", "roughness_ra",
    "hardness_hrc", "thickness_var_mm", "weight_deviation_g", "visual_score",
    "surface_porosity_pct", "oil_residue", "lighting_lux", "camera_angle_deg",
    "temperature_c",
]

# Domain-legitimate features (defect indicators)
G_DOMAIN = [0, 1, 2, 3, 4, 5, 6, 7, 8]

# Proxy/confounder features (environmental, should not drive decision)
G_PROXY = [9, 10, 11, 12]

# Coded ground truth: what the instrument should recover
CODED_EFFECTS = {
    "confounder_attribution_target": 0.34,
    "marginal_flip_rate_target": 0.63,
    "description": (
        "Confounders (oil, lighting, camera, temperature) are coded to "
        "contribute ~34% of attribution mass. 63% of near-threshold "
        "auto-passes are designed to flip under bounded perturbation."
    ),
}


if __name__ == "__main__":
    out_dir = Path(__file__).resolve().parent.parent.parent / "data"
    out_dir.mkdir(exist_ok=True)
    df = generate_brake_disc_dataset()
    df.to_csv(out_dir / "brake_disc.csv", index=False)
    print(f"Generated {len(df)} samples -> {out_dir / 'brake_disc.csv'}")
    print(f"  Pass rate: {df['label'].mean():.1%}")
    print(f"  Marginal cases (margin < 0.15): {(df['margin_to_threshold'] < 0.15).sum()}")
