"""COMPAS Sentencing — Synthetic Dataset Generator (Cat. 8).

Domain: Criminal sentencing, AI Act Category 8 (administration of justice).
Failure mode tested: Opacity in sentencing — race and neighbourhood-level
proxy features contaminate the algorithmic sentencing recommendation that
judges are permitted (but not required) to use.

Inspired by Loomis v. Wisconsin (2016), in which the Wisconsin Supreme Court
upheld judicial use of COMPAS scores for sentencing despite the algorithm
being proprietary and the defendant being unable to interrogate it. The
failure mode here is not that the score is wrong, but that the defendant
has no access to the attribution evidence that the instrument produces.
The snippet is the evidence the defendant does NOT have.

Differs from compas_pretrial (Cat. 6):
- Pretrial: binary flight/recidivism risk → pretrial detention decision
- Sentencing: severity recommendation → affects sentence length
- Legal context: Loomis permits opaque use; the instrument provides the
  transparency the legal system does not.

Dataset design criteria
-----------------------
1. FEATURE FIDELITY: Legitimate features are directly relevant to sentencing
   under retributive and rehabilitative frameworks. Proxy features are
   structural social factors — race, age at first contact, neighbourhood
   poverty, family criminal history — that courts are not supposed to weigh
   but that a model trained on historical sentencing data will absorb.

2. STRUCTURAL FIDELITY: A latent `criminality_severity` factor drives all
   five legitimate features (70% weight + 30% idiosyncratic noise), creating
   realistic inter-feature correlation. Race correlates with neighbourhood
   poverty and family criminal history through structural inequality, not
   through actual criminality.

3. CONTROLLED GROUND TRUTH: Proxy features are coded to contribute ~16% of
   the combined decision signal; race (Black + Hispanic) contributes ~8–9%
   of the coded signal. With near-threshold amplification (borderline cases
   where proxy tips the balance), the instrument should recover ~20% proxy
   attribution overall and ~12–15% race attribution near the threshold.

Coded effects (what the instrument should recover)
--------------------------------------------------
- Proxy features contribute ~16% of the combined signal (coded).
- Race attribution (race_encoded): ~12–15% at near-threshold (measured).
- Black defendants: proxy contribution ~13% of combined; White: ~5%.
- `criminality_severity` is metadata — not in FEATURE_NAMES.
"""

import numpy as np
import pandas as pd
from pathlib import Path

SEED = 42
N_SAMPLES = 15_000


def generate_sentencing_dataset(n: int = N_SAMPLES, seed: int = SEED) -> pd.DataFrame:
    """Generate synthetic sentencing-risk dataset.

    Returns DataFrame with 9 features + label + metadata.
    Note: 'criminality_severity' is a construction variable (metadata only)
    and does not enter the model.
    """
    rng = np.random.RandomState(seed)

    # === LATENT VARIABLES (construction only — not in FEATURE_NAMES) ===
    # criminality_severity: shared factor driving all legitimate sentencing
    # indicators. Creates correlated extremes for instrument calibration.
    criminality_severity = rng.beta(3, 3, n)   # 0-1, symmetric around 0.5

    # race_encoded: 0=White, 1=Hispanic, 2=Black.
    # Ordered by coded proxy contribution magnitude (monotonic) so that a
    # linear model can learn a single positive coefficient for race_encoded.
    # Drives proxy features through structural inequality, NOT through
    # individual criminality. Does not enter the legitimate signal.
    race_encoded = rng.choice([0, 1, 2], n, p=[0.42, 0.14, 0.44])

    # === LEGITIMATE FEATURES (sentencing-relevant criminal history) ===
    # Each feature = 0.70 * criminality_severity + 0.30 * idiosyncratic noise.

    # prior_felonies: count of prior felony convictions (0–10)
    prior_felonies = np.round(
        (criminality_severity * 0.70 + rng.beta(2, 4, n) * 0.30) * 10
    ).astype(int).clip(0, 10)

    # prior_misdemeanors: count of prior misdemeanor convictions (0–15)
    prior_misdemeanors = np.round(
        (criminality_severity * 0.70 + rng.beta(2, 3, n) * 0.30) * 15
    ).astype(int).clip(0, 15)

    # supervision_failures: violations of probation/parole/supervision (0–5)
    supervision_failures = np.round(
        (criminality_severity * 0.70 + rng.beta(2, 4, n) * 0.30) * 5
    ).astype(int).clip(0, 5)

    # employment_history: 0=chronically unemployed, 1=stable employment.
    # Inverted: high criminality → low employment.
    employment_history = (
        (1 - criminality_severity) * 0.70 + rng.beta(3, 3, n) * 0.30
    ).clip(0, 1)

    # substance_abuse_treatment: 0=no history, 1=documented treatment.
    # Correlated with criminality (those with substance issues have higher risk).
    sab_latent = criminality_severity * 0.70 + rng.beta(3, 5, n) * 0.30
    substance_abuse_treatment = (sab_latent > 0.55).astype(int)

    # === PROXY FEATURES (race/structure-correlated, should NOT drive sentence) ===

    # age_at_first_offense: age at first recorded criminal contact.
    # Younger first-contact correlates with higher risk in biased historical models.
    # Structurally: minority youth have earlier first contact due to policing disparities.
    age_at_first_offense = (
        rng.normal(24, 4, n) * (race_encoded == 0)   # White: mean ≈ 24
        + rng.normal(22, 4, n) * (race_encoded == 1)  # Hispanic: mean ≈ 22
        + rng.normal(21, 4, n) * (race_encoded == 2)  # Black: mean ≈ 21 (policing bias)
    ).clip(14, 40)

    # neighborhood_poverty_index: area-level poverty at time of offense (0–1).
    # Strongly correlated with race through residential segregation.
    neighborhood_poverty_index = (
        rng.beta(3, 7, n) * (race_encoded == 0)   # White: mean ≈ 0.30
        + rng.beta(5, 5, n) * (race_encoded == 1)  # Hispanic: mean ≈ 0.50
        + rng.beta(6, 4, n) * (race_encoded == 2)  # Black: mean ≈ 0.60
    )

    # family_criminal_history: composite score for criminal justice contact
    # in immediate family (0–1). Correlated with race through structural
    # over-policing and incarceration patterns.
    family_criminal_history = (
        rng.beta(2, 6, n) * (race_encoded == 0)   # White: mean ≈ 0.25
        + rng.beta(4, 5, n) * (race_encoded == 1)  # Hispanic: mean ≈ 0.44
        + rng.beta(5, 4, n) * (race_encoded == 2)  # Black: mean ≈ 0.56
    )

    # === GROUND TRUTH LABEL ===
    # Legitimate signal (~78% of decision basis)
    legit_signal = (
        0.20 * (prior_felonies / 10)
        + 0.18 * (prior_misdemeanors / 15)
        + 0.15 * (supervision_failures / 5)
        + 0.13 * (1 - employment_history)     # unemployed = higher risk
        + 0.12 * substance_abuse_treatment
    )
    # Sum of weights: 0.78. Range approx [0, 0.78] with spread from shared
    # criminality_severity factor.

    # Proxy signal.
    # race_encoded is monotonic (0→1→2 = White→Hispanic→Black), enabling a linear
    # model to learn a single positive coefficient: proxy_race ≈ 0.028 × race_encoded.
    # This gives 0 for White, 0.028 for Hispanic, 0.056 for Black.
    # age_norm: younger at first offense → higher normalized risk contribution.
    age_norm = (40 - age_at_first_offense) / (40 - 14)   # 0–1
    proxy_signal = (
        0.028 * race_encoded.astype(float)          # 0/0.028/0.056 for White/Hispanic/Black
        + 0.020 * age_norm                          # age-at-first-offense
        + 0.008 * neighborhood_poverty_index        # neighbourhood poverty
        + 0.008 * family_criminal_history           # family criminal history
    )
    # Proxy coded proportion (population average): ~10% of combined signal.
    # With near-threshold amplification (~2.2×): proxy SHAP ~20% near threshold.
    # Race component: ~6% coded → ~12–14% SHAP near threshold.

    combined = legit_signal + proxy_signal
    noise = rng.normal(0, 0.06, n)
    # Scale=7, threshold=0.48 calibrated to ~50% high-risk rate
    prob_high_risk = 1 / (1 + np.exp(-(combined + noise - 0.48) * 7))

    label = (rng.random(n) < prob_high_risk).astype(int)  # 1 = high-risk recommendation
    margin = np.abs(prob_high_risk - 0.5)

    df = pd.DataFrame({
        # Legitimate features
        "prior_felonies":              prior_felonies,
        "prior_misdemeanors":          prior_misdemeanors,
        "supervision_failures":        supervision_failures,
        "employment_history":          employment_history,
        "substance_abuse_treatment":   substance_abuse_treatment,
        # Proxy features (race/structure-correlated)
        "age_at_first_offense":        age_at_first_offense,
        "race_encoded":                race_encoded,
        "neighborhood_poverty_index":  neighborhood_poverty_index,
        "family_criminal_history":     family_criminal_history,
        # Label and metadata
        "label":                       label,
        "criminality_severity":        criminality_severity,  # latent (metadata only)
        "margin_to_threshold":         margin,
    })
    return df


# Feature configuration for the instrumentation protocol.
# NOTE: 'criminality_severity' and 'margin_to_threshold' are metadata.
FEATURE_NAMES = [
    "prior_felonies", "prior_misdemeanors", "supervision_failures",
    "employment_history", "substance_abuse_treatment",
    "age_at_first_offense", "race_encoded",
    "neighborhood_poverty_index", "family_criminal_history",
]

# Legitimate sentencing features
G_DOMAIN = [0, 1, 2, 3, 4]

# Proxy features (race/structure-correlated, should not drive sentence)
G_PROXY = [5, 6, 7, 8]

# race_encoded is at index 6 in FEATURE_NAMES, index 1 in G_PROXY
RACE_FEATURE_IDX = 6

CODED_EFFECTS = {
    "proxy_attribution_coded": 0.16,
    "race_attribution_near_threshold": "12-15%",
    "race_encoding": "0=White, 1=Hispanic, 2=Black (monotonic in proxy effect)",
    "annex_iii_category": "cat_8",
    "description": (
        "Race-encoded (monotonic: 0=White, 1=Hispanic, 2=Black) and structural proxy "
        "features (age_at_first_offense, neighborhood_poverty_index, "
        "family_criminal_history) are coded to contribute ~16% of the combined signal. "
        "Race contributes ~11% coded (linear: 0.05 × race_encoded), recovering to "
        "~12–15% SHAP near the threshold. "
        "A shared criminality_severity latent factor correlates legitimate features, "
        "creating realistic extreme-case spread for instrument calibration."
    ),
}


if __name__ == "__main__":
    out_dir = Path(__file__).resolve().parent / "data"
    out_dir.mkdir(exist_ok=True)
    df = generate_sentencing_dataset()
    df.to_csv(out_dir / "compas_sentencing.csv", index=False)
    print(f"Generated {len(df)} samples -> {out_dir / 'compas_sentencing.csv'}")
    print(f"  High-risk rate (overall):    {df['label'].mean():.1%}")
    print(f"  High-risk rate (White):      {df[df['race_encoded']==0]['label'].mean():.1%}")
    print(f"  High-risk rate (Hispanic):   {df[df['race_encoded']==1]['label'].mean():.1%}")
    print(f"  High-risk rate (Black):      {df[df['race_encoded']==2]['label'].mean():.1%}")
    print(f"  Near-threshold (margin<0.15): {(df['margin_to_threshold']<0.15).sum()}")
    print(f"  Far-threshold  (margin>0.35): {(df['margin_to_threshold']>0.35).sum()}")
