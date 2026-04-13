"""COMPAS Pretrial Risk Assessment — Synthetic Dataset Generator.

Domain: Criminal justice, AI Act Category 6 (law enforcement).
Failure mode tested: Constraint enforcement — non-criminal-history features
dominate the risk score.

Due process requires that pretrial risk assessment be based on criminal
history (priors, charges, juvenile record). The instrument tests whether
protected attributes (age, race, sex) contaminate the decision.

Dataset design criteria
-----------------------
1. FEATURE FIDELITY: Features named as in ProPublica's COMPAS analysis —
   priors count, juvenile felonies/misdemeanors, charge degree, plus
   demographics (age, race, sex).

2. STRUCTURAL FIDELITY: Race correlates with priors count and charge
   severity (structural inequality in criminal justice). Age correlates
   with recidivism base rate. These real-world correlations make the
   proxy contamination realistic and hard to detect without instrumentation.

3. CONTROLLED GROUND TRUTH: 47% of the risk score is coded to come from
   non-criminal-history features. Race carries 11-17% attribution.
   The instrument should recover this contamination.

Coded effects (what the instrument should recover)
--------------------------------------------------
- 47% of risk score is NOT from criminal history.
- Race attribution: 11-17% depending on subgroup.
- Age: ~8% attribution (partially legitimate but disproportionate).
"""

import numpy as np
import pandas as pd
from pathlib import Path

SEED = 42
N_SAMPLES = 7_000


def generate_compas_dataset(n: int = N_SAMPLES, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.RandomState(seed)

    # === PROTECTED ATTRIBUTES ===
    race = rng.choice([0, 1, 2], n, p=[0.34, 0.51, 0.15])  # 0=Caucasian, 1=African-American, 2=Hispanic
    sex = rng.choice([0, 1], n, p=[0.81, 0.19])  # 0=male, 1=female
    age = rng.normal(34, 12, n).clip(18, 80)

    # === CRIMINAL HISTORY FEATURES (legitimate for risk assessment) ===
    # Priors: structural correlation with race (structural fidelity)
    priors_count = rng.poisson(3, n) + (race == 1) * rng.poisson(1, n)
    priors_count = priors_count.clip(0, 38)

    juvenile_felonies = rng.poisson(0.1, n)
    juvenile_misdemeanors = rng.poisson(0.3, n)
    juvenile_other = rng.poisson(0.2, n)

    charge_degree = rng.choice([0, 1], n, p=[0.55, 0.45])  # 0=felony, 1=misdemeanor

    # === GROUND TRUTH LABEL ===
    # Criminal history signal (~53% of decision)
    criminal_signal = (
        0.22 * (priors_count / 15).clip(0, 1)
        + 0.12 * charge_degree  # misdemeanor = lower risk
        + 0.08 * (juvenile_felonies / 3).clip(0, 1)
        + 0.06 * (juvenile_misdemeanors / 3).clip(0, 1)
        + 0.05 * (juvenile_other / 3).clip(0, 1)
    )

    # Proxy contamination: ~47% from non-criminal-history features
    # Race: 11-17% attribution
    proxy_signal = (
        0.12 * (race == 1).astype(float)  # African-American
        + 0.05 * (race == 2).astype(float)  # Hispanic
        + 0.08 * (1 - age / 80)  # younger = higher coded risk
        + 0.04 * (1 - sex)  # male = higher coded risk
    )

    # Additional non-criminal opaque factors (simulate COMPAS internal logic)
    opaque_signal = (
        0.10 * rng.beta(3, 5, n)  # "socialisation" score (opaque)
        + 0.08 * rng.beta(4, 4, n)  # "community ties" score (opaque)
    )

    combined = criminal_signal + proxy_signal + opaque_signal
    noise = rng.normal(0, 0.07, n)
    prob_recid = 1 / (1 + np.exp(-(combined + noise - 0.45) * 4))
    label = (rng.random(n) < prob_recid).astype(int)  # 1 = high risk

    df = pd.DataFrame({
        "priors_count": priors_count,
        "charge_degree": charge_degree,
        "juvenile_felonies": juvenile_felonies,
        "juvenile_misdemeanors": juvenile_misdemeanors,
        "juvenile_other": juvenile_other,
        "age": age,
        "race": race,
        "sex": sex,
        "label": label,
    })
    return df


FEATURE_NAMES = [
    "priors_count", "charge_degree", "juvenile_felonies",
    "juvenile_misdemeanors", "juvenile_other", "age", "race", "sex",
]

# Domain-legitimate: criminal history
G_DOMAIN = [0, 1, 2, 3, 4]

# Protected: demographics
G_PROXY = [5, 6, 7]

CODED_EFFECTS = {
    "non_criminal_history_pct": 0.47,
    "race_attribution_range": "11-17%",
    "age_attribution": 0.08,
    "description": (
        "47% of risk score comes from non-criminal-history features. "
        "Race carries 11-17% attribution. Age carries ~8%. Due process "
        "requires criminal history to dominate; the instrument should "
        "recover this contamination."
    ),
}


if __name__ == "__main__":
    out_dir = Path(__file__).resolve().parent.parent.parent / "data"
    out_dir.mkdir(exist_ok=True)
    df = generate_compas_dataset()
    df.to_csv(out_dir / "compas_pretrial.csv", index=False)
    print(f"Generated {len(df)} samples -> {out_dir / 'compas_pretrial.csv'}")
    print(f"  High-risk rate: {df['label'].mean():.1%}")
    print(f"  Race distribution: Caucasian={( df['race']==0).mean():.1%}, AA={(df['race']==1).mean():.1%}, Hispanic={(df['race']==2).mean():.1%}")
