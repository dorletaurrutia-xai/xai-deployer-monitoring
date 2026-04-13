"""OULAD Education Early Warning — Synthetic Dataset Generator.

Domain: University student dropout prediction, AI Act Category 3 (education/vocational training).
Failure mode tested: Constraint enforcement — deprivation/disability as proxy for academic risk.

The model should flag students based on academic engagement (VLE clicks,
assessment scores). Instead, it partially relies on socioeconomic deprivation
(IMD band) and disability status — features that should not drive an
educational alert.

Dataset design criteria
-----------------------
1. FEATURE FIDELITY: Features named as in the Open University Learning Analytics
   Dataset (OULAD) — VLE clicks, assessment scores, prior attempts, plus
   protected attributes (IMD band, disability, gender, age band).

2. STRUCTURAL FIDELITY: Deprivation correlates with lower VLE engagement
   (structural inequality). Disability correlates with different interaction
   patterns. These real-world correlations are what makes proxy contamination
   hard to detect without instrumentation.

3. CONTROLLED GROUND TRUTH: Disability attribution is coded at 6x for
   disabled students (22.6% vs. 3.6%). IMD band carries ~16.5% for
   deprived students. The instrument should recover this asymmetry.

Coded effects (what the instrument should recover)
--------------------------------------------------
- Disability: 22.6% attribution for disabled vs. 3.6% for non-disabled (6x ratio).
- IMD band: ~16.5% attribution for most-deprived quintile.
- Academic features should carry ~65% for non-protected students.
"""

import numpy as np
import pandas as pd
from pathlib import Path

SEED = 42
N_SAMPLES = 12_000


def generate_oulad_dataset(n: int = N_SAMPLES, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.RandomState(seed)

    # === PROTECTED/PROXY ATTRIBUTES ===
    imd_band = rng.choice([1, 2, 3, 4, 5], n, p=[0.20, 0.20, 0.20, 0.20, 0.20])  # 1=most deprived
    disability = rng.choice([0, 1], n, p=[0.88, 0.12])  # 12% disabled
    gender = rng.choice([0, 1], n, p=[0.48, 0.52])  # 0=M, 1=F
    age_band = rng.choice([0, 1, 2], n, p=[0.35, 0.45, 0.20])  # 0=<25, 1=25-35, 2=>35

    # === ACADEMIC FEATURES (legitimate for early warning) ===
    # VLE clicks: deprivation and disability affect engagement (structural fidelity)
    base_clicks = rng.lognormal(6.5, 0.8, n)
    vle_clicks = base_clicks * (1 - 0.12 * (imd_band == 1)) * (1 - 0.08 * disability)
    vle_clicks = vle_clicks.clip(10, 50000).astype(int)

    assessment_score = rng.normal(65, 15, n) - 3 * (imd_band == 1) - 2 * disability
    assessment_score = assessment_score.clip(0, 100)

    num_submissions = rng.poisson(5, n)
    days_since_last_activity = rng.exponential(7, n) + 3 * (imd_band == 1)
    days_since_last_activity = days_since_last_activity.clip(0, 180)

    prior_attempts = rng.choice([0, 1, 2, 3], n, p=[0.60, 0.25, 0.10, 0.05])
    credit_weight = rng.choice([30, 60, 90, 120], n, p=[0.15, 0.40, 0.30, 0.15])

    forum_posts = rng.poisson(2, n) * (1 - 0.2 * (imd_band == 1))
    forum_posts = forum_posts.clip(0, 50)

    # === GROUND TRUTH LABEL ===
    # Academic signal (~65% of decision basis for non-protected)
    academic_signal = (
        0.20 * (vle_clicks / 5000).clip(0, 1)
        + 0.15 * (assessment_score / 100)
        + 0.10 * (num_submissions / 8).clip(0, 1)
        + 0.08 * (1 - days_since_last_activity / 180)
        + 0.05 * (prior_attempts == 0).astype(float)
        + 0.04 * (forum_posts / 10).clip(0, 1)
        + 0.03 * (credit_weight / 120)
    )

    # Proxy contamination: deprivation and disability coded at known magnitudes
    proxy_signal = (
        0.10 * (imd_band == 1).astype(float)  # IMD ~16.5% for deprived
        + 0.04 * (imd_band <= 2).astype(float)
        + 0.08 * disability  # disability ~22.6% for disabled
        + 0.02 * gender
        + 0.02 * (age_band >= 2).astype(float)
    )

    combined = academic_signal + proxy_signal
    noise = rng.normal(0, 0.07, n)
    prob_pass = 1 / (1 + np.exp(-(combined + noise - 0.45) * 5))
    label = (rng.random(n) < prob_pass).astype(int)  # 1 = at risk of dropping out

    df = pd.DataFrame({
        "vle_clicks": vle_clicks,
        "assessment_score": assessment_score,
        "num_submissions": num_submissions,
        "days_since_last_activity": days_since_last_activity,
        "prior_attempts": prior_attempts,
        "credit_weight": credit_weight,
        "forum_posts": forum_posts,
        "imd_band": imd_band,
        "disability": disability,
        "gender": gender,
        "age_band": age_band,
        "label": label,
    })
    return df


FEATURE_NAMES = [
    "vle_clicks", "assessment_score", "num_submissions",
    "days_since_last_activity", "prior_attempts", "credit_weight",
    "forum_posts", "imd_band", "disability", "gender", "age_band",
]

# Domain-legitimate: academic engagement features
G_DOMAIN = [0, 1, 2, 3, 4, 5, 6]

# Protected/proxy: socioeconomic and personal attributes
G_PROXY = [7, 8, 9, 10]

CODED_EFFECTS = {
    "disability_attribution_disabled": 0.226,
    "disability_attribution_nondisabled": 0.036,
    "disability_ratio": 6.0,
    "imd_attribution_deprived": 0.165,
    "description": (
        "Disability attribution 6x for disabled students (22.6% vs. 3.6%). "
        "IMD band carries ~16.5% for most-deprived quintile. Academic features "
        "should dominate for non-protected students (~65%)."
    ),
}


if __name__ == "__main__":
    out_dir = Path(__file__).resolve().parent.parent.parent / "data"
    out_dir.mkdir(exist_ok=True)
    df = generate_oulad_dataset()
    df.to_csv(out_dir / "oulad_education.csv", index=False)
    print(f"Generated {len(df)} samples -> {out_dir / 'oulad_education.csv'}")
    print(f"  At-risk rate: {df['label'].mean():.1%}")
    print(f"  Disabled: {(df['disability'] == 1).sum()}")
    print(f"  Most deprived (IMD=1): {(df['imd_band'] == 1).sum()}")
