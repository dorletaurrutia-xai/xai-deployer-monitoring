"""Amazon Hiring Algorithm — Synthetic Dataset Generator (Cat. 4).

Domain: Automated CV/resume screening, AI Act Category 4 (employment).
Failure mode tested: Constraint enforcement — gender-correlated proxy features
contaminate shortlisting decisions.

Inspired by Amazon's documented ML hiring tool (2014–2018), which learned to
penalise CVs containing signals correlated with female gender because it was
trained on 10 years of accepted CVs (predominantly male). Amazon withdrew it
in 2018 without deployment.

Dataset design criteria
-----------------------
1. FEATURE FIDELITY: Features named as in a real CV-screening system.
   Legitimate features are job-relevant merit signals. Proxy features are
   plausible CV artefacts that a screening model could exploit as gender
   shortcuts: technical keyword score (biased towards masculine terminology),
   extracurricular leadership score (male-coded in historical corpora via
   sports captaincy), all-women-college score (penalised in biased training
   data), and referral network score (male network homophily).

2. STRUCTURAL FIDELITY: A latent `candidate_quality` factor drives all five
   merit features (high-quality candidates excel across all merit dimensions
   simultaneously). This creates realistic correlation among merit features
   and sufficient extreme cases for instrument calibration.

   A separate latent `gender` variable (0=male, 1=female) drives the four
   proxy features but never enters the model. Merit features are independent
   of gender by design.

3. CONTROLLED GROUND TRUTH: Proxy features are coded to contribute ~25% of
   selection attribution overall. Cases where proxy attribution exceeds 30%
   (proxy-dominated selections) target ~20% of near-threshold decisions.

Coded effects (what the instrument should recover)
--------------------------------------------------
- Proxy features contribute ~25% of attribution overall.
- ~20% of near-threshold selections are proxy-dominated (>30% proxy attribution).
- Male shortlist rate ≈ 55–57%, female ≈ 43–45% (~12pp gap from proxy signal).
"""

import numpy as np
import pandas as pd
from pathlib import Path

SEED = 42
N_SAMPLES = 20_000


def generate_hiring_dataset(n: int = N_SAMPLES, seed: int = SEED) -> pd.DataFrame:
    """Generate synthetic CV-screening dataset.

    Returns DataFrame with 9 features + labels + metadata.
    Note: 'gender' and 'candidate_quality' are construction variables
    (metadata only) — not in FEATURE_NAMES and do not enter the model.
    """
    rng = np.random.RandomState(seed)

    # === LATENT VARIABLES (construction only — not in FEATURE_NAMES) ===
    # Candidate quality: shared factor driving all merit dimensions.
    # Creates realistic merit feature correlation and extreme-case spread.
    candidate_quality = rng.beta(3, 3, n)   # 0-1, roughly symmetric around 0.5

    # Gender: 0=male, 1=female. Drives proxy features, not merit.
    gender = rng.randint(0, 2, n)

    # === LEGITIMATE FEATURES (job-relevant merit indicators) ===
    # Each merit feature = 0.7 * quality_factor + 0.3 * idiosyncratic noise.
    # This induces feature correlations consistent with real candidate data
    # (strong candidates tend to score high on all dimensions).

    years_experience = (
        candidate_quality * 0.70 + rng.beta(2, 2, n) * 0.30
    ) * 25                                         # 0-25 years
    years_experience = years_experience.clip(0, 25)

    num_skills_matched = (
        candidate_quality * 0.70 + rng.beta(2, 2, n) * 0.30
    )                                              # 0-1 (fraction of job-spec skills)

    # Education: driven by quality but capped at PhD (ordinal 0-3)
    edu_latent = candidate_quality * 0.70 + rng.beta(2, 2, n) * 0.30
    education_level = (
        (edu_latent > 0.30).astype(int)            # >= BSc
        + (edu_latent > 0.60).astype(int)          # >= MSc
        + (edu_latent > 0.85).astype(int)          # PhD
    )                                              # 0=none, 1=BSc, 2=MSc, 3=PhD

    portfolio_score = (
        candidate_quality * 0.70 + rng.beta(2, 2, n) * 0.30
    )                                              # 0-1

    interview_score = (
        candidate_quality * 0.70 + rng.beta(2, 2, n) * 0.30
    )                                              # 0-1

    # === PROXY FEATURES (gender-correlated, should NOT drive selection) ===
    # All continuous scores (0-1), aggregated from multiple CV signals.

    # technical_keyword_score: density of technical terms that correlate with
    # male-coded language in historical training data ("executed", "captured",
    # sports references). Continuous; male CVs score higher on average.
    technical_keyword_score = (
        rng.beta(7, 3, n) * (1 - gender)          # male: Beta(7,3) mean≈0.70
        + rng.beta(5, 5, n) * gender               # female: Beta(5,5) mean≈0.50
    )

    # sports_captain_flag: extracurricular leadership score derived from
    # sports captaincy, athletic titles, club presidency. Male-coded in data.
    sports_captain_flag = (
        rng.beta(5, 4, n) * (1 - gender)          # male: mean≈0.56
        + rng.beta(3, 6, n) * gender               # female: mean≈0.33
    )

    # all_women_college: score indicating affinity with women's-only institutions.
    # High for female candidates; near-zero for male. Penalised in biased data.
    all_women_college = (
        rng.beta(2, 9, n) * (1 - gender)          # male: mean≈0.18 (noise)
        + rng.beta(5, 4, n) * gender               # female: mean≈0.56
    )

    # referral_from_male_employee: composite score capturing endorsement strength
    # from current male employees. Network homophily: male employees refer and
    # endorse male candidates more strongly.
    referral_from_male_employee = (
        rng.beta(6, 3, n) * (1 - gender)          # male: mean≈0.67
        + rng.beta(3, 6, n) * gender               # female: mean≈0.33
    )

    # === GROUND TRUTH LABEL ===
    # Merit signal (~75% of decision basis — job-relevant features)
    merit_signal = (
        0.20 * (years_experience / 25)
        + 0.18 * num_skills_matched
        + 0.12 * (education_level / 3)
        + 0.15 * portfolio_score
        + 0.10 * interview_score
    )
    # Sum of weights: 0.75. Range approx [0, 0.75] with substantial spread
    # due to candidate_quality factor inducing correlated extremes.

    # Proxy signal: coded at ~25% overall.
    # Male-coded patterns are historically favoured (positive contribution);
    # all_women_college is penalised (negative contribution).
    proxy_signal = (
        0.12 * technical_keyword_score
        + 0.06 * sports_captain_flag
        + 0.05 * referral_from_male_employee
        - 0.02 * all_women_college
    )
    # Sum of absolute weights: 0.25. Effective proxy ≈ 0.21-0.25 net.

    combined = merit_signal + proxy_signal
    noise = rng.normal(0, 0.06, n)
    # Threshold 0.52 calibrated to ~50% shortlist rate
    prob_select = 1 / (1 + np.exp(-(combined + noise - 0.52) * 8))

    label = (rng.random(n) < prob_select).astype(int)  # 1 = shortlisted
    margin = np.abs(prob_select - 0.5)

    df = pd.DataFrame({
        # Legitimate features
        "years_experience":             years_experience,
        "num_skills_matched":           num_skills_matched,
        "education_level":              education_level,
        "portfolio_score":              portfolio_score,
        "interview_score":              interview_score,
        # Proxy features (gender-correlated)
        "technical_keyword_score":      technical_keyword_score,
        "sports_captain_flag":          sports_captain_flag,
        "all_women_college":            all_women_college,
        "referral_from_male_employee":  referral_from_male_employee,
        # Label and metadata
        "label":                        label,
        "gender":                       gender,            # latent (metadata only)
        "candidate_quality":            candidate_quality, # latent (metadata only)
        "margin_to_threshold":          margin,
    })
    return df


# Feature configuration for the instrumentation protocol.
# NOTE: 'gender', 'candidate_quality', 'margin_to_threshold' are metadata.
FEATURE_NAMES = [
    "years_experience", "num_skills_matched", "education_level",
    "portfolio_score", "interview_score",
    "technical_keyword_score", "sports_captain_flag",
    "all_women_college", "referral_from_male_employee",
]

# Domain-legitimate features (job-relevant merit indicators)
G_DOMAIN = [0, 1, 2, 3, 4]

# Proxy features (gender-correlated, should not drive selection)
G_PROXY = [5, 6, 7, 8]

CODED_EFFECTS = {
    "proxy_attribution_overall": 0.25,
    "proxy_dominated_near_threshold_pct": 0.20,
    "annex_iii_category": "cat_4",
    "description": (
        "Gender-correlated proxy features (technical_keyword_score, "
        "sports_captain_flag, all_women_college, referral_from_male_employee) "
        "are coded to contribute ~25% of selection attribution overall. "
        "~20% of near-threshold selections are proxy-dominated (>30% proxy). "
        "A shared candidate_quality latent factor correlates merit features, "
        "creating realistic extreme-case spread for instrument calibration."
    ),
}


if __name__ == "__main__":
    out_dir = Path(__file__).resolve().parent / "data"
    out_dir.mkdir(exist_ok=True)
    df = generate_hiring_dataset()
    df.to_csv(out_dir / "amazon_hiring.csv", index=False)
    print(f"Generated {len(df)} samples -> {out_dir / 'amazon_hiring.csv'}")
    print(f"  Shortlist rate (overall):     {df['label'].mean():.1%}")
    print(f"  Shortlist rate (male):        {df[df['gender']==0]['label'].mean():.1%}")
    print(f"  Shortlist rate (female):      {df[df['gender']==1]['label'].mean():.1%}")
    print(f"  Near-threshold (margin<0.15): {(df['margin_to_threshold']<0.15).sum()}")
    print(f"  Far-threshold  (margin>0.35): {(df['margin_to_threshold']>0.35).sum()}")
