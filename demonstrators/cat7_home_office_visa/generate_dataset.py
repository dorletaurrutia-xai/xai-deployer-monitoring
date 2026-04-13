"""Home Office Visa Streaming — Synthetic Dataset Generator (Cat. 7).

Domain: Immigration/visa processing, AI Act Category 7 (migration, asylum,
border management).
Failure mode tested: Constraint enforcement — nationality_proxy_streaming.
An opaque nationality risk score dominates the visa streaming decision,
creating a black-box-inside-black-box structure analogous to HELOC's
external bureau score.

Inspired by the Home Office UK algorithmic visa streaming tool (2015–2020),
which classified visa applications into red/amber/green risk streams.
The Foxglove / JCWI 2020 audit revealed that nationality was the dominant
factor, systematically downgrading applicants from 'high-risk' countries
regardless of individual merit. The Home Office withdrew it in August 2020.

The black-box-inside-a-black-box structure:
  The deployer (Home Office caseworker) uses a visa approval model that
  incorporates `nationality_risk_score` — itself a legacy streaming score
  whose internal logic is not visible to the current system's operators.
  The instrument surfaces how much weight this opaque sub-score carries,
  making the delegation auditable without requiring access to its internals.

Dataset design criteria
-----------------------
1. FEATURE FIDELITY: Legitimate features are standard visa eligibility
   criteria (residence, income, sponsorship, compliance history, travel
   purpose). Proxy features are nationality-correlated: the opaque streaming
   score, the country risk tier, and a travel complexity flag that is
   structurally higher for applicants from high-tier countries.

2. STRUCTURAL FIDELITY: A latent `applicant_legitimacy` factor drives all
   five legitimate features (70% weight + 30% idiosyncratic noise), creating
   realistic correlated applicant profiles. The `nationality_risk_score` is
   generated from `country_of_origin_tier` (dominant, 45% weight) plus a
   partial legitimacy component (25% weight) plus opaque variance — mirroring
   how the real streaming tool embedded nationality into an otherwise
   quality-correlated score.

3. CONTROLLED GROUND TRUTH: nationality_risk_score is coded to carry ~25%
   of the combined decision signal. With 5 legitimate features (vs. 12 in
   HELOC), the proxy faces less competition for attribution mass; near-
   threshold cases show ~28–32% proxy attribution, matching the target.

Coded effects (what the instrument should recover)
--------------------------------------------------
- nationality_risk_score attribution near-threshold: ~28–32%.
- Approval rate gap: tier-0 (safe) ~65%, tier-2 (high-risk) ~35% (~30pp gap).
- `applicant_legitimacy` is metadata — not in FEATURE_NAMES.
"""

import numpy as np
import pandas as pd
from pathlib import Path

SEED = 42
N_SAMPLES = 15_000


def generate_visa_dataset(n: int = N_SAMPLES, seed: int = SEED) -> pd.DataFrame:
    """Generate synthetic visa-streaming dataset.

    Returns DataFrame with 8 features + label + metadata.
    Note: 'applicant_legitimacy' is a construction variable (metadata only)
    and does not enter the model.
    """
    rng = np.random.RandomState(seed)

    # === LATENT VARIABLES (construction only — not in FEATURE_NAMES) ===
    # applicant_legitimacy: shared factor driving all legitimate visa criteria.
    # Creates correlated extremes — strong applicants score high across all
    # legitimate dimensions, weak applicants score low.
    applicant_legitimacy = rng.beta(3, 3, n)   # 0-1, symmetric around 0.5

    # country_of_origin_tier: 0=low-risk, 1=medium-risk, 2=high-risk.
    # Monotonic ordinal encoding so a linear model can learn a single
    # positive coefficient. Drives nationality_risk_score and travel_history_flag.
    # Distribution reflects roughly the Home Office's tier distribution.
    country_of_origin_tier = rng.choice([0, 1, 2], n, p=[0.45, 0.30, 0.25])

    # === LEGITIMATE FEATURES (visa eligibility criteria) ===
    # Each feature = 0.70 * applicant_legitimacy + 0.30 * idiosyncratic noise.

    # years_in_country: years of documented residence/visits (0–20)
    years_in_country = (
        applicant_legitimacy * 0.70 + rng.beta(2, 3, n) * 0.30
    ) * 20
    years_in_country = years_in_country.clip(0, 20)

    # income_level: normalised income relative to application threshold (0–1)
    income_level = (
        applicant_legitimacy * 0.70 + rng.beta(3, 3, n) * 0.30
    ).clip(0, 1)

    # sponsor_type: 0=no sponsor, 1=employer sponsor, 2=family sponsor
    # Stronger sponsors (2=family) are typical of better-established applicants.
    spon_latent = applicant_legitimacy * 0.70 + rng.beta(3, 4, n) * 0.30
    sponsor_type = (
        (spon_latent > 0.35).astype(int)
        + (spon_latent > 0.70).astype(int)
    )  # 0=none, 1=employer, 2=family

    # prior_visa_compliance: track record of previous visa compliance (0–1)
    prior_visa_compliance = (
        applicant_legitimacy * 0.70 + rng.beta(4, 3, n) * 0.30
    ).clip(0, 1)

    # purpose_of_travel: 0=tourism, 1=business, 2=study, 3=family reunion
    # Higher purposes correlate with legitimate long-term ties.
    purp_latent = applicant_legitimacy * 0.70 + rng.beta(3, 3, n) * 0.30
    purpose_of_travel = (
        (purp_latent > 0.25).astype(int)
        + (purp_latent > 0.50).astype(int)
        + (purp_latent > 0.75).astype(int)
    )  # 0=tourism, 1=business, 2=study, 3=family

    # === PROXY FEATURES (nationality-correlated, should NOT drive approval) ===

    # nationality_risk_score: opaque legacy streaming score (0–100, higher=more risk).
    # Partially driven by nationality tier and legitimacy, plus a large opaque component
    # (the legacy system's internal logic). The deployer cannot decompose this score.
    # This is the black-box-inside-a-black-box: increased opaque variance ensures the
    # model must rely on the score directly rather than reconstructing it from tier.
    nat_base = (
        - 0.20 * (applicant_legitimacy - 0.5)   # higher legitimacy → slightly lower risk
        + 0.25 * country_of_origin_tier / 2      # tier partially drives score (reduced)
    )
    nat_opaque = rng.normal(0, 0.22, n)          # larger opaque variance → less reconstructable
    nationality_risk_score = (
        1 / (1 + np.exp(-(nat_base + nat_opaque) * 4)) * 70 + 15
    ).clip(5, 95)                                # range ≈ [15, 85], mean ≈ 50-55

    # country_of_origin_tier: directly-observable tier classification (0–2).
    # Monotonic in proxy contribution — lower tier = less risk, higher = more.
    # (already constructed above as latent variable; included as a model feature)

    # travel_history_flag: flag for complex/multi-country travel history.
    # Structurally higher for applicants from high-tier countries due to
    # documentation requirements and prior visa refusals.
    travel_prob = 0.10 + 0.25 * (country_of_origin_tier / 2)
    travel_history_flag = (rng.random(n) < travel_prob).astype(int)

    # === GROUND TRUTH LABEL (1 = approved) ===
    # Legitimate signal: positive contribution to approval probability (~70% basis)
    legit_signal = (
        0.20 * (years_in_country / 20)
        + 0.15 * income_level
        + 0.12 * (sponsor_type / 2)
        + 0.13 * prior_visa_compliance
        + 0.10 * (purpose_of_travel / 3)
    )
    # Sum of weights: 0.70. High-legitimacy applicants score up to 0.70.

    # Proxy risk signal: negative contribution (higher risk → lower approval).
    # nationality_risk_score is the SOLE main proxy driver (~30% coded weight),
    # following the HELOC score-on-score pattern: the opaque score carries the
    # nationality signal; tier and travel have only tiny residual direct effects.
    proxy_signal = (
        0.30 * (nationality_risk_score / 100)    # dominant opaque proxy (~30% basis)
        + 0.005 * (country_of_origin_tier / 2)   # residual direct effect (visible in proxy_detail)
        + 0.005 * travel_history_flag             # residual direct effect
    )
    # Effective proxy contribution: 0.30 × mean(nrs/100) ≈ 0.165 at population mean.

    # Combined approval signal: legitimacy minus risk
    # proxy_signal is dominated by nationality_risk_score (sole coded driver, ~30%).
    # country_of_origin_tier and travel_history_flag have tiny direct contributions —
    # they act as observable correlates of nationality_risk_score in the model.
    combined = legit_signal - proxy_signal
    noise = rng.normal(0, 0.05, n)
    # Threshold 0.185, scale 7: calibrated to ~50% approval rate overall
    prob_approved = 1 / (1 + np.exp(-(combined + noise - 0.185) * 7))

    label = (rng.random(n) < prob_approved).astype(int)  # 1 = approved
    margin = np.abs(prob_approved - 0.5)

    df = pd.DataFrame({
        # Legitimate features
        "years_in_country":        years_in_country,
        "income_level":            income_level,
        "sponsor_type":            sponsor_type,
        "prior_visa_compliance":   prior_visa_compliance,
        "purpose_of_travel":       purpose_of_travel,
        # Proxy features (nationality-correlated)
        "nationality_risk_score":  nationality_risk_score,
        "country_of_origin_tier":  country_of_origin_tier,
        "travel_history_flag":     travel_history_flag,
        # Label and metadata
        "label":                   label,
        "applicant_legitimacy":    applicant_legitimacy,    # latent (metadata only)
        "margin_to_threshold":     margin,
    })
    return df


# Feature configuration for the instrumentation protocol.
# NOTE: 'applicant_legitimacy' and 'margin_to_threshold' are metadata.
FEATURE_NAMES = [
    "years_in_country", "income_level", "sponsor_type",
    "prior_visa_compliance", "purpose_of_travel",
    "nationality_risk_score", "country_of_origin_tier", "travel_history_flag",
]

# Legitimate visa eligibility features
G_DOMAIN = [0, 1, 2, 3, 4]

# Proxy features (nationality-correlated, should not drive approval)
G_PROXY = [5, 6, 7]

# nationality_risk_score is at index 5 in FEATURE_NAMES (index 0 in G_PROXY)
NRS_FEATURE_IDX = 5

CODED_EFFECTS = {
    "nationality_risk_score_attribution_target": 0.30,
    "cases_nationality_dominated_pct": 0.25,
    "annex_iii_category": "cat_7",
    "description": (
        "nationality_risk_score (opaque legacy streaming score) is coded to carry "
        "~25% of the combined approval signal, recovering to ~28-32% SHAP near "
        "the threshold. country_of_origin_tier and travel_history_flag add ~5pp. "
        "The applicant_legitimacy latent factor correlates all five legitimate "
        "features, creating realistic extreme-case spread for calibration."
    ),
}


if __name__ == "__main__":
    out_dir = Path(__file__).resolve().parent / "data"
    out_dir.mkdir(exist_ok=True)
    df = generate_visa_dataset()
    df.to_csv(out_dir / "home_office_visa.csv", index=False)
    print(f"Generated {len(df)} samples -> {out_dir / 'home_office_visa.csv'}")
    print(f"  Approval rate (overall):         {df['label'].mean():.1%}")
    print(f"  Approval rate (tier 0 — safe):   {df[df['country_of_origin_tier']==0]['label'].mean():.1%}")
    print(f"  Approval rate (tier 1 — medium): {df[df['country_of_origin_tier']==1]['label'].mean():.1%}")
    print(f"  Approval rate (tier 2 — high):   {df[df['country_of_origin_tier']==2]['label'].mean():.1%}")
    print(f"  Near-threshold (margin<0.15): {(df['margin_to_threshold']<0.15).sum()}")
    print(f"  Far-threshold  (margin>0.35): {(df['margin_to_threshold']>0.35).sum()}")
