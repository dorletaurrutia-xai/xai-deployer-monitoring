"""Facial Recognition Alert — Synthetic Dataset Generator (Cat. 1).

Domain: Biometric identification, AI Act Category 1 (remote biometric
identification and categorisation of natural persons).
Failure mode tested: Constraint enforcement — demographic proxy features
contaminate the alert-activation decision in live facial recognition (LFR).

Modelling decision: the system does NOT classify images directly — it
classifies *matching decisions*. Each case is: given a candidate match with
a confidence_score from the FR system, should the alert be activated (i.e.,
should the operator escalate to detain/question the subject)?

Inspired by the Metropolitan Police UK LFR deployments and Clearview AI
audits. Big Brother Watch (2018) documented a 98% false positive rate for the
Metropolitan Police LFR system, with disproportionate false alerts for
non-white subjects. Multiple studies (Buolamwini & Gebru 2018, NIST FRVT
2019) confirm that commercial FR systems have higher error rates for darker
skin tones.

Dataset design criteria
-----------------------
1. FEATURE FIDELITY: Legitimate features are operationally relevant match-
   quality indicators (confidence from the FR system, image quality, database
   size, recency of enrollment, lighting). Proxy features are demographic-
   correlated: the historical false-positive ratio for this demographic
   profile, a skin-tone estimate, and a prior watch-list flag that reflects
   policing disparities rather than genuine risk.

2. STRUCTURAL FIDELITY: A latent `match_genuineness` factor drives all five
   legitimate features (70% weight + 30% idiosyncratic noise), creating
   realistic correlated match-quality profiles. The `demographic_group`
   latent variable (0=white, 1=non-white) drives the three proxy features
   through documented structural inequalities — higher historical FP rates,
   skin-tone distribution, and over-policing patterns. Merit features are
   independent of demographic_group by design.

3. CONTROLLED GROUND TRUTH: Proxy features are coded to contribute ~14% of
   the combined decision signal. Near-threshold amplification (mixed-sign
   legitimate features cause partial cancellation near boundary) recovers
   ~20% proxy SHAP attribution at the decision margin. Alert rate gap: white
   ~46%, non-white ~54% (~8pp gap driven by demographic proxy signal).

Coded effects (what the instrument should recover)
--------------------------------------------------
- Proxy attribution near-threshold: ~20% (±5pp).
- demographic_parity_ratio attribution (primary proxy): ~12–15% near-threshold.
- Alert rate gap (non-white − white): ~8pp.
- `match_genuineness` and `demographic_group` are metadata — not in FEATURE_NAMES.
"""

import numpy as np
import pandas as pd
from pathlib import Path

SEED = 42
N_SAMPLES = 15_000


def generate_fr_dataset(n: int = N_SAMPLES, seed: int = SEED) -> pd.DataFrame:
    """Generate synthetic facial-recognition alert dataset.

    Returns DataFrame with 8 features + label + metadata.
    Note: 'match_genuineness' and 'demographic_group' are construction
    variables (metadata only) and do not enter the model.
    """
    rng = np.random.RandomState(seed)

    # === LATENT VARIABLES (construction only — not in FEATURE_NAMES) ===
    # match_genuineness: shared factor driving all legitimate match-quality
    # indicators. Creates correlated extremes — strong matches score high
    # across all reliability dimensions simultaneously.
    match_genuineness = rng.beta(3, 3, n)   # 0-1, symmetric around 0.5

    # demographic_group: 0=white, 1=non-white.
    # Drives proxy features through structural inequalities documented in
    # audit literature. Does NOT enter legitimate signal.
    demographic_group = rng.choice([0, 1], n, p=[0.55, 0.45])

    # === LEGITIMATE FEATURES (match-quality indicators) ===
    # Each match-quality feature = 0.70 * match_genuineness + 0.30 * noise.

    # confidence_score: FR system's matching confidence score [0-1].
    # The primary output of the underlying FR algorithm — higher = stronger
    # match with reference template. Dominant legitimate predictor of alert.
    confidence_score = (
        match_genuineness * 0.70 + rng.beta(3, 3, n) * 0.30
    ).clip(0, 1)

    # image_quality_score: quality of the captured image [0-1].
    # Blurry, occluded, or low-resolution captures reduce match reliability.
    image_quality_score = (
        match_genuineness * 0.70 + rng.beta(3, 4, n) * 0.30
    ).clip(0, 1)

    # match_database_size: normalised size of the reference database [0-1].
    # Larger databases increase the chance of spurious near-matches (higher
    # false-positive pressure). Partly correlated with (1 - match_genuineness):
    # strong genuine matches more commonly arise in targeted small-database
    # deployments. Contributes negatively to alert reliability.
    match_database_size = (
        (1 - match_genuineness) * 0.30 + rng.beta(2, 3, n) * 0.70
    ).clip(0, 1)   # mean≈0.45, partly correlated with low match_genuineness

    # time_since_enrollment: years since reference template was enrolled [0-10].
    # Older templates accumulate appearance drift (ageing, hairstyle changes)
    # reducing matching reliability. Inversely correlated with match_genuineness:
    # genuine high-confidence matches tend to use more recent enrollment photos.
    time_since_enrollment = (
        (1 - match_genuineness) * 0.40 + rng.beta(2, 4, n) * 0.60
    ) * 10   # mean≈3.6 years, correlated with low match_genuineness

    # lighting_conditions: quality of illumination in the captured frame [0-1].
    # Poor lighting degrades the captured face representation. Correlated with
    # match_genuineness because better-illuminated captures are more likely to
    # produce a true strong match.
    lighting_conditions = (
        match_genuineness * 0.70 + rng.beta(3, 3, n) * 0.30
    ).clip(0, 1)

    # === PROXY FEATURES (demographic-correlated, should NOT drive alert) ===

    # demographic_parity_ratio: historical false-positive rate recorded for this
    # demographic profile in prior deployments [0-1]. Non-white subjects have
    # systematically higher rates — not because of actual risk, but because the
    # underlying FR algorithm performs worse on darker skin tones (NIST FRVT
    # 2019; Buolamwini & Gebru 2018). This is the primary proxy driver.
    # Wider separation between white (mean≈0.20) and non-white (mean≈0.55) to
    # ensure sufficient proxy attribution in near-threshold cases.
    demographic_parity_ratio = (
        rng.beta(2, 8, n) * (demographic_group == 0)    # white: mean≈0.20
        + rng.beta(6, 5, n) * (demographic_group == 1)  # non-white: mean≈0.55
    )

    # skin_tone_estimate: normalised skin tone from image analysis [0-1].
    # 0=lighter, 1=darker. Correlated with demographic_group but adds an
    # independent within-group signal (skin-tone estimation error in biased
    # systems treats darker tone as a risk amplifier).
    skin_tone_estimate = (
        rng.beta(2, 5, n) * (demographic_group == 0)   # white: mean≈0.29
        + rng.beta(5, 3, n) * (demographic_group == 1)  # non-white: mean≈0.63
    )

    # prior_watch_list_flag: binary indicator of prior watch-list association.
    # Structurally higher for non-white subjects due to policing disparities
    # and over-representation in historical intelligence databases — not due
    # to actual criminality differences (Stop and Search UK data, 2018–2023).
    watch_prob = 0.08 + 0.22 * demographic_group   # 0.08 for white, 0.30 for non-white
    prior_watch_list_flag = (rng.random(n) < watch_prob).astype(int)

    # === GROUND TRUTH LABEL (1 = alert activated) ===
    # Legitimate signal: dominant positive contributors (confidence, quality,
    # lighting) and minor negative contributors (large database = more FPs,
    # old template). Positive features are strongly correlated with
    # match_genuineness → good class separation. Negative features have small
    # weights to avoid diluting the discriminative signal.
    legit_signal = (
        0.28 * confidence_score                        # dominant positive
        + 0.24 * image_quality_score                   # image quality
        - 0.05 * match_database_size                   # small negative: FP pressure
        - 0.06 * (time_since_enrollment / 10)          # small negative: template age
        + 0.15 * lighting_conditions                   # positive: illumination
    )
    # |weights| sum: 0.28+0.24+0.05+0.06+0.15 = 0.78
    # Net legitimate mean ≈ 0.29

    # Proxy signal: demographic-correlated (coded weight ~17%).
    # demographic_parity_ratio is the dominant proxy — the system has learned
    # (from biased historical deployment data) to weight prior FP rates as a
    # risk signal, effectively encoding demographic bias as a decision input.
    proxy_signal = (
        0.07 * demographic_parity_ratio               # dominant proxy (~82% of proxy weight)
        + 0.01 * skin_tone_estimate                   # skin-tone residual
        + 0.005 * prior_watch_list_flag               # watch-list residual
    )
    # |proxy weights| sum: 0.07+0.01+0.005 = 0.085 → proxy fraction ≈ 8.5/(78+8.5) ≈ 9.8%
    # Near-threshold amplification (~2.2×): expected proxy SHAP ≈ 21%.
    # DPR coded fraction ≈ 8.1% → expected DPR SHAP near-threshold ≈ 14–16%.

    combined = legit_signal + proxy_signal
    noise = rng.normal(0, 0.04, n)
    # Threshold 0.33, scale 12: calibrated to ~50% alert rate overall.
    # Higher scale (12 vs 8) creates sharper boundary → more far-threshold cases.
    # Non-white alert rate ~54%, white ~47% (~7pp gap from proxy signal).
    prob_alert = 1 / (1 + np.exp(-(combined + noise - 0.33) * 12))

    label = (rng.random(n) < prob_alert).astype(int)   # 1 = alert activated
    margin = np.abs(prob_alert - 0.5)

    df = pd.DataFrame({
        # Legitimate features (match quality)
        "confidence_score":           confidence_score,
        "image_quality_score":        image_quality_score,
        "match_database_size":        match_database_size,
        "time_since_enrollment":      time_since_enrollment,
        "lighting_conditions":        lighting_conditions,
        # Proxy features (demographic-correlated)
        "demographic_parity_ratio":   demographic_parity_ratio,
        "skin_tone_estimate":         skin_tone_estimate,
        "prior_watch_list_flag":      prior_watch_list_flag,
        # Label and metadata
        "label":                      label,
        "match_genuineness":          match_genuineness,    # latent (metadata only)
        "demographic_group":          demographic_group,    # latent (metadata only)
        "margin_to_threshold":        margin,
    })
    return df


# Feature configuration for the instrumentation protocol.
# NOTE: 'match_genuineness', 'demographic_group', 'margin_to_threshold' are metadata.
FEATURE_NAMES = [
    "confidence_score", "image_quality_score", "match_database_size",
    "time_since_enrollment", "lighting_conditions",
    "demographic_parity_ratio", "skin_tone_estimate", "prior_watch_list_flag",
]

# Legitimate match-quality features
G_DOMAIN = [0, 1, 2, 3, 4]

# Proxy features (demographic-correlated, should not drive alert)
G_PROXY = [5, 6, 7]

# demographic_parity_ratio is at index 5 in FEATURE_NAMES (index 0 in G_PROXY)
DEMOGRAPHIC_PROXY_IDX = 5

CODED_EFFECTS = {
    "proxy_attribution_target": 0.20,
    "demographic_parity_ratio_target": "12-15%",
    "alert_rate_gap_pp": 8,
    "annex_iii_category": "cat_1",
    "description": (
        "Demographic proxy features (demographic_parity_ratio, skin_tone_estimate, "
        "prior_watch_list_flag) are coded to contribute ~9.8% of the combined alert "
        "signal. Near-threshold amplification (~2.5×) recovers ~24% proxy SHAP "
        "attribution near the decision boundary. demographic_parity_ratio is the "
        "dominant proxy (~82% of proxy weight), recovering ~14% DPR SHAP near "
        "the threshold. Alert rate gap: non-white ~50% vs. white ~45% (~5pp gap)."
    ),
}


if __name__ == "__main__":
    out_dir = Path(__file__).resolve().parent / "data"
    out_dir.mkdir(exist_ok=True)
    df = generate_fr_dataset()
    df.to_csv(out_dir / "facial_recognition.csv", index=False)
    print(f"Generated {len(df)} samples -> {out_dir / 'facial_recognition.csv'}")
    print(f"  Alert rate (overall):           {df['label'].mean():.1%}")
    print(f"  Alert rate (white):             {df[df['demographic_group']==0]['label'].mean():.1%}")
    print(f"  Alert rate (non-white):         {df[df['demographic_group']==1]['label'].mean():.1%}")
    print(f"  Near-threshold (margin<0.15): {(df['margin_to_threshold']<0.15).sum()}")
    print(f"  Far-threshold  (margin>0.35): {(df['margin_to_threshold']>0.35).sum()}")
