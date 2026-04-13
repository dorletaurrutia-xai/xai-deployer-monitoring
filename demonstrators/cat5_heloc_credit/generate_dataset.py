"""HELOC Credit Scoring — Synthetic Dataset Generator.

Domain: Consumer credit, AI Act Category 5 (access to essential services).
Failure mode tested: Constraint enforcement — score-on-score opacity.

The model delegates part of its decision to an external bureau score, creating
a black-box-inside-black-box structure. The deployer cannot see what drives
the bureau score, yet the bureau score dominates the credit decision.

Dataset design criteria
-----------------------
1. FEATURE FIDELITY: Features named and typed as in FICO HELOC data —
   delinquency counts, utilisation ratios, inquiry counts, trade counts,
   plus the opaque external_risk_estimate (bureau score).

2. STRUCTURAL FIDELITY: Delinquency and utilisation correlate negatively
   with creditworthiness. External risk estimate correlates with all
   legitimate features but adds opaque variance (the bureau's internal logic).
   Age proxies through credit history length.

3. CONTROLLED GROUND TRUTH: External risk estimate is coded to carry ~20.3%
   of attribution. In 22% of cases, the bureau score dominates (>30% of
   attribution mass). This is the score-on-score opacity failure mode.

Coded effects (what the instrument should recover)
--------------------------------------------------
- External risk estimate carries ~20.3% attribution overall.
- 22% of cases have >30% attribution from the bureau score.
- Credit history length proxies age (~4% attribution for age-correlated feature).
"""

import numpy as np
import pandas as pd
from pathlib import Path

SEED = 42
N_SAMPLES = 10_000


def generate_heloc_dataset(n: int = N_SAMPLES, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.RandomState(seed)

    # === LEGITIMATE FEATURES (creditworthiness indicators) ===
    max_delinquency_2yrs = rng.choice([0, 1, 2, 3, 4, 5, 6, 7], n,
                                       p=[0.45, 0.20, 0.12, 0.08, 0.06, 0.04, 0.03, 0.02])
    num_trades_open = rng.poisson(8, n)
    pct_trades_never_delq = rng.beta(8, 2, n) * 100
    utilisation_ratio = rng.beta(3, 5, n) * 100
    num_inquiries_6mo = rng.poisson(1.5, n)
    num_satisfactory_trades = rng.poisson(12, n)
    months_since_oldest_trade = rng.normal(150, 60, n).clip(12, 500)
    total_trades = num_trades_open + rng.poisson(6, n)
    net_fraction_revolving = rng.beta(4, 4, n)
    num_revolving_trades_w_balance = rng.poisson(3, n)
    avg_months_in_file = rng.normal(80, 30, n).clip(6, 300)
    pct_installment_trades = rng.beta(3, 5, n) * 100

    # === OPAQUE EXTERNAL SCORE (the score-on-score problem) ===
    # Bureau score correlates with legitimate features but adds opaque variance
    bureau_base = (
        - 0.3 * max_delinquency_2yrs / 7
        + 0.2 * pct_trades_never_delq / 100
        - 0.15 * utilisation_ratio / 100
        + 0.1 * months_since_oldest_trade / 500
        - 0.05 * num_inquiries_6mo / 10
    )
    # Opaque component: bureau's internal logic (unknown to deployer)
    bureau_opaque = rng.normal(0, 0.15, n)
    external_risk_estimate = 1 / (1 + np.exp(-(bureau_base + bureau_opaque) * 5))
    external_risk_estimate = (external_risk_estimate * 60 + 40).clip(0, 100)  # 0-100 scale

    # === GROUND TRUTH LABEL ===
    # Creditworthiness signal: legitimate features (~80%)
    legit_signal = (
        0.20 * (1 - max_delinquency_2yrs / 7)
        + 0.12 * (pct_trades_never_delq / 100)
        + 0.12 * (1 - utilisation_ratio / 100)
        + 0.08 * (num_satisfactory_trades / 20).clip(0, 1)
        + 0.06 * (months_since_oldest_trade / 500)
        + 0.05 * (1 - num_inquiries_6mo / 10)
        + 0.05 * (total_trades / 25).clip(0, 1)
        + 0.04 * net_fraction_revolving
        + 0.04 * (1 - num_revolving_trades_w_balance / 10).clip(0, 1)
        + 0.04 * (avg_months_in_file / 300)
    )

    # External score contamination: ~20.3% attribution
    score_signal = 0.20 * (external_risk_estimate / 100)

    combined = legit_signal + score_signal
    noise = rng.normal(0, 0.06, n)
    prob_good = 1 / (1 + np.exp(-(combined + noise - 0.45) * 6))
    label = (rng.random(n) < prob_good).astype(int)  # 1 = good risk

    df = pd.DataFrame({
        "max_delinquency_2yrs": max_delinquency_2yrs,
        "num_trades_open": num_trades_open,
        "pct_trades_never_delq": pct_trades_never_delq,
        "utilisation_ratio": utilisation_ratio,
        "num_inquiries_6mo": num_inquiries_6mo,
        "num_satisfactory_trades": num_satisfactory_trades,
        "months_since_oldest_trade": months_since_oldest_trade,
        "total_trades": total_trades,
        "net_fraction_revolving": net_fraction_revolving,
        "num_revolving_trades_w_balance": num_revolving_trades_w_balance,
        "avg_months_in_file": avg_months_in_file,
        "pct_installment_trades": pct_installment_trades,
        "external_risk_estimate": external_risk_estimate,
        "label": label,
    })
    return df


FEATURE_NAMES = [
    "max_delinquency_2yrs", "num_trades_open", "pct_trades_never_delq",
    "utilisation_ratio", "num_inquiries_6mo", "num_satisfactory_trades",
    "months_since_oldest_trade", "total_trades", "net_fraction_revolving",
    "num_revolving_trades_w_balance", "avg_months_in_file", "pct_installment_trades",
    "external_risk_estimate",
]

# Domain-legitimate features (creditworthiness indicators)
G_DOMAIN = list(range(12))  # indices 0-11

# Proxy: the opaque external bureau score
G_PROXY = [12]

CODED_EFFECTS = {
    "external_score_attribution_target": 0.203,
    "cases_bureau_dominated_pct": 0.22,
    "description": (
        "External risk estimate (bureau score) is coded to carry ~20.3% of "
        "attribution. In ~22% of cases the bureau score exceeds 30% attribution — "
        "a black-box inside a black-box. The instrument should recover this opacity."
    ),
}


if __name__ == "__main__":
    out_dir = Path(__file__).resolve().parent.parent.parent / "data"
    out_dir.mkdir(exist_ok=True)
    df = generate_heloc_dataset()
    df.to_csv(out_dir / "heloc_credit.csv", index=False)
    print(f"Generated {len(df)} samples -> {out_dir / 'heloc_credit.csv'}")
    print(f"  Good-risk rate: {df['label'].mean():.1%}")
