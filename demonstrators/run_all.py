"""Run all three demonstrators end-to-end.

For each demonstrator:
1. Generate synthetic dataset (with coded effects)
2. Train GradientBoosting classifier
3. Apply instrumentation protocol (Algorithm 1)
4. Run Mann-Whitney U discrimination test
5. Report whether instrument recovers coded structure

This script validates instrument calibration, not deployment findings.
Every percentage was coded into the synthetic design; the question is
whether the telemetry reads back the structure we built in.
"""

import sys
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from instrumentation.protocol import InstrumentationProtocol

# Import demonstrator configs
from cat2_cobot_safety.brake_disc.generate_dataset import (
    generate_brake_disc_dataset,
    FEATURE_NAMES as BD_FEATURES,
    G_DOMAIN as BD_G_DOMAIN,
    G_PROXY as BD_G_PROXY,
    CODED_EFFECTS as BD_CODED,
)
from cat2_cobot_safety.generate_dataset import (
    generate_cobot_dataset,
    FEATURE_NAMES as CB_FEATURES,
    G_DOMAIN as CB_G_DOMAIN,
    G_PROXY as CB_G_PROXY,
    CODED_EFFECTS as CB_CODED,
)
from _archive.energy_scheduling.generate_dataset import (
    generate_energy_dataset,
    FEATURE_NAMES as EN_FEATURES,
    G_DOMAIN as EN_G_DOMAIN,
    G_PROXY as EN_G_PROXY,
    CODED_EFFECTS as EN_CODED,
)


def train_model(X_train, y_train):
    """Train GradientBoosting classifier. After this, only predict_proba is used."""
    model = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model


def run_demonstrator(name, df, feature_names, g_domain, g_proxy, coded_effects, n_cases=30):
    """Run one demonstrator: train, instrument, test discrimination."""
    print(f"\n{'='*70}")
    print(f"  DEMONSTRATOR: {name}")
    print(f"  Coded effects: {coded_effects['description']}")
    print(f"{'='*70}\n")

    feature_cols = feature_names
    X = df[feature_cols].values
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Step 2: Train model (after this, ONLY predict_proba is used)
    model = train_model(X_train, y_train)
    acc = model.score(X_test, y_test)
    print(f"  Model accuracy: {acc:.3f}")

    # Identify near-threshold and far-from-threshold cases
    scores = model.predict_proba(X_test)[:, 1]
    margins = np.abs(scores - 0.5)
    near_mask = margins < 0.15
    far_mask = margins > 0.35

    near_idx = np.where(near_mask)[0]
    far_idx = np.where(far_mask)[0]
    print(f"  Near-threshold cases: {len(near_idx)}")
    print(f"  Far-from-threshold cases: {len(far_idx)}")

    # Step 3: Apply instrumentation protocol
    X_val = X_train  # validation set for envelope + background
    protocol = InstrumentationProtocol(
        model=model,
        X_val=X_val,
        feature_names=feature_names,
        g_domain=g_domain,
        g_proxy=g_proxy,
        threshold=0.5,
    )

    # Instrument a sample of near and far cases
    n_near = min(n_cases, len(near_idx))
    n_far = min(n_cases, len(far_idx))

    near_snippets = []
    far_snippets = []

    print(f"\n  Instrumenting {n_near} near-threshold cases...")
    for i, idx in enumerate(near_idx[:n_near]):
        snippet = protocol.generate_snippet(
            X_test[idx],
            case_id=f"{name}-near-{i:04d}",
            compute_integrity=True,
        )
        near_snippets.append(snippet)

    print(f"  Instrumenting {n_far} far-from-threshold cases...")
    for i, idx in enumerate(far_idx[:n_far]):
        snippet = protocol.generate_snippet(
            X_test[idx],
            case_id=f"{name}-far-{i:04d}",
            compute_integrity=True,
        )
        far_snippets.append(snippet)

    # Step 5: Mann-Whitney U discrimination test
    results = {}

    # Flip-rate: near vs far
    near_flips = [s.decision_robustness["flip_rate"] for s in near_snippets]
    far_flips = [s.decision_robustness["flip_rate"] for s in far_snippets]

    if len(near_flips) > 1 and len(far_flips) > 1:
        stat, p_flip = mannwhitneyu(near_flips, far_flips, alternative="greater")
        results["flip_rate"] = {
            "near_mean": float(np.mean(near_flips)),
            "far_mean": float(np.mean(far_flips)),
            "p_value": float(p_flip),
            "discriminates": p_flip < 0.01,
        }
        print(f"\n  FLIP-RATE (decision robustness):")
        print(f"    Near-threshold mean: {np.mean(near_flips):.3f}")
        print(f"    Far-from-threshold mean: {np.mean(far_flips):.3f}")
        print(f"    Mann-Whitney U p = {p_flip:.2e} {'*** DISCRIMINATES' if p_flip < 0.01 else ''}")

    # Proxy attribution: near cases
    near_proxy = [
        sum(s.constraint_enforcement["proxy_attribution"].values())
        for s in near_snippets
    ]
    far_proxy = [
        sum(s.constraint_enforcement["proxy_attribution"].values())
        for s in far_snippets
    ]
    results["proxy_attribution"] = {
        "near_mean": float(np.mean(near_proxy)),
        "far_mean": float(np.mean(far_proxy)),
    }
    print(f"\n  PROXY ATTRIBUTION (constraint enforcement):")
    print(f"    Near-threshold mean: {np.mean(near_proxy):.3f}")
    print(f"    Far-from-threshold mean: {np.mean(far_proxy):.3f}")

    # Legitimate ratio
    near_legit = [s.constraint_enforcement["legitimate_feature_ratio"] for s in near_snippets]
    far_legit = [s.constraint_enforcement["legitimate_feature_ratio"] for s in far_snippets]
    results["legitimate_ratio"] = {
        "near_mean": float(np.mean(near_legit)),
        "far_mean": float(np.mean(far_legit)),
    }
    print(f"\n  LEGITIMATE-FEATURE RATIO:")
    print(f"    Near-threshold mean: {np.mean(near_legit):.3f}")
    print(f"    Far-from-threshold mean: {np.mean(far_legit):.3f}")

    # Envelope distance
    near_env = [s.envelope_validity["distance_to_nearest_prototype"] for s in near_snippets]
    far_env = [s.envelope_validity["distance_to_nearest_prototype"] for s in far_snippets]
    results["envelope_distance"] = {
        "near_mean": float(np.mean(near_env)),
        "far_mean": float(np.mean(far_env)),
    }

    # Verdicts
    near_verdicts = [s.action_verdict for s in near_snippets]
    far_verdicts = [s.action_verdict for s in far_snippets]
    results["verdicts"] = {
        "near_flagged_pct": sum(1 for v in near_verdicts if v != "PROCEED") / len(near_verdicts),
        "far_flagged_pct": sum(1 for v in far_verdicts if v != "PROCEED") / len(far_verdicts),
    }
    print(f"\n  VERDICTS:")
    print(f"    Near-threshold flagged: {results['verdicts']['near_flagged_pct']:.1%}")
    print(f"    Far-from-threshold flagged: {results['verdicts']['far_flagged_pct']:.1%}")

    # Save example snippet
    if near_snippets:
        example = near_snippets[0]
        out_dir = Path(__file__).resolve().parent.parent / "results"
        out_dir.mkdir(exist_ok=True)
        example.save(str(out_dir / f"{name}_example_snippet.json"))
        print(f"\n  Example snippet saved -> results/{name}_example_snippet.json")

    return results


def main():
    print("=" * 70)
    print("  DECISION-BOUNDARY INSTRUMENTATION: REPLICATION SCRIPT")
    print("  Validates instrument calibration, not deployment findings.")
    print("  Every coded effect is a design parameter to be recovered.")
    print("=" * 70)

    all_results = {}

    # === BRAKE DISC ===
    df_bd = generate_brake_disc_dataset()
    all_results["brake_disc"] = run_demonstrator(
        "brake_disc", df_bd, BD_FEATURES, BD_G_DOMAIN, BD_G_PROXY, BD_CODED
    )

    # === COBOT SAFETY ===
    df_cb = generate_cobot_dataset()
    all_results["cobot_safety"] = run_demonstrator(
        "cobot_safety", df_cb, CB_FEATURES, CB_G_DOMAIN, CB_G_PROXY, CB_CODED
    )

    # === ENERGY SCHEDULING ===
    df_en = generate_energy_dataset()
    all_results["energy_scheduling"] = run_demonstrator(
        "energy_scheduling", df_en, EN_FEATURES, EN_G_DOMAIN, EN_G_PROXY, EN_CODED
    )

    # Save all results
    out_dir = Path(__file__).resolve().parent.parent / "results"
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / "replication_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n\nAll results saved -> results/replication_results.json")

    # Summary
    print("\n" + "=" * 70)
    print("  SUMMARY: Does the instrument recover the coded structure?")
    print("=" * 70)
    for name, res in all_results.items():
        flip_ok = res.get("flip_rate", {}).get("discriminates", False)
        print(f"\n  {name}:")
        print(f"    Flip-rate discriminates near vs far: {'YES' if flip_ok else 'NO'}")
        print(f"    Proxy attribution (near): {res['proxy_attribution']['near_mean']:.3f}")
        print(f"    Near-threshold flagged: {res['verdicts']['near_flagged_pct']:.1%}")


if __name__ == "__main__":
    main()
