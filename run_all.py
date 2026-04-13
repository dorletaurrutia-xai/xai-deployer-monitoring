"""Root-level entry point: run all eight demonstrators.

Usage (from repo root):
    python run_all.py            # all 8 demonstrators
    python run_all.py --full     # also runs core protocol replication (cat2 legacy)
"""

import sys
import argparse
from pathlib import Path

# ── path setup ────────────────────────────────────────────────────────────────
THIS_DIR = Path(__file__).resolve().parent
DEMO_DIR = THIS_DIR / "demonstrators"

sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(DEMO_DIR))

# ── Cat. 1, 4, 7, 8 — SimpleLR + KernelSHAP ──────────────────────────────────
from run_new import main as run_new_main

# ── Cat. 3, 5, 6 — GradientBoostingClassifier + InstrumentationProtocol ──────
from cat3_oulad_education.run_analysis import run_analysis as run_oulad
from cat5_heloc_credit.run_analysis import run_analysis as run_heloc
from cat6_compas_pretrial.run_analysis import run_analysis as run_compas


def _print_summary(label: str, res: dict) -> None:
    print(f"\n  {label}")
    print(f"  Accuracy: {res['accuracy']:.3f} | near: {res['n_near']} | far: {res['n_far']}")
    print(f"  Flip p={res.get('flip_mwu_p', 1):.2e}  "
          f"Flagged near={res.get('near_flagged_pct', 0):.1%}  "
          f"far={res.get('far_flagged_pct', 0):.1%}")


def main():
    parser = argparse.ArgumentParser(description="Run all eight AI Act demonstrators.")
    parser.add_argument(
        "--full", action="store_true",
        help="Also run core protocol replication (cat2 brake_disc, cat2 cobot, "
             "energy_scheduling). Requires scikit-learn.",
    )
    parser.add_argument("--n_cases", type=int, default=40,
                        help="Cases per near/far partition (default: 40)")
    args = parser.parse_args()

    # Cat. 1, 4, 7, 8
    run_new_main(n_cases=args.n_cases)

    # Cat. 3, 5, 6
    print("\n" + "=" * 70)
    print("  Cat. 3 / 5 / 6 — GradientBoostingClassifier + InstrumentationProtocol")
    print("=" * 70)

    res3 = run_oulad(n_cases=args.n_cases)
    _print_summary("Cat. 3 — OULAD Education (deprivation_disability_proxy)", res3)

    res5 = run_heloc(n_cases=args.n_cases)
    _print_summary("Cat. 5 — HELOC Credit (score_on_score_opacity)", res5)

    res6 = run_compas(n_cases=args.n_cases)
    _print_summary("Cat. 6 — COMPAS Pretrial (non_criminal_history_contamination)", res6)

    if args.full:
        print("\n" + "=" * 70)
        print("  CORE PROTOCOL REPLICATION (Cat. 2 / archived energy)")
        print("  Requires scikit-learn.")
        print("=" * 70)
        try:
            from run_all import main as run_all_main  # demonstrators/run_all.py
            run_all_main()
        except ImportError as exc:
            print(f"\n  [SKIP] scikit-learn not available: {exc}")
            print("  Install with: pip install scikit-learn")


if __name__ == "__main__":
    main()
