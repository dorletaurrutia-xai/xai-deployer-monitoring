"""Run the four new Annex III demonstrators (Cat. 1, 4, 7, 8).

Mirrors run_all.py's reporting style but delegates to each demonstrator's
own run_analysis module (SimpleLR + KernelSHAP, no sklearn required).
Saves calibration snapshots to each demonstrator's analysis_history/.

Usage:
    cd demonstrators
    python run_new.py
"""

import sys
from pathlib import Path
from datetime import date

import numpy as np

# ── path setup ────────────────────────────────────────────────────────────────
THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from cat1_facial_recognition.run_analysis import (
    run_analysis as run_fr,
    build_markdown as build_fr_md,
    ANALYSIS_HISTORY_DIR as FR_HIST,
)
from cat4_amazon_hiring.run_analysis import (
    run_analysis as run_ah,
    build_markdown as build_ah_md,
    ANALYSIS_HISTORY_DIR as AH_HIST,
)
from cat7_home_office_visa.run_analysis import (
    run_analysis as run_hov,
    build_markdown as build_hov_md,
    ANALYSIS_HISTORY_DIR as HOV_HIST,
)
from cat8_compas_sentencing.run_analysis import (
    run_analysis as run_cs,
    build_markdown as build_cs_md,
    ANALYSIS_HISTORY_DIR as CS_HIST,
)


DEMONSTRATORS = [
    ("facial_recognition",  "Cat. 1 — Biometric Identification",
     run_fr,  build_fr_md,  FR_HIST,
     "proxy_near", "near_dpr_mean"),
    ("amazon_hiring",       "Cat. 4 — Employment",
     run_ah,  build_ah_md,  AH_HIST,
     "proxy_near", None),
    ("home_office_visa",    "Cat. 7 — Migration",
     run_hov, build_hov_md, HOV_HIST,
     "proxy_near", "near_nrs_mean"),
    ("compas_sentencing",   "Cat. 8 — Administration of Justice",
     run_cs,  build_cs_md,  CS_HIST,
     "proxy_near", "near_race_mean"),
]


def fmt_p(p: float) -> str:
    return f"{p:.2e}"


def run_one(name, title, run_fn, build_md_fn, hist_dir, proxy_key, secondary_key,
            today: str, n_cases: int = 40) -> dict:
    print(f"\n{'='*70}")
    print(f"  {title.upper()}")
    print(f"  demonstrator: {name}")
    print(f"{'='*70}")

    res = run_fn(n_cases=n_cases)

    print(f"  Accuracy:   {res.get('accuracy', 0):.3f}  |  "
          f"near: {res.get('n_near', 0)}  |  far: {res.get('n_far', 0)}")
    print(f"  Flip-rate:  near={res.get('near_flip_mean', 0):.3f}  "
          f"far={res.get('far_flip_mean', 0):.3f}  "
          f"p={fmt_p(res.get('flip_mwu_p', 1))} "
          f"{'*** DISCRIMINATES' if res.get('flip_discriminates') else ''}")
    print(f"  Proxy attr: near={res.get('near_proxy_mean', 0):.3f}±"
          f"{res.get('near_proxy_std', 0):.3f}  "
          f"far={res.get('far_proxy_mean', 0):.3f}±"
          f"{res.get('far_proxy_std', 0):.3f}")
    if secondary_key and secondary_key in res:
        label = secondary_key.replace("near_", "").replace("_mean", "")
        std_key = secondary_key.replace("_mean", "_std")
        print(f"  {label:10s}: near={res[secondary_key]:.3f}±"
              f"{res.get(std_key, 0):.3f}")
    print(f"  Env dist:   near={res.get('near_env_mean', 0):.3f}  "
          f"far={res.get('far_env_mean', 0):.3f}  "
          f"p={fmt_p(res.get('env_mwu_p', 1))} "
          f"{'(discriminates)' if res.get('env_discriminates') else '(expected: within-manifold)'}")
    print(f"  Attr stab:  near={res.get('near_astab_mean', 0):.3f}  "
          f"far={res.get('far_astab_mean', 0):.3f}  "
          f"p={fmt_p(res.get('astab_mwu_p', 1))} "
          f"{'*** DISCRIMINATES' if res.get('astab_discriminates') else ''}")
    print(f"  Flagged:    near={res.get('near_flagged_pct', 0):.1%}  "
          f"far={res.get('far_flagged_pct', 0):.1%}")

    # Save markdown snapshot
    hist_dir.mkdir(parents=True, exist_ok=True)
    md = build_md_fn(res, today)
    out_path = hist_dir / f"{name}_analysis_{today}.md"
    out_path.write_text(md, encoding="utf-8")
    print(f"\n  Snapshot saved → {out_path.name}")

    return res


def main(n_cases: int = 40):
    today = date.today().strftime("%Y%m%d")

    print("=" * 70)
    print("  NEW DEMONSTRATORS REPLICATION (Cat. 1 / 4 / 7 / 8)")
    print("  SimpleLR + KernelSHAP. Validates instrument calibration.")
    print("  Every percentage is a design parameter to be recovered.")
    print("=" * 70)

    all_results = {}

    for name, title, run_fn, build_md_fn, hist_dir, proxy_key, secondary_key in DEMONSTRATORS:
        res = run_one(name, title, run_fn, build_md_fn, hist_dir,
                      proxy_key, secondary_key, today, n_cases)
        all_results[name] = res

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  SUMMARY: Does the instrument recover the coded structure?")
    print(f"{'='*70}")
    print(f"  {'Demonstrator':<25} {'Proxy near':>10} {'Flip p':>12} {'Env p':>12} {'Disc?':>6}")
    print(f"  {'-'*65}")
    for name, title, *_ in DEMONSTRATORS:
        res = all_results[name]
        flip_ok = res.get("flip_discriminates", False)
        env_ok  = res.get("env_discriminates", False)
        disc = "✓ flip" + (" ✓ env" if env_ok else "")
        print(f"  {name:<25} "
              f"{res.get('near_proxy_mean', 0):>10.3f} "
              f"{fmt_p(res.get('flip_mwu_p', 1)):>12} "
              f"{fmt_p(res.get('env_mwu_p', 1)):>12} "
              f"  {disc}")

    print(f"\n  Snapshots saved to each demonstrator's analysis_history/.")


if __name__ == "__main__":
    main()
