"""Decision robustness signal (Rasmussen/Parasuraman).

Measures how close the decision is to reversal under bounded perturbation.
Produces the flip-rate: fraction of perturbations that reverse the output.
"""

import numpy as np


def compute_decision_robustness(
    model_predict_proba,
    x: np.ndarray,
    feature_ranges: np.ndarray,
    threshold: float = 0.50,
    n_pert: int = 50,
    noise_scale: float = 0.05,
    tau_flip: float = 0.30,
) -> dict:
    """Compute decision robustness via bounded perturbation.

    Parameters
    ----------
    model_predict_proba : callable
        Black-box predict_proba function.
    x : np.ndarray
        Feature vector for the current case (1D).
    feature_ranges : np.ndarray
        Per-feature range (max - min) for scaling noise.
    threshold : float
        Decision threshold.
    n_pert : int
        Number of perturbations.
    noise_scale : float
        Scale of Gaussian noise relative to feature range.
    tau_flip : float
        Threshold for flip-rate. Above this -> CONTESTED.

    Returns
    -------
    dict with keys: status, flip_rate
    """
    base_score = model_predict_proba(x.reshape(1, -1))[0, 1]
    base_decision = int(base_score >= threshold)

    flips = 0
    for _ in range(n_pert):
        noise = np.random.normal(0, noise_scale, len(x)) * feature_ranges
        x_pert = np.clip(x + noise, 0, None)  # domain-safe clipping
        score_pert = model_predict_proba(x_pert.reshape(1, -1))[0, 1]
        if int(score_pert >= threshold) != base_decision:
            flips += 1

    flip_rate = flips / n_pert
    status = "CONTESTED" if flip_rate > tau_flip else "OK"

    return {
        "status": status,
        "flip_rate": float(flip_rate),
    }
