"""Record integrity signal (cross-cutting admissibility gate).

Measures whether the attribution is reproducible under micro-noise.
An unstable attribution is noise, not evidence.
Falls back to shap_lite if the shap library is not available.
"""

import numpy as np

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False


def _compute_attribution(model_predict_proba, x, X_background, n_samples):
    """Compute feature attribution vector."""
    if HAS_SHAP:
        explainer = shap.KernelExplainer(model_predict_proba, X_background)
        sv = explainer.shap_values(x.reshape(1, -1), nsamples=n_samples, silent=True)
        return sv[1][0] if isinstance(sv, list) else sv[0]
    else:
        from ..shap_lite import kernel_shap_lite
        return kernel_shap_lite(model_predict_proba, x, X_background, n_samples)


def compute_record_integrity(
    model_predict_proba,
    x: np.ndarray,
    X_background: np.ndarray,
    feature_ranges: np.ndarray,
    n_stab: int = 10,
    noise_scale: float = 0.01,
    n_samples: int = 80,
    tau_stab: float = 2.0,
) -> dict:
    """Compute record integrity via attribution stability check.

    Parameters
    ----------
    model_predict_proba : callable
        Black-box predict_proba function.
    x : np.ndarray
        Feature vector for the current case (1D).
    X_background : np.ndarray
        Background sample for attribution.
    feature_ranges : np.ndarray
        Per-feature range for scaling micro-noise.
    n_stab : int
        Number of stability repetitions.
    noise_scale : float
        Scale of micro-noise.
    n_samples : int
        Attribution samples per repetition.
    tau_stab : float
        Threshold for sigma. Above this -> FAIL.

    Returns
    -------
    dict with keys: integrity_check, attribution_stability_sigma
    """
    ranks = []
    for _ in range(n_stab):
        noise = np.random.normal(0, noise_scale, len(x)) * feature_ranges
        x_noisy = x + noise
        attr_j = _compute_attribution(model_predict_proba, x_noisy, X_background, n_samples)
        ranks.append(np.argsort(np.argsort(-np.abs(attr_j))))

    sigma = float(np.mean(np.std(np.array(ranks), axis=0)))
    integrity = "FAIL" if sigma > tau_stab else "PASS"

    return {
        "integrity_check": integrity,
        "attribution_stability_sigma": sigma,
    }
