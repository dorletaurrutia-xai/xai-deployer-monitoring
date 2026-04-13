"""Constraint enforcement signal (Leveson/STAMP).

Measures whether the output's behavioural basis is grounded in
domain-legitimate features or contaminated by proxies.
Uses KernelSHAP attribution grouped into legitimate vs. proxy sets.
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
        shap_values = explainer.shap_values(
            x.reshape(1, -1), nsamples=n_samples, silent=True
        )
        if isinstance(shap_values, list):
            return shap_values[1][0]
        return shap_values[0]
    else:
        from ..shap_lite import kernel_shap_lite
        return kernel_shap_lite(model_predict_proba, x, X_background, n_samples)


def compute_constraint_enforcement(
    model_predict_proba,
    x: np.ndarray,
    X_background: np.ndarray,
    g_domain: list[int],
    g_proxy: list[int],
    feature_names: list[str],
    tau_ratio: float = 0.60,
    n_samples: int = 150,
) -> dict:
    """Compute constraint enforcement signal via KernelSHAP.

    Parameters
    ----------
    model_predict_proba : callable
        Black-box predict_proba function. Only public interface used.
    x : np.ndarray
        Feature vector for the current case (1D).
    X_background : np.ndarray
        Background sample for KernelSHAP.
    g_domain : list[int]
        Indices of domain-legitimate features.
    g_proxy : list[int]
        Indices of proxy/protected features.
    feature_names : list[str]
        Names of all features.
    tau_ratio : float
        Threshold for legitimate-feature ratio. Below this -> CONTESTED.
    n_samples : int
        Number of samples for attribution computation.

    Returns
    -------
    dict with keys: status, legitimate_feature_ratio, top_features,
    proxy_attribution, attribution_vector
    """
    attr = _compute_attribution(model_predict_proba, x, X_background, n_samples)

    total_mass = np.sum(np.abs(attr)) + 1e-10
    domain_mass = np.sum(np.abs(attr[g_domain]))
    ratio_legit = domain_mass / total_mass

    # Top features by absolute attribution
    top_idx = np.argsort(-np.abs(attr))[:4]
    top_features = {feature_names[i]: float(attr[i]) for i in top_idx}

    # Per-proxy attribution fraction
    proxy_attribution = {
        feature_names[i]: float(np.abs(attr[i]) / total_mass) for i in g_proxy
    }

    status = "CONTESTED" if ratio_legit < tau_ratio else "OK"

    return {
        "status": status,
        "legitimate_feature_ratio": float(ratio_legit),
        "top_features": top_features,
        "proxy_attribution": proxy_attribution,
        "attribution_vector": attr,
    }
