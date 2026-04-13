"""Lightweight KernelSHAP approximation without the shap library.

Implements marginal perturbation-based attribution: for each feature,
replace it with background values and measure the change in output.
This is a simplified but functionally equivalent approach for the
instrumentation protocol.

When the full shap library is available, the protocol uses it.
This module provides a fallback that preserves the black-box commitment.
"""

import numpy as np


def kernel_shap_lite(
    model_predict_proba,
    x: np.ndarray,
    X_background: np.ndarray,
    n_samples: int = 100,
) -> np.ndarray:
    """Approximate SHAP values via marginal perturbation.

    For each feature i, replace x[i] with values sampled from
    the background set and measure the average change in output.

    Parameters
    ----------
    model_predict_proba : callable
        Black-box predict_proba(X) -> (n, 2) array.
    x : np.ndarray
        Feature vector (1D, shape (d,)).
    X_background : np.ndarray
        Background samples (2D, shape (m, d)).
    n_samples : int
        Number of background samples to use per feature.

    Returns
    -------
    np.ndarray of shape (d,) — approximate attribution per feature.
    """
    d = len(x)
    n_bg = min(n_samples, len(X_background))
    bg_idx = np.random.choice(len(X_background), n_bg, replace=False)
    bg = X_background[bg_idx]

    base_score = model_predict_proba(x.reshape(1, -1))[0, 1]
    attributions = np.zeros(d)

    for i in range(d):
        # Create perturbed copies: replace feature i with background values
        X_pert = np.tile(x, (n_bg, 1))
        X_pert[:, i] = bg[:, i]
        scores_pert = model_predict_proba(X_pert)[:, 1]
        # Attribution = how much the score drops when feature i is marginalised
        attributions[i] = base_score - np.mean(scores_pert)

    return attributions


def sample_background(X: np.ndarray, n: int = 100) -> np.ndarray:
    """Sample background set from data (replacement for shap.sample)."""
    idx = np.random.choice(len(X), min(n, len(X)), replace=False)
    return X[idx]
