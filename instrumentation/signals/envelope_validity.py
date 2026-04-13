"""Envelope validity signal (Rasmussen).

Measures whether the current case sits within the validated
operating envelope or has migrated beyond it.
Uses k-NN distance to the validation set.
"""

import numpy as np


class _NumpyScaler:
    """Z-score scaler using numpy (no sklearn dependency)."""

    def fit(self, X: np.ndarray) -> "_NumpyScaler":
        self.mean_ = X.mean(axis=0)
        self.scale_ = X.std(axis=0) + 1e-10
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_) / self.scale_


def compute_envelope_validity(
    x: np.ndarray,
    X_val: np.ndarray,
    k: int = 5,
    tau_envelope: float = 2.0,
    scaler=None,
) -> dict:
    """Compute envelope validity via k-NN distance.

    Parameters
    ----------
    x : np.ndarray
        Feature vector for the current case (1D).
    X_val : np.ndarray
        Validation set (2D).
    k : int
        Number of neighbours.
    tau_envelope : float
        Distance threshold. Above this -> CONTESTED.
    scaler : scaler object or None
        If provided, scales features before distance computation.
        Accepts sklearn StandardScaler or _NumpyScaler (duck-typed).

    Returns
    -------
    dict with keys: status, distance_to_nearest_prototype
    """
    if scaler is not None:
        X_scaled = scaler.transform(X_val)
        x_scaled = scaler.transform(x.reshape(1, -1))[0]
    else:
        X_scaled = X_val
        x_scaled = x

    # k-NN via pairwise Euclidean distances (numpy only)
    diffs = X_scaled - x_scaled
    dists = np.sqrt((diffs ** 2).sum(axis=1))
    k_nearest = np.sort(dists)[:k]
    d_proto = float(k_nearest.mean())

    status = "CONTESTED" if d_proto > tau_envelope else "OK"

    return {
        "status": status,
        "distance_to_nearest_prototype": d_proto,
    }
