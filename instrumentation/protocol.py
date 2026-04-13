"""Algorithm 1: Decision-Boundary Instrumentation Protocol.

Generates structured telemetry for a single case at the output-action boundary.
Identical across all demonstrators; only feature groupings change.

All computations use ONLY the public inference interface (predict_proba).
This is an architectural commitment, not a limitation.
"""

import numpy as np

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    from .shap_lite import sample_background

from .signals.constraint_enforcement import compute_constraint_enforcement
from .signals.envelope_validity import _NumpyScaler
from .signals.envelope_validity import compute_envelope_validity
from .signals.decision_robustness import compute_decision_robustness
from .signals.record_integrity import compute_record_integrity
from .snippet import EvidenceSnippet


class InstrumentationProtocol:
    """Decision-boundary instrumentation protocol (Algorithm 1).

    Parameters
    ----------
    model : object
        Any model exposing predict_proba(X) -> array of shape (n, 2).
        Accessed ONLY via this interface.
    X_val : np.ndarray
        Validation set for background/envelope computation.
    feature_names : list[str]
        Names of all features.
    g_domain : list[int]
        Indices of domain-legitimate features.
    g_proxy : list[int]
        Indices of proxy/protected features.
    threshold : float
        Decision threshold (default 0.50).
    """

    def __init__(
        self,
        model,
        X_val: np.ndarray,
        feature_names: list[str],
        g_domain: list[int],
        g_proxy: list[int],
        threshold: float = 0.50,
    ):
        self.model = model
        self.predict_proba = model.predict_proba
        self.X_val = X_val
        self.feature_names = feature_names
        self.g_domain = g_domain
        self.g_proxy = g_proxy
        self.threshold = threshold

        # Precompute
        self.scaler = _NumpyScaler().fit(X_val)
        self.feature_ranges = X_val.max(axis=0) - X_val.min(axis=0) + 1e-10
        if HAS_SHAP:
            self.X_background = shap.sample(X_val, min(100, len(X_val)))
        else:
            self.X_background = sample_background(X_val, 100)

    def generate_snippet(
        self,
        x: np.ndarray,
        case_id: str = "auto",
        model_version: str = "unknown",
        # Hyperparameters
        n_shap_samples: int = 150,
        k_neighbours: int = 5,
        n_perturbations: int = 50,
        n_stability: int = 10,
        # Thresholds
        tau_ratio: float = 0.60,
        tau_envelope: float = 2.0,
        tau_flip: float = 0.30,
        tau_stab: float = 2.0,
        # Flags
        compute_integrity: bool = True,
    ) -> EvidenceSnippet:
        """Generate a full evidence snippet for a single case.

        Parameters
        ----------
        x : np.ndarray
            Feature vector (1D).
        case_id : str
            Case identifier.
        model_version : str
            Model version identifier.
        compute_integrity : bool
            If False, skip the expensive stability check.

        Returns
        -------
        EvidenceSnippet
        """
        x = np.asarray(x, dtype=float)

        # Step 1-2: Score and margin
        score = float(self.predict_proba(x.reshape(1, -1))[0, 1])
        margin = abs(score - self.threshold)

        # Steps 3-6: Constraint enforcement (KernelSHAP)
        ce = compute_constraint_enforcement(
            self.predict_proba,
            x,
            self.X_background,
            self.g_domain,
            self.g_proxy,
            self.feature_names,
            tau_ratio=tau_ratio,
            n_samples=n_shap_samples,
        )

        # Step 7-8: Envelope validity (k-NN)
        ev = compute_envelope_validity(
            x,
            self.X_val,
            k=k_neighbours,
            tau_envelope=tau_envelope,
            scaler=self.scaler,
        )

        # Steps 9-15: Decision robustness (flip-rate)
        dr = compute_decision_robustness(
            self.predict_proba,
            x,
            self.feature_ranges,
            threshold=self.threshold,
            n_pert=n_perturbations,
            tau_flip=tau_flip,
        )

        # Steps 16-22: Record integrity (attribution stability)
        if compute_integrity:
            ri = compute_record_integrity(
                self.predict_proba,
                x,
                self.X_background,
                self.feature_ranges,
                n_stab=n_stability,
                tau_stab=tau_stab,
            )
        else:
            ri = {"integrity_check": "NOT_COMPUTED", "attribution_stability_sigma": None}

        # Build snippet
        snippet = EvidenceSnippet(
            case_id=case_id,
            output_score=score,
            threshold=self.threshold,
            margin_to_threshold=margin,
            constraint_enforcement=ce,
            envelope_validity=ev,
            decision_robustness=dr,
            record_integrity=ri,
            model_version=model_version,
        )
        snippet.compute_verdict()

        return snippet
