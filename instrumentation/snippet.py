"""Evidence snippet: the atomic unit of the audit trail.

A structured, machine-readable JSON record binding output, context,
and XAI-derived signals at each decision event.
"""

import json
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class EvidenceSnippet:
    """Structured evidence record produced at each decision event."""

    case_id: str
    output_score: float
    threshold: float
    margin_to_threshold: float

    # Constraint enforcement (Leveson/STAMP)
    constraint_enforcement: dict = field(default_factory=dict)

    # Envelope validity (Rasmussen)
    envelope_validity: dict = field(default_factory=dict)

    # Decision robustness (Rasmussen/Parasuraman)
    decision_robustness: dict = field(default_factory=dict)

    # Record integrity (admissibility gate)
    record_integrity: dict = field(default_factory=dict)

    # Action verdict
    action_verdict: str = "PROCEED"

    # Metadata
    model_version: str = "unknown"
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    # Optional extensions
    extensions: dict = field(default_factory=dict)

    def compute_verdict(self) -> str:
        """Non-compensatory verdict: most restrictive status wins."""
        statuses = []
        for signal in [
            self.constraint_enforcement,
            self.envelope_validity,
            self.decision_robustness,
        ]:
            if isinstance(signal, dict) and "status" in signal:
                statuses.append(signal["status"])

        integrity = self.record_integrity.get("integrity_check", "PASS")
        if integrity == "FAIL":
            self.action_verdict = "BLOCK"
        elif any(s == "CONTESTED" for s in statuses):
            self.action_verdict = "PROCEED WITH FLAG"
        else:
            self.action_verdict = "PROCEED"
        return self.action_verdict

    def to_dict(self) -> dict:
        """Serialise to dictionary (JSON-ready)."""
        d = asdict(self)
        # Remove raw attribution vector from serialisation
        if "attribution_vector" in d.get("constraint_enforcement", {}):
            d["constraint_enforcement"] = {
                k: v
                for k, v in d["constraint_enforcement"].items()
                if k != "attribution_vector"
            }
        return d

    def to_json(self, indent: int = 2) -> str:
        """Serialise to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def save(self, path: str) -> None:
        """Save snippet to a JSON file."""
        with open(path, "w") as f:
            f.write(self.to_json())
