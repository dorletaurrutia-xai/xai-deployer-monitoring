"""Decision-boundary instrumentation for deployer-side AI monitoring."""

from .protocol import InstrumentationProtocol
from .snippet import EvidenceSnippet

__version__ = "0.1.0"
__all__ = ["InstrumentationProtocol", "EvidenceSnippet"]
