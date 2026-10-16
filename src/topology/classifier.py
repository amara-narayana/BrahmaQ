"""Tractability decision logic built on the topological signature.

The rule implemented here is the one specified for Q-TMCE:

    IF persistence_entropy > entropy_threshold OR beta_1 > 0
        -> {"tractable": False, "reason": "Topological obstruction detected"}
    ELSE
        -> {"tractable": True, "reason": "Simple topology verified"}

``beta_2`` is reported for completeness but does not currently veto a molecule;
cavities are rare in small molecules and the prominence threshold controls the
count (see :mod:`src.topology.filter` for the numerical caveats).

The classifier returns a plain ``dict`` that is safe to ``json.dumps`` -- this
is the "JSON classification result" required by the specification.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional, Sequence

from .filter import PersistenceSignature, compute_persistence

__all__ = ["TopologyConfig", "TopologicalClassifier", "classify_topology"]

#: A molecule reaching this many significant 1-cycles is treated as
#: topologically obstructed regardless of entropy.
_DEFAULT_ENTROPY_THRESHOLD = 0.5


class TopologyConfig(dict):
    """Dict-backed configuration holder with attribute-free, JSON-friendly keys."""

    @property
    def entropy_threshold(self) -> float:
        return float(self.get("entropy_threshold", _DEFAULT_ENTROPY_THRESHOLD))


def classify_topology(
    signature: PersistenceSignature,
    entropy_threshold: float = _DEFAULT_ENTROPY_THRESHOLD,
    allow_beta_1: int = 0,
) -> Dict[str, Any]:
    """Apply the tractability decision to a persistence signature.

    Parameters
    ----------
    signature:
        Output of :func:`src.topology.filter.compute_persistence`.
    entropy_threshold:
        Persistence entropy above which the geometry is deemed obstructed.
    allow_beta_1:
        Number of significant 1-cycles tolerated before vetoing (0 = strict,
        matching the specification).

    Returns
    -------
    dict
        ``{"tractable": bool, "reason": str, "metrics": {...}, "signature": {...}}``
    """
    entropy = float(signature.persistence_entropy)
    beta_0 = int(signature.betti_numbers.get(0, 0))
    beta_1 = int(signature.betti_numbers.get(1, 0))
    beta_2 = int(signature.betti_numbers.get(2, 0))

    checks = {
        "entropy_ok": entropy <= entropy_threshold,
        "cycles_ok": beta_1 <= allow_beta_1,
        "connected_ok": beta_0 <= 1,
    }

    if not checks["entropy_ok"]:
        reason = (
            f"Topological obstruction detected: persistence entropy "
            f"{entropy:.4f} exceeds threshold {entropy_threshold:.2f} "
            f"(competing length scales across the filtration)."
        )
    elif not checks["cycles_ok"]:
        reason = (
            f"Topological obstruction detected: beta_1 = {beta_1} persistent "
            f"1-cycle(s) above prominence {signature.prominence:.2f} A."
        )
    elif not checks["connected_ok"]:
        reason = (
            f"Topological obstruction detected: beta_0 = {beta_0} connected "
            f"component(s) at reference scale {signature.reference_scale:.2f} A "
            "(the geometry is fragmented)."
        )
    else:
        reason = (
            "Low persistence entropy, no topological obstructions."
            if checks["entropy_ok"]
            else "Simple topology verified."
        )

    tractable = all(checks.values())

    return {
        "tractable": tractable,
        "reason": reason,
        "checks": checks,
        "metrics": {
            "persistence_entropy": entropy,
            "entropy_mode": signature.entropy_mode,
            "entropy_raw": signature.entropy_raw,
            "entropy_normalized": signature.entropy_normalized,
            "entropy_threshold": float(entropy_threshold),
            "beta_0": beta_0,
            "beta_1": beta_1,
            "beta_2": beta_2,
            "n_significant_features": signature.n_features,
            "max_persistence": signature.max_persistence,
            "reference_scale": signature.reference_scale,
            "prominence": signature.prominence,
            "diameter": signature.diameter,
        },
        "signature": signature.to_dict(),
    }


class TopologicalClassifier:
    """Config-driven tractability classifier.

    Example
    -------
    >>> clf = TopologicalClassifier({"entropy_threshold": 0.5, "prominence": 0.3})
    >>> result = clf.run([[0.0, 0.0, 0.0], [0.0, 0.0, 0.7348]])
    >>> result["tractable"]
    True
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        config = dict(config or {})
        self.entropy_threshold = float(
            config.get("entropy_threshold", _DEFAULT_ENTROPY_THRESHOLD)
        )
        self.allow_beta_1 = int(config.get("allow_beta_1", 0))
        self.reference_scale = float(config.get("reference_scale", 1.4))
        self.prominence = float(config.get("prominence", 0.3))
        self.maxdim = int(config.get("maxdim", 2))
        self.entropy_mode = str(config.get("entropy_mode", "normalized"))
        self.gudhi_cross_check = bool(config.get("gudhi_cross_check", True))
        if self.entropy_mode not in {"normalized", "raw"}:
            raise ValueError(f"entropy_mode must be 'normalized' or 'raw', got {self.entropy_mode!r}")
        self.last_signature: Optional[PersistenceSignature] = None

    def run(self, coordinates: Sequence[Sequence[float]]) -> Dict[str, Any]:
        """Compute the signature and classify; caches the last signature."""
        self.last_signature = compute_persistence(
            coordinates,
            maxdim=self.maxdim,
            reference_scale=self.reference_scale,
            prominence=self.prominence,
            entropy_mode=self.entropy_mode,
            gudhi_cross_check=self.gudhi_cross_check,
        )
        return classify_topology(
            self.last_signature,
            entropy_threshold=self.entropy_threshold,
            allow_beta_1=self.allow_beta_1,
        )

    def run_json(self, coordinates: Sequence[Sequence[float]]) -> str:
        """Same as :meth:`run`, serialised to a JSON string."""
        return json.dumps(self.run(coordinates), indent=2, sort_keys=True)


if __name__ == "__main__":  # pragma: no cover - manual smoke check
    h2 = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.7348]]
    ring = [
        [0.0, 1.397, 0.0],
        [1.2097, 0.6985, 0.0],
        [1.2097, -0.6985, 0.0],
        [0.0, -1.397, 0.0],
        [-1.2097, -0.6985, 0.0],
        [-1.2097, 0.6985, 0.0],
    ]
    print("H2   ->", json.dumps(classify_topology(
        compute_persistence(h2)), indent=2))
    print("ring ->", json.dumps(classify_topology(
        compute_persistence(ring)), indent=2))
