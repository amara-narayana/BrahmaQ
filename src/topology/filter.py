"""Persistent homology of the nuclear point cloud (Vietoris--Rips filtration).

The tractability filter treats the 3-D nuclear coordinates of a molecule as a
finite metric space ``(X, d)`` and studies the Vietoris--Rips filtration
``VR_epsilon(X)`` built on it:

    VR_eps = { sigma subset of X : diam(sigma) <= eps }

Filtering over ``eps`` yields persistence diagrams ``Dgm_k`` for the homology
groups ``H_k``.  From those diagrams we extract two quantities:

* **Persistence entropy** ``E`` -- the Shannon entropy of the normalised bar
  lifetimes ``l_i = death_i - birth_i``:

      p_i = l_i / sum_j l_j ,      E = -sum_i p_i log p_i

  ``E`` measures how *spread out* the topological features are across scales.
  A molecule whose merging history is dominated by a single length scale
  (a rigid, simply-connected geometry) has few bars and small ``E``; a geometry
  with many competing length scales has large ``E``.

* **Betti numbers** ``(beta_0, beta_1, beta_2)`` evaluated at a reference
  scale ``L_ref`` (see :func:`betti_numbers`).  ``beta_0`` counts connected
  components (fragmentation) and ``beta_1`` counts robust 1-cycles (rings).

Numerical-honesty notes (these matter for interpreting the classifier):

1.  The Rips complex of a *finite* point cloud becomes a full simplex at large
    ``eps``, so ``H_1`` and ``H_2`` never carry an essential (infinite) class.
    Ripser therefore returns only finite bars for ``k >= 1``.  A "Betti number"
    obtained by counting infinite bars would be identically zero and useless,
    so ``beta_1`` here counts 1-cycles whose *persistence* exceeds a prominence
    threshold, and ``beta_0`` counts components alive at ``L_ref``.
2.  Hydrogens sit close to their heavy neighbours, which can spawn short-lived
    spurious 1-cycles.  Bars shorter than ``min_persistence`` are therefore
    treated as discretisation artefacts, not obstructions.  The threshold is a
    modelling choice, exposed in ``config.json``.
3.  ``gudhi`` (when importable) independently computes the *exact* Betti
    numbers of ``VR(L_ref)`` from the simplex tree.  Those values are recorded
    as a cross-check against the ripser-derived counts.

References
----------
Edelsbrunner & Harer, *Computational Topology: An Introduction* (2010).
Atienza et al., "Persistence entropy: a stable entropy for persistent
homology", arXiv:1905.08434 (2019).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from ripser import ripser

__all__ = [
    "PersistenceSignature",
    "point_cloud_from_coordinates",
    "compute_persistence",
    "persistence_entropy",
    "betti_numbers",
    "cross_check_betti_gudhi",
    "analyse",
]

# Distances below this are considered numerically degenerate (coincident nuclei).
_EPS_DISTANCE = 1e-9


# ---------------------------------------------------------------------------
# Point cloud construction
# ---------------------------------------------------------------------------
def point_cloud_from_coordinates(coordinates: Sequence[Sequence[float]]) -> np.ndarray:
    """Convert nuclear coordinates into a validated ``(n_atoms, 3)`` float64 array.

    Parameters
    ----------
    coordinates:
        Sequence of ``[x, y, z]`` triplets in Angstrom (RDKit's native unit).

    Raises
    ------
    ValueError
        If the input is not a non-empty ``(n, 3)`` array of finite numbers.
    """
    points = np.asarray(coordinates, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(
            f"coordinates must have shape (n_atoms, 3); received {points.shape}"
        )
    if points.shape[0] == 0:
        raise ValueError("coordinates must contain at least one atom")
    if not np.all(np.isfinite(points)):
        raise ValueError("coordinates contain NaN/Inf -- check the RDKit geometry")
    return points


def _pairwise_distances(points: np.ndarray) -> np.ndarray:
    """Euclidean distance matrix of the point cloud."""
    delta = points[:, None, :] - points[None, :, :]
    return np.sqrt(np.einsum("ijk,ijk->ij", delta, delta))


# ---------------------------------------------------------------------------
# Persistence computation
# ---------------------------------------------------------------------------
@dataclass
class PersistenceSignature:
    """Topological summary of a nuclear point cloud.

    Attributes
    ----------
    n_points:
        Number of nuclei in the cloud.
    diagrams:
        ``{dimension: ndarray(n_bars, 2)}`` persistence diagrams as returned by
        ripser.  ``death = inf`` marks the single essential ``H_0`` class.
    persistence_entropy:
        The decision value used by the classifier (see ``entropy_mode``).
    entropy_raw:
        Shannon entropy in nats of the pooled finite bar lifetimes.
    entropy_normalized:
        ``entropy_raw / log(n_bars)`` when more than one bar exists, else 0.0.
        Scale-free, in ``[0, 1]``.
    betti_numbers:
        ``{0: beta_0, 1: beta_1, 2: beta_2}`` -- see :func:`betti_numbers`.
    max_persistence:
        Longest finite bar per dimension (Angstrom).
    n_features:
        Number of *significant* bars per dimension.
    reference_scale:
        Scale (Angstrom) at which ``beta_0`` was evaluated.
    prominence:
        Minimum persistence for a bar to count as a feature.
    """

    n_points: int
    diagrams: Dict[int, np.ndarray]
    persistence_entropy: float
    entropy_raw: float
    entropy_normalized: float
    betti_numbers: Dict[int, int]
    max_persistence: Dict[int, float]
    n_features: Dict[int, int]
    reference_scale: float
    prominence: float
    maxdim: int
    entropy_mode: str = "normalized"
    diameter: float = 0.0
    cross_check: Optional[Dict[str, Any]] = None
    warnings: List[str] = field(default_factory=list)

    # -- serialisation ------------------------------------------------------
    def to_dict(self, include_diagrams: bool = False) -> Dict[str, Any]:
        """JSON-safe view of the signature (``inf`` becomes ``null``)."""
        payload: Dict[str, Any] = {
            "n_points": self.n_points,
            "persistence_entropy": self.persistence_entropy,
            "entropy_raw": self.entropy_raw,
            "entropy_normalized": self.entropy_normalized,
            "entropy_mode": self.entropy_mode,
            "betti_numbers": {"beta_0": self.betti_numbers[0],
                              "beta_1": self.betti_numbers[1],
                              "beta_2": self.betti_numbers[2]},
            "max_persistence": {str(k): v for k, v in self.max_persistence.items()},
            "n_features": {str(k): v for k, v in self.n_features.items()},
            "reference_scale": self.reference_scale,
            "prominence": self.prominence,
            "diameter": self.diameter,
            "maxdim": self.maxdim,
        }
        if include_diagrams:
            payload["diagrams"] = {
                str(dim): [[float(b), (None if not np.isfinite(d) else float(d))]
                           for b, d in np.asarray(dgm, dtype=float)]
                for dim, dgm in self.diagrams.items()
            }
        if self.cross_check is not None:
            payload["gudhi_cross_check"] = self.cross_check
        if self.warnings:
            payload["warnings"] = list(self.warnings)
        return payload


def persistence_entropy(
    diagram: np.ndarray, normalize: bool = True
) -> float:
    """Shannon entropy of the bar lifetimes of a single persistence diagram.

    ``normalize=True`` divides by ``log(n_bars)`` so that the result lies in
    ``[0, 1]`` and is comparable across molecules of different size.
    Non-finite bars (the essential class) and zero-length bars are ignored.
    """
    diagram = np.asarray(diagram, dtype=float)
    if diagram.size == 0:
        return 0.0
    lifetimes = diagram[:, 1] - diagram[:, 0]
    lifetimes = lifetimes[np.isfinite(lifetimes) & (lifetimes > _EPS_DISTANCE)]
    if lifetimes.size == 0:
        return 0.0
    probabilities = lifetimes / lifetimes.sum()
    raw = float(-np.sum(probabilities * np.log(probabilities)))
    if normalize and lifetimes.size > 1:
        return raw / float(np.log(lifetimes.size))
    return raw


def betti_numbers(
    diagrams: Dict[int, np.ndarray],
    reference_scale: float,
    prominence: float = 0.3,
) -> Dict[int, int]:
    """Betti numbers of the Rips filtration evaluated for physical relevance.

    * ``beta_0``: ``H_0`` bars *alive at* ``reference_scale`` -- the number of
      connected components that survive past the reference length.  ``1`` for a
      bonded molecule; ``> 1`` means the geometry has fragmented.
    * ``beta_k`` (``k >= 1``): bars with ``persistence >= prominence`` -- robust
      cycles rather than the essential classes of an *infinite* complex, which
      cannot exist for a finite point cloud (see module docstring).
    """
    betti: Dict[int, int] = {}
    for dim, diagram in diagrams.items():
        diagram = np.asarray(diagram, dtype=float)
        if dim == 0:
            if diagram.size == 0:
                betti[dim] = 0
                continue
            births, deaths = diagram[:, 0], diagram[:, 1]
            alive = (births <= reference_scale) & (reference_scale < deaths)
            betti[dim] = int(np.count_nonzero(alive))
        else:
            if diagram.size == 0:
                betti[dim] = 0
                continue
            lifetimes = diagram[:, 1] - diagram[:, 0]
            finite = np.isfinite(lifetimes)
            betti[dim] = int(np.count_nonzero(finite & (lifetimes >= prominence)))
    return betti


def _significant_features(
    diagrams: Dict[int, np.ndarray], prominence: float
) -> Dict[int, int]:
    """Number of bars with ``persistence >= prominence`` per dimension."""
    counts: Dict[int, int] = {}
    for dim, diagram in diagrams.items():
        diagram = np.asarray(diagram, dtype=float)
        if diagram.size == 0:
            counts[dim] = 0
            continue
        lifetimes = diagram[:, 1] - diagram[:, 0]
        finite = np.isfinite(lifetimes)
        counts[dim] = int(np.count_nonzero(finite & (lifetimes >= prominence)))
    return counts


def _max_persistence(diagrams: Dict[int, np.ndarray]) -> Dict[int, float]:
    """Longest finite bar per dimension (0.0 when the diagram is empty)."""
    longest: Dict[int, float] = {}
    for dim, diagram in diagrams.items():
        diagram = np.asarray(diagram, dtype=float)
        if diagram.size == 0:
            longest[dim] = 0.0
            continue
        lifetimes = (diagram[:, 1] - diagram[:, 0])
        lifetimes = lifetimes[np.isfinite(lifetimes)]
        longest[dim] = float(lifetimes.max()) if lifetimes.size else 0.0
    return longest


def cross_check_betti_gudhi(
    points: np.ndarray, reference_scale: float, maxdim: int = 2
) -> Optional[Dict[str, Any]]:
    """Exact Betti numbers of ``VR(reference_scale)`` via ``gudhi``.

    Unlike the ripser-based counts, this is the genuine dimension of the
    homology of the finite simplicial complex at one scale, because essential
    classes of a finite complex *can* exist.  Used as an independent
    verification of the ripser pipeline; returns ``None`` when gudhi is not
    installed.
    """
    try:
        import gudhi  # noqa: PLC0415  (optional dependency, imported lazily)
    except Exception:  # pragma: no cover - environment dependent
        return None

    try:
        rips = gudhi.RipsComplex(points=points, max_edge_length=float(reference_scale))
        simplex_tree = rips.create_simplex_tree(max_dimension=maxdim + 1)
        n_simplices = int(simplex_tree.num_simplices())

        # GUDHI's compute_persistence() segfaults (C++ access violation, not a
        # Python exception) when the complex holds only 0-simplices, so that
        # path is avoided.  A pure-vertex complex is trivial anyway: its Betti
        # vector is (n_vertices, 0, 0, ...) by inspection.
        edges = n_simplices - points.shape[0]
        if edges <= 0:
            return {
                "backend": f"gudhi {gudhi.__version__}",
                "complex": f"RipsComplex(max_edge_length={reference_scale})",
                "betti_numbers": {f"beta_{k}": (int(points.shape[0]) if k == 0 else 0)
                                  for k in range(maxdim + 1)},
                "n_simplices": n_simplices,
                "note": "disconnected at this scale: Betti vector read off directly",
            }

        # ``persistence_dim_max=True`` is required: without it the maximal
        # dimension present in the complex is skipped, which silently drops
        # beta_1 whenever no triangles have formed yet (the usual case for a
        # molecular ring at a covalent scale).
        pairs = simplex_tree.persistence(persistence_dim_max=True)
        betti = {f"beta_{k}": 0 for k in range(maxdim + 1)}
        for dim, (_, death) in pairs:
            if death == float("inf") and f"beta_{dim}" in betti:
                betti[f"beta_{dim}"] += 1
        return {
            "backend": f"gudhi {gudhi.__version__}",
            "complex": f"RipsComplex(max_edge_length={reference_scale})",
            "betti_numbers": betti,
            "n_simplices": n_simplices,
            "definition": (
                "exact Betti numbers of the finite complex VR(L_ref); for k >= 1 "
                "these differ from the prominence-filtered bar counts, which "
                "measure persistence over the whole filtration"
            ),
        }
    except Exception as exc:  # pragma: no cover - environment dependent
        return {"backend": "gudhi", "error": f"{type(exc).__name__}: {exc}"}


def compute_persistence(
    coordinates: Sequence[Sequence[float]],
    maxdim: int = 2,
    reference_scale: float = 1.4,
    prominence: float = 0.3,
    entropy_mode: str = "normalized",
    gudhi_cross_check: bool = True,
) -> PersistenceSignature:
    """Run the Vietoris--Rips filtration over the nuclear point cloud.

    Parameters
    ----------
    coordinates:
        ``(n_atoms, 3)`` nuclear coordinates in Angstrom.
    maxdim:
        Highest homology dimension to compute (clamped to ``n_atoms - 1``).
    reference_scale:
        Scale (Angstrom) at which ``beta_0`` -- and the gudhi cross-check -- is
        evaluated.  ~1.4 A is a covalent-reach length: bonded atoms are merged,
        dissociated atoms are not.
    prominence:
        Minimum bar persistence for a cycle to count as a real topological
        feature rather than a discretisation artefact.
    entropy_mode:
        ``"normalized"`` (default, threshold 0.5 is scale-free) or ``"raw"``.
    gudhi_cross_check:
        Compute the exact Betti numbers of ``VR(reference_scale)`` with gudhi.

    Returns
    -------
    PersistenceSignature
    """
    points = point_cloud_from_coordinates(coordinates)
    n_points = int(points.shape[0])
    distances = _pairwise_distances(points)
    diameter = float(distances.max()) if n_points > 1 else 0.0
    warnings: List[str] = []

    # Ripser cannot resolve homology above (n_points - 1) and needs >= 2 points.
    effective_maxdim = int(max(0, min(maxdim, n_points - 1)))
    if effective_maxdim < maxdim:
        warnings.append(
            f"maxdim reduced to {effective_maxdim}: a cloud of {n_points} points "
            "supports no higher-dimensional homology."
        )

    if n_points < 2:
        diagrams: Dict[int, np.ndarray] = {
            dim: np.empty((0, 2)) for dim in range(maxdim + 1)
        }
        warnings.append("single-atom cloud: persistence diagrams are empty")
    elif diameter <= _EPS_DISTANCE:
        diagrams = {dim: np.empty((0, 2)) for dim in range(maxdim + 1)}
        warnings.append("all nuclei coincide: degenerate filtration, no features")
    else:
        # The distance matrix is passed explicitly (distance_matrix=True)
        # rather than letting ripser infer the metric from the array shape:
        # for H2 the cloud is 2x3 and ripser would warn that it has "more
        # columns than rows", even though rows are the points by construction.
        result = ripser(
            distances,
            maxdim=effective_maxdim,
            thresh=np.inf,
            distance_matrix=True,
            do_cocycles=False,
        )
        diagrams = {
            dim: np.asarray(dgm, dtype=float).reshape(-1, 2)
            for dim, dgm in enumerate(result["dgms"])
        }
        for dim in range(maxdim + 1):
            diagrams.setdefault(dim, np.empty((0, 2)))

    # -- persistence entropy (pooled over dimensions) ----------------------
    pooled = np.vstack(
        [np.asarray(diagrams[dim], dtype=float) for dim in sorted(diagrams)]
        or [np.empty((0, 2))]
    ) if diagrams else np.empty((0, 2))
    lifetimes = pooled[:, 1] - pooled[:, 0] if pooled.size else np.empty(0)
    lifetimes = lifetimes[np.isfinite(lifetimes) & (lifetimes > _EPS_DISTANCE)]
    entropy_raw = persistence_entropy(
        np.column_stack([np.zeros_like(lifetimes), lifetimes])
        if lifetimes.size
        else np.empty((0, 2)),
        normalize=False,
    )
    entropy_normalized = persistence_entropy(
        np.column_stack([np.zeros_like(lifetimes), lifetimes])
        if lifetimes.size
        else np.empty((0, 2)),
        normalize=True,
    )
    if entropy_mode == "raw":
        decision_entropy = entropy_raw
    elif entropy_mode == "normalized":
        decision_entropy = entropy_normalized
    else:
        raise ValueError(f"unknown entropy_mode {entropy_mode!r}")

    betti = betti_numbers(diagrams, reference_scale, prominence)
    cross_check = (
        cross_check_betti_gudhi(points, reference_scale, maxdim=effective_maxdim)
        if gudhi_cross_check and n_points >= 2
        else None
    )
    if cross_check and "betti_numbers" in cross_check:
        # beta_0 has the same definition in both pipelines, so it must agree.
        # beta_k for k >= 1 does not: gudhi ranks the homology of VR(L_ref),
        # while the ripser count below counts bars anywhere in the filtration.
        cross_check["beta_0_agrees"] = (
            cross_check["betti_numbers"].get("beta_0") == betti[0]
        )
        if not cross_check["beta_0_agrees"]:
            warnings.append(
                "gudhi and ripser disagree on beta_0 "
                f"({cross_check['betti_numbers'].get('beta_0')} vs {betti[0]})"
            )

    return PersistenceSignature(
        n_points=n_points,
        diagrams=diagrams,
        persistence_entropy=float(decision_entropy),
        entropy_raw=float(entropy_raw),
        entropy_normalized=float(entropy_normalized),
        betti_numbers=betti,
        max_persistence=_max_persistence(diagrams),
        n_features=_significant_features(diagrams, prominence),
        reference_scale=float(reference_scale),
        prominence=float(prominence),
        maxdim=effective_maxdim,
        entropy_mode=entropy_mode,
        diameter=diameter,
        cross_check=cross_check,
        warnings=warnings,
    )


def analyse(
    coordinates: Sequence[Sequence[float]],
    threshold: float = 0.5,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Convenience wrapper: persistence + threshold decision, JSON-ready."""
    signature = compute_persistence(coordinates, **kwargs)
    return {
        "persistence_entropy": signature.persistence_entropy,
        "threshold": threshold,
        "betti_numbers": signature.to_dict()["betti_numbers"],
        "signature": signature.to_dict(),
    }


if __name__ == "__main__":  # pragma: no cover - manual smoke check
    h2 = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.7348]]
    print(json.dumps(analyse(h2), indent=2))
