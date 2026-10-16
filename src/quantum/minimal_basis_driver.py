"""PySCF-free fallback driver: STO-3G *s*-orbital integrals + closed-shell RHF.

Why this exists
---------------
``PySCFDriver`` is the production path (see :mod:`src.quantum.hamiltonian_builder`),
but PySCF is not installable everywhere: **PyPI publishes no Windows wheel for
any PySCF 2.x** and conda-forge only ships it for Python 3.11, whose numpy-2 ABI
is incompatible with qiskit 0.45.2.  Without a fallback the engine cannot run at
all on such a host.

This module reproduces exactly the object the rest of the pipeline consumes -- an
``ElectronicStructureProblem`` carrying MO-basis one- and two-body integrals plus
the nuclear repulsion -- using closed-form Gaussian integrals (Rys/Boys ``F0``,
all s-type) and a small RHF SCF.  Only ``math`` and ``numpy`` are used.

Scope and validation
--------------------
* Basis: **STO-3G 1s on H and He only.**  Anything else raises a clear error
  pointing at PySCF, because p- and d-type integrals are not implemented here.
* Restricted closed shell (even electron count, singlet).
* Verified against literature anchors for H2/STO-3G in STO-3G at 0.735 A:
  RHF ``-1.1167 Ha`` and FCI ``-1.1373 Ha`` (see ``tests/test_smoke.py``).

Integral conventions
--------------------
ERIs are built in **chemists'** notation ``(mn|lk)`` and converted with
qiskit-nature's own ``to_physicist_ordering`` before being handed to
``ElectronicIntegrals``, so the index convention is the library's, not a guess.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "MinimalBasisResult",
    "build_problem",
    "pyscf_available",
    "supported_element",
    "SUPPORTED_ELEMENTS",
]

#: STO-3G contraction for the 1s orbital: (exponent, coefficient) per primitive.
#: Data from Hehre, Stewart & Pople, J. Chem. Phys. 51, 2657 (1969).
STO3G_S_BASIS: Dict[str, List[Tuple[float, float]]] = {
    "H": [(3.42525091, 0.15432897), (0.62391373, 0.53532814), (0.16885540, 0.44463454)],
    "He": [(6.36242139, 0.15432897), (1.15892300, 0.53532814), (0.31364979, 0.44463454)],
}

NUCLEAR_CHARGES: Dict[str, float] = {"H": 1.0, "He": 2.0}

SUPPORTED_ELEMENTS = tuple(sorted(STO3G_S_BASIS))

#: The closed-form Gaussian integrals below are in atomic units, so nuclear
#: positions must be converted from Angstrom to Bohr before use.
ANGSTROM_TO_BOHR = 1.8897259885793597


def supported_element(symbol: str) -> bool:
    """True if this driver can build an integral set for ``symbol``."""
    return symbol in STO3G_S_BASIS


def pyscf_available() -> bool:
    """True when PySCF can actually be imported (not just referenced)."""
    try:
        import pyscf  # noqa: F401,PLC0415
    except Exception:
        return False
    return True


# ---------------------------------------------------------------------------
# Primitive Gaussian integrals (s-type)
# ---------------------------------------------------------------------------
def _boys0(t: float) -> float:
    """Boys function of order 0: ``F0(t) = 1/2 sqrt(pi/t) erf(sqrt(t))``."""
    if t <= 0.0:
        return 1.0
    return 0.5 * math.sqrt(math.pi / t) * math.erf(math.sqrt(t))


def _dist2(a: np.ndarray, b: np.ndarray) -> float:
    d = a - b
    return float(d @ d)


def _overlap_primitive(a: float, A: np.ndarray, b: float, B: np.ndarray) -> float:
    ab = a + b
    return (math.pi / ab) ** 1.5 * math.exp(-a * b / ab * _dist2(A, B))


def _kinetic_primitive(a: float, A: np.ndarray, b: float, B: np.ndarray) -> float:
    ab = a + b
    r2 = _dist2(A, B)
    return a * b / ab * (3.0 - 2.0 * a * b / ab * r2) * _overlap_primitive(a, A, b, B)


def _nuclear_primitive(
    a: float, A: np.ndarray, b: float, B: np.ndarray, C: np.ndarray, z: float
) -> float:
    ab = a + b
    P = (a * A + b * B) / ab
    return (
        -2.0 * math.pi / ab * z
        * math.exp(-a * b / ab * _dist2(A, B))
        * _boys0(ab * _dist2(P, C))
    )


def _eri_primitive(
    a: float, A: np.ndarray, b: float, B: np.ndarray,
    c: float, C: np.ndarray, d: float, D: np.ndarray,
) -> float:
    """``(ab|cd)`` for four primitive s Gaussians (chemists' notation)."""
    ab, cd = a + b, c + d
    abcd = ab + cd
    P = (a * A + b * B) / ab
    Q = (c * C + d * D) / cd
    return (
        2.0 * math.pi ** 2.5 / (ab * cd * math.sqrt(abcd))
        * math.exp(-a * b / ab * _dist2(A, B))
        * math.exp(-c * d / cd * _dist2(C, D))
        * _boys0(ab * cd / abcd * _dist2(P, Q))
    )


# ---------------------------------------------------------------------------
# Contracted basis integrals
# ---------------------------------------------------------------------------
def _primitive_normalisation(a: float) -> float:
    """``(2a/pi)^(3/4)`` -- the normalisation of a primitive s Gaussian."""
    return (2.0 * a / math.pi) ** 0.75


def _normalised_contraction(symbol: str, A: np.ndarray) -> List[Tuple[float, float]]:
    """STO-3G primitives rescaled so the contracted 1s function is unit-normalised.

    Published STO-3G coefficients (Hehre, Stewart & Pople 1969) multiply
    *normalised* primitive Gaussians, while the integral kernels here use the
    bare ``exp(-a|r-A|^2)`` form.  Each coefficient therefore carries an extra
    ``(2a/pi)^(3/4)`` factor; omitting it makes the contracted orbital far too
    diffuse and shifts the FCI energy by ~0.1 Ha.
    """
    primitives = [
        (a, c * _primitive_normalisation(a)) for a, c in STO3G_S_BASIS[symbol]
    ]
    norm = 0.0
    for a, ca in primitives:
        for b, cb in primitives:
            norm += ca * cb * _overlap_primitive(a, A, b, A)
    scale = 1.0 / math.sqrt(norm)
    return [(a, c * scale) for a, c in primitives]


@dataclass
class _Basis:
    symbols: List[str]
    positions: List[np.ndarray]
    contractions: List[List[Tuple[float, float]]]


def _build_basis(symbols: Sequence[str], coordinates: np.ndarray) -> _Basis:
    contractions = []
    positions = []
    for index, symbol in enumerate(symbols):
        if symbol not in STO3G_S_BASIS:
            raise ValueError(
                f"the minimal-basis fallback driver supports {SUPPORTED_ELEMENTS} "
                f"only; got {symbol!r}. Install PySCF (Linux/macOS: `pip install "
                "pyscf`; Windows: use WSL or a conda env on Python 3.11) to run "
                "this molecule."
            )
        position = np.asarray(coordinates[index], dtype=float)
        positions.append(position)
        contractions.append(_normalised_contraction(symbol, position))
    return _Basis(list(symbols), positions, contractions)


def _one_electron_matrices(basis: _Basis, charges: Sequence[float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(basis.symbols)
    overlap = np.zeros((n, n))
    kinetic = np.zeros((n, n))
    nuclear = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            A, B = basis.positions[i], basis.positions[j]
            for a, ca in basis.contractions[i]:
                for b, cb in basis.contractions[j]:
                    factor = ca * cb
                    overlap[i, j] += factor * _overlap_primitive(a, A, b, B)
                    kinetic[i, j] += factor * _kinetic_primitive(a, A, b, B)
                    for k, z in enumerate(charges):
                        nuclear[i, j] += factor * _nuclear_primitive(
                            a, A, b, B, basis.positions[k], z
                        )
    return overlap, kinetic, nuclear


def _eri_tensor(basis: _Basis) -> np.ndarray:
    n = len(basis.symbols)
    eri = np.zeros((n, n, n, n))
    for i in range(n):
        for j in range(n):
            for k in range(n):
                for l in range(n):  # noqa: E741 - orbital index, standard notation
                    total = 0.0
                    for a, ca in basis.contractions[i]:
                        for b, cb in basis.contractions[j]:
                            for c, cc in basis.contractions[k]:
                                for d, cd in basis.contractions[l]:
                                    total += (
                                        ca * cb * cc * cd
                                        * _eri_primitive(
                                            a, basis.positions[i], b, basis.positions[j],
                                            c, basis.positions[k], d, basis.positions[l],
                                        )
                                    )
                    eri[i, j, k, l] = total
    return eri


def _nuclear_repulsion(basis: _Basis, charges: Sequence[float]) -> float:
    total = 0.0
    n = len(basis.symbols)
    for i in range(n):
        for j in range(i + 1, n):
            distance = math.sqrt(_dist2(basis.positions[i], basis.positions[j]))
            total += charges[i] * charges[j] / distance
    return total


# ---------------------------------------------------------------------------
# Closed-shell RHF
# ---------------------------------------------------------------------------
@dataclass
class MinimalBasisResult:
    """The constructed problem plus the SCF diagnostics behind it."""

    problem: Any
    driver_name: str
    rhf_energy: float
    n_electrons: int
    n_occupied: int
    n_orbitals: int
    scf_iterations: int
    scf_converged: bool
    mo_energies: List[float]
    nuclear_repulsion: float
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "driver": self.driver_name,
            "rhf_energy": self.rhf_energy,
            "n_electrons": self.n_electrons,
            "n_occupied": self.n_occupied,
            "n_orbitals": self.n_orbitals,
            "scf_iterations": self.scf_iterations,
            "scf_converged": self.scf_converged,
            "mo_energies": self.mo_energies,
            "nuclear_repulsion": self.nuclear_repulsion,
            "warnings": self.warnings,
        }


def _scf(
    overlap: np.ndarray,
    core: np.ndarray,
    eri: np.ndarray,
    n_electrons: int,
    max_iterations: int = 200,
    tolerance: float = 1e-10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, bool, List[str]]:
    """RHF in the AO basis.

    Returns ``(C, eps, P, F, iterations, converged, warnings)`` -- coefficients,
    orbital energies, density, the final Fock matrix and the diagnostics.
    """
    warnings_: List[str] = []
    n = overlap.shape[0]
    n_occupied = n_electrons // 2

    eigenvalues, eigenvectors = np.linalg.eigh(overlap)
    if np.any(eigenvalues <= 1e-12):
        raise ValueError("linearly dependent basis: overlap matrix is singular")
    X = eigenvectors @ np.diag(eigenvalues ** -0.5) @ eigenvectors.T

    density = np.zeros((n, n))
    fock = core.copy()
    converged = False
    damping = 1.0  # 1.0 = plain SCF, < 1.0 = density damping
    iterations = 0

    for iterations in range(1, max_iterations + 1):
        coulomb = np.einsum("ls,mnls->mn", density, eri)
        exchange = np.einsum("ls,mlns->mn", density, eri)
        fock = core + coulomb - 0.5 * exchange
        orth_fock = X.T @ fock @ X
        orbital_energies, coefficients_prime = np.linalg.eigh(orth_fock)
        coefficients = X @ coefficients_prime
        occupied = coefficients[:, :n_occupied]
        new_density = 2.0 * occupied @ occupied.T
        if damping < 1.0:
            new_density = damping * new_density + (1.0 - damping) * density
        delta = float(np.abs(new_density - density).max())
        density = new_density
        if delta < tolerance:
            converged = True
            break
        if iterations == 30 and not converged and damping == 1.0:
            # Plain iteration can oscillate for stretched geometries; fall back
            # to density damping rather than silently returning garbage.
            damping = 0.5
            warnings_.append("plain SCF did not converge in 30 cycles; density damping enabled")

    # Rebuild the Fock matrix from the final density so the energy is consistent.
    coulomb = np.einsum("ls,mnls->mn", density, eri)
    exchange = np.einsum("ls,mlns->mn", density, eri)
    fock = core + coulomb - 0.5 * exchange
    if not converged:
        warnings_.append(
            f"SCF did not reach {tolerance:g} in {max_iterations} cycles "
            f"(last density change {delta:.3e})"
        )
    return coefficients, orbital_energies, density, fock, iterations, converged, warnings_


def build_problem(
    symbols: Sequence[str],
    coordinates: Sequence[Sequence[float]],
    basis: str = "sto3g",
    charge: int = 0,
    multiplicity: int = 1,
) -> MinimalBasisResult:
    """Build an ``ElectronicStructureProblem`` from STO-3G s-orbital integrals.

    Parameters
    ----------
    symbols, coordinates:
        Nuclear species and Cartesian positions (Angstrom).
    basis:
        Only ``"sto3g"`` is implemented.
    charge, multiplicity:
        Restricted closed-shell calculations only: the electron count must be
        even and the multiplicity 1.

    Raises
    ------
    ValueError
        For unsupported elements, basis sets, open-shell or charged-with-odd-
        electron systems -- all of which require PySCF.
    """
    if basis.lower() != "sto3g":
        raise ValueError(
            f"the minimal-basis fallback driver implements STO-3G only, not {basis!r}; "
            "install PySCF for other basis sets"
        )
    if multiplicity != 1:
        raise ValueError(
            f"the minimal-basis fallback driver is restricted closed-shell "
            f"(multiplicity 1), got multiplicity {multiplicity}; use PySCF"
        )

    # Angstrom -> Bohr: everything below is in atomic units.
    coordinates = np.asarray(coordinates, dtype=float) * ANGSTROM_TO_BOHR
    basis_ = _build_basis(symbols, coordinates)
    charges = [NUCLEAR_CHARGES[symbol] for symbol in symbols]
    n_electrons = int(round(sum(charges))) - int(charge)
    if n_electrons % 2 != 0:
        raise ValueError(
            f"{n_electrons} electrons cannot be treated closed-shell; use PySCF"
        )
    if n_electrons <= 0:
        raise ValueError("no electrons: nothing to compute")

    overlap, kinetic, nuclear = _one_electron_matrices(basis_, charges)
    core = kinetic + nuclear
    eri = _eri_tensor(basis_)
    e_nuc = _nuclear_repulsion(basis_, charges)

    coefficients, orbital_energies, density, fock, iterations, converged, warnings_ = _scf(
        overlap, core, eri, n_electrons
    )
    # Closed-shell RHF electronic energy: E = 1/2 Tr[P (H + F)].
    e_electronic = 0.5 * float(np.sum(density * (core + fock)))
    rhf_energy = e_electronic + e_nuc

    # -- transform the integrals into the MO basis -------------------------
    h1_mo = coefficients.T @ core @ coefficients
    eri_mo = np.einsum(
        "mp,nq,lr,ks,mnlk->pqrs", coefficients, coefficients, coefficients, coefficients, eri
    )

    # -- hand them to qiskit-nature in its own index convention -------------
    from qiskit_nature.second_q.hamiltonians import ElectronicEnergy
    from qiskit_nature.second_q.operators import ElectronicIntegrals
    from qiskit_nature.second_q.operators.tensor_ordering import (
        IndexType,
        to_physicist_ordering,
    )
    from qiskit_nature.second_q.problems import ElectronicStructureProblem

    h2_phys = np.asarray(
        to_physicist_ordering(eri_mo, index_order=IndexType.CHEMIST), dtype=float
    )
    # Restricted closed shell: the alpha, beta and mixed blocks are identical.
    integrals = ElectronicIntegrals.from_raw_integrals(
        h1_mo, h2_phys, h1_mo, h2_phys, h2_phys, auto_index_order=False
    )
    energy = ElectronicEnergy(integrals, constants={"nuclear_repulsion_energy": e_nuc})
    problem = ElectronicStructureProblem(energy)
    problem.num_particles = (n_electrons // 2, n_electrons // 2)

    if not np.allclose(h1_mo, h1_mo.T, atol=1e-10):
        warnings_.append("one-body integrals are not symmetric")

    return MinimalBasisResult(
        problem=problem,
        driver_name=f"minimal-basis STO-3G s-orbital (fallback, {len(basis_.symbols)} functions)",
        rhf_energy=rhf_energy,
        n_electrons=n_electrons,
        n_occupied=n_electrons // 2,
        n_orbitals=len(basis_.symbols),
        scf_iterations=iterations,
        scf_converged=converged,
        mo_energies=[float(e) for e in orbital_energies],
        nuclear_repulsion=e_nuc,
        warnings=warnings_,
    )
