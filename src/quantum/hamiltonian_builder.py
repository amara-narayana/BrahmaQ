"""Second-quantised Hamiltonian construction (Qiskit Nature driver).

Pipeline
--------
1.  ``rdkit`` parses the SMILES string and produces a 3-D geometry
    (ETKDG embedding + MMFF94 relaxation for polyatomics; an explicit diatomic
    construction for two-atom molecules, where distance-geometry embedding is
    ill-posed).
2.  ``qiskit_nature.second_q.drivers.PySCFDriver`` runs a Hartree-Fock
    calculation in the requested basis and returns an
    ``ElectronicStructureProblem``.
3.  The fermionic Hamiltonian is mapped onto qubits with ``ParityMapper``.
    Parity mapping removes two qubits by construction (the parity of the
    occupation numbers is fixed), so the mapper is given ``num_particles``;
    this is the current API for the legacy
    ``QubitConverter(ParityMapper(), two_qubit_reduction=True)`` -- for H2 in
    STO-3G it reduces 4 spin-orbitals to **2 qubits**.

Version notes (qiskit-nature 0.6.2)
-----------------------------------
*   ``qiskit_nature.drivers`` / ``PyMolecule`` were removed in 0.6.0; the
    supported driver is ``qiskit_nature.second_q.drivers.PySCFDriver``.
*   ``QubitConverter`` still exists in 0.6.x but is soft-deprecated; passing
    ``num_particles`` to the mapper is the forward-compatible equivalent.
*   ``QubitOperator`` (opflow) is soft-deprecated in qiskit-terra 0.45 and was
    dropped by ``qiskit-algorithms`` 0.2.x, so the qubit operator produced here
    is a ``SparsePauliOp`` -- the same object with the same algebra, usable by
    ``qiskit_algorithms.minimum_eigensolvers.VQE`` without opflow adapters.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

import qiskit_nature.settings
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import ParityMapper
from qiskit_nature.second_q.operators import FermionicOp
from qiskit_nature.second_q.problems import ElectronicStructureProblem
from qiskit.quantum_info import SparsePauliOp

from .minimal_basis_driver import NUCLEAR_CHARGES

# qiskit-nature 0.6.x still emits ``PauliSumOp`` (an opflow type that
# qiskit-terra 0.45 deprecates and qiskit-algorithms 0.2 has dropped) unless this
# switch is flipped.  Doing so makes the mappers return ``SparsePauliOp``
# directly, which is what every later stage of the engine consumes.
qiskit_nature.settings.use_pauli_sum_op = False

__all__ = [
    "MolecularGeometry",
    "HamiltonianData",
    "build_geometry",
    "build_hamiltonian",
    "build_problem",
    "DIATOMIC_REFERENCE_LENGTHS",
]

# RDKit is chatty about valences in the console; keep the engine's report clean.
RDLogger.DisableLog("rdApp.*")

#: Experimental equilibrium bond lengths (Angstrom) for two-atom molecules,
#: used when the caller does not supply one.  Distance-geometry embedding of a
#: two-atom molecule has no internal degrees of freedom to optimise, so an
#: explicit length is required by construction.
DIATOMIC_REFERENCE_LENGTHS: Dict[Tuple[str, str], float] = {
    ("H", "H"): 0.7348,     # H2, the reference molecule for this engine
    ("H", "Li"): 1.5949,    # LiH
    ("H", "F"): 0.9171,     # HF
    ("H", "O"): 0.9697,     # OH
    ("H", "N"): 1.0362,     # NH
    ("H", "C"): 1.1202,     # CH
    ("Li", "Li"): 2.6730,   # Li2
    ("Be", "H"): 1.3425,    # BeH
    ("H", "Be"): 1.3425,
}

_DEFAULT_DIATOMIC_LENGTH = 1.0

# Registry of electronic-structure backends; "pyscf" is the production driver
# (see :func:`build_problem` for the ``driver`` argument).
_DRIVERS = ("auto", "pyscf", "minimal")


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
@dataclass
class MolecularGeometry:
    """3-D nuclear coordinates plus provenance metadata."""

    smiles: str
    name: str
    symbols: List[str]
    coordinates: List[List[float]]
    source: str
    bond_length: Optional[float] = None
    bond_length_source: str = "n/a"
    warnings: List[str] = field(default_factory=list)

    @property
    def n_atoms(self) -> int:
        return len(self.symbols)

    @property
    def coords(self) -> np.ndarray:
        """``(n_atoms, 3)`` float64 array."""
        return np.asarray(self.coordinates, dtype=np.float64)

    @property
    def xyz_block(self) -> str:
        """Multi-line XYZ block (Angstrom)."""
        lines = [f"{len(self.symbols)}", self.name]
        lines += [
            f"{sym:<3s} {xyz[0]: .8f} {xyz[1]: .8f} {xyz[2]: .8f}"
            for sym, xyz in zip(self.symbols, self.coordinates)
        ]
        return "\n".join(lines)

    @property
    def pyscf_atom(self) -> str:
        """PySCF-style atom specification: ``"H 0.0 0.0 0.0; H 0.0 0.0 0.73"``."""
        return "; ".join(
            f"{sym} {xyz[0]:.10f} {xyz[1]:.10f} {xyz[2]:.10f}"
            for sym, xyz in zip(self.symbols, self.coordinates)
        )

    @property
    def bond_lengths(self) -> List[float]:
        """All pairwise internuclear distances (Angstrom)."""
        delta = self.coords[:, None, :] - self.coords[None, :, :]
        dist = np.sqrt(np.einsum("ijk,ijk->ij", delta, delta))
        return [float(dist[i, j]) for i in range(self.n_atoms)
                for j in range(i + 1, self.n_atoms)]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "smiles": self.smiles,
            "name": self.name,
            "symbols": self.symbols,
            "coordinates": [[float(c) for c in xyz] for xyz in self.coordinates],
            "source": self.source,
            "bond_length": self.bond_length,
            "bond_length_source": self.bond_length_source,
            "warnings": self.warnings,
        }


def build_geometry(
    smiles: str,
    bond_length: Optional[float] = None,
    optimize: bool = True,
    random_seed: int = 0xF00D,
) -> MolecularGeometry:
    """Build a relaxed 3-D geometry for ``smiles`` using RDKit.

    Parameters
    ----------
    smiles:
        Molecule specification, e.g. ``"[H][H]"`` or ``"[H][Li]"``.
    bond_length:
        Internuclear separation in Angstrom.  Only meaningful for two-atom
        molecules (and required, in practice, because there is nothing to
        optimise).  Falls back to :data:`DIATOMIC_REFERENCE_LENGTHS`.
    optimize:
        Run an MMFF94 (or UFF) relaxation for polyatomic molecules.
    random_seed:
        Fixed ETKDG seed so runs are reproducible.

    Raises
    ------
    ValueError
        If the SMILES cannot be parsed.
    RuntimeError
        If distance-geometry embedding fails for a polyatomic molecule.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES {smiles!r}")
    mol = Chem.AddHs(mol)
    symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
    warnings_: List[str] = []
    name = Chem.rdMolDescriptors.CalcMolFormula(mol)

    # -- single atom --------------------------------------------------------
    if len(symbols) == 1:
        return MolecularGeometry(
            smiles=smiles, name=name, symbols=symbols,
            coordinates=[[0.0, 0.0, 0.0]], source="single-atom",
            bond_length=None, bond_length_source="n/a", warnings=warnings_,
        )

    # -- diatomic: constructed explicitly ----------------------------------
    if len(symbols) == 2:
        key = tuple(sorted(symbols))
        if bond_length is not None:
            length, source = float(bond_length), "explicit"
        elif key in DIATOMIC_REFERENCE_LENGTHS:
            length, source = DIATOMIC_REFERENCE_LENGTHS[key], "reference-table"
        else:
            length, source = _DEFAULT_DIATOMIC_LENGTH, "default-fallback"
            warnings_.append(
                f"no reference bond length for {key}; using {length} A. "
                "Pass bond_length explicitly for quantitative results."
            )
        half = length / 2.0
        coordinates = [[0.0, 0.0, -half], [0.0, 0.0, half]]
        return MolecularGeometry(
            smiles=smiles, name=name, symbols=symbols, coordinates=coordinates,
            source="diatomic-constructed", bond_length=length,
            bond_length_source=source, warnings=warnings_,
        )

    # -- polyatomic: ETKDG embedding (+ optional force-field relaxation) ---
    params = AllChem.ETKDGv3()
    params.randomSeed = int(random_seed)
    if AllChem.EmbedMolecule(mol, params) != 0:
        params.useRandomCoords = True
        if AllChem.EmbedMolecule(mol, params) != 0:
            raise RuntimeError(
                f"RDKit failed to embed a 3-D geometry for {smiles!r}; "
                "supply coordinates explicitly instead."
            )
        warnings_.append("ETKDG retried with random coordinates")

    ff = "none"
    if optimize:
        if AllChem.MMFFHasAllMoleculeParams(mol):
            status = AllChem.MMFFOptimizeMolecule(mol, maxIters=2000)
            ff = "MMFF94"
        else:
            status = AllChem.UFFOptimizeMolecule(mol, maxIters=2000)
            ff = "UFF"
        if status == -1:
            warnings_.append("force-field relaxation did not converge")

    conformer = mol.GetConformer()
    coordinates = [
        [float(p) for p in conformer.GetAtomPosition(i)] for i in range(mol.GetNumAtoms())
    ]
    return MolecularGeometry(
        smiles=smiles, name=name, symbols=symbols, coordinates=coordinates,
        source=f"rdkit-etkdg+{ff}", bond_length=None,
        bond_length_source="optimised-geometry", warnings=warnings_,
    )


# ---------------------------------------------------------------------------
# Hamiltonian
# ---------------------------------------------------------------------------
@dataclass
class HamiltonianData:
    """Everything downstream stages need about the qubit Hamiltonian."""

    qubit_operator: SparsePauliOp
    fermionic_operator: FermionicOp
    mapper: ParityMapper
    problem: ElectronicStructureProblem
    geometry: MolecularGeometry
    basis: str
    num_spatial_orbitals: int
    num_particles: Tuple[int, int]
    num_qubits: int
    charge: int
    multiplicity: int
    nuclear_repulsion_energy: float
    electronic_energy_offset: float
    two_qubit_reduction: bool
    driver: str = "pyscf"
    reference_energy: Optional[float] = None
    warnings: List[str] = field(default_factory=list)

    # -- derived quantities -------------------------------------------------
    def exact_ground_state_energy(self) -> float:
        """Exact diagonalisation of the qubit Hamiltonian (the FCI reference).

        The parity-reduced Hamiltonian acts on the physical (particle-number
        and spin-preserving) subspace only, so its lowest eigenvalue is the FCI
        energy of the molecule in the chosen basis.  Adding the electronic
        energy offset -- nuclear repulsion, which Qiskit Nature factors out of
        ``second_q_ops`` for ``ElectronicStructureProblem`` -- yields the total
        molecular energy.
        """
        matrix = self.qubit_operator.to_matrix(sparse=False)
        if not np.allclose(matrix, matrix.conj().T, atol=1e-9):
            raise RuntimeError("qubit Hamiltonian is not Hermitian")
        return float(np.linalg.eigvalsh(matrix)[0].real + self.electronic_energy_offset)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "driver": self.driver,
            "reference_energy": self.reference_energy,
            "basis": self.basis,
            "num_spatial_orbitals": self.num_spatial_orbitals,
            "num_particles": list(self.num_particles),
            "num_qubits": self.num_qubits,
            "charge": self.charge,
            "multiplicity": self.multiplicity,
            "nuclear_repulsion_energy": self.nuclear_repulsion_energy,
            "electronic_energy_offset": self.electronic_energy_offset,
            "two_qubit_reduction": self.two_qubit_reduction,
            "geometry": self.geometry.to_dict(),
            "fermionic_terms": len(self.fermionic_operator),
            "qubit_pauli_terms": len(self.qubit_operator),
            "pauli_strings": [str(p) for p in self.qubit_operator.paulis],
            "warnings": self.warnings,
        }


def build_problem(
    geometry: MolecularGeometry,
    basis: str = "sto3g",
    charge: int = 0,
    multiplicity: int = 1,
    driver: str = "auto",
) -> Tuple[ElectronicStructureProblem, str, Optional[float], List[str]]:
    """Run the electronic-structure driver and return the problem to map.

    ``driver`` selects the backend:

    ``"pyscf"``
        ``qiskit_nature.second_q.drivers.PySCFDriver`` (the production path).
    ``"minimal"``
        the built-in STO-3G s-orbital + RHF fallback
        (:mod:`src.quantum.minimal_basis_driver`), which needs no external
        quantum-chemistry package.
    ``"auto"``
        PySCF when it is importable, otherwise the fallback -- with a warning
        recorded either way.

    Returns
    -------
    tuple
        ``(problem, driver_label, rhf_reference_energy_or_None, warnings)``
    """
    from . import minimal_basis_driver as minimal  # noqa: PLC0415 (avoid cycle)

    warnings_: List[str] = []
    requested = driver.lower()
    if requested not in {"auto", "pyscf", "minimal"}:
        raise ValueError(f"driver must be 'auto', 'pyscf' or 'minimal', got {driver!r}")

    use_pyscf = requested == "pyscf" or (requested == "auto" and minimal.pyscf_available())
    if requested == "pyscf" and not minimal.pyscf_available():
        raise ImportError(
            "driver='pyscf' was requested but PySCF cannot be imported. PyPI ships "
            "no Windows wheel for PySCF; use driver='minimal' or 'auto' there, or "
            "install PySCF from conda-forge on Linux/macOS."
        )

    if use_pyscf:
        spin = multiplicity - 1  # PySCF wants 2S
        pyscf_driver = PySCFDriver(
            atom=geometry.pyscf_atom,
            basis=basis,
            charge=int(charge),
            spin=int(spin),
        )
        problem = pyscf_driver.run()
        return problem, f"PySCF ({basis})", None, warnings_

    if requested == "auto":
        warnings_.append(
            "PySCF is not importable; using the built-in STO-3G minimal-basis "
            "fallback driver (H/He 1s, restricted closed shell)"
        )
    result = minimal.build_problem(
        geometry.symbols, geometry.coordinates, basis=basis,
        charge=charge, multiplicity=multiplicity,
    )
    warnings_.extend(result.warnings)
    return result.problem, result.driver_name, result.rhf_energy, warnings_


def build_hamiltonian(
    geometry: MolecularGeometry,
    basis: str = "sto3g",
    charge: int = 0,
    multiplicity: int = 1,
    two_qubit_reduction: bool = True,
    driver: str = "auto",
) -> HamiltonianData:
    """Build the electronic problem and map its Hamiltonian onto qubits.

    Parameters
    ----------
    geometry:
        Output of :func:`build_geometry`.
    basis:
        One-particle basis set (``"sto3g"``, ``"6-31g"``, ... for PySCF;
        STO-3G only for the fallback).
    charge, multiplicity:
        Molecular charge and spin multiplicity; ``(0, 1)`` for a singlet.
    two_qubit_reduction:
        Enable the parity-mapping two-qubit reduction by handing the mapper the
        particle numbers.  With ``False`` the full ``2 * n_spatial`` register is
        kept (useful for cross-checks).
    driver:
        ``"auto"`` (default), ``"pyscf"`` or ``"minimal"`` -- see
        :func:`build_problem`.

    Raises
    ------
    ValueError
        If the two-qubit reduction is requested with an odd electron count,
        where parity mapping cannot remove two qubits.
    """
    problem, driver_label, rhf_reference, warnings_ = build_problem(
        geometry, basis=basis, charge=charge, multiplicity=multiplicity, driver=driver
    )

    num_spatial_orbitals = int(problem.num_spatial_orbitals)
    if problem.num_particles is None:
        # Drivers normally set this; derive it from the electron count if not.
        n_electrons = sum(NUCLEAR_CHARGES.get(s, 0) for s in geometry.symbols) - int(charge)
        n_alpha = (n_electrons + int(multiplicity) - 1) // 2
        num_particles = (n_alpha, n_electrons - n_alpha)
        warnings_.append(
            f"driver did not set num_particles; derived {num_particles} from the electron count"
        )
    else:
        num_particles = tuple(int(n) for n in problem.num_particles)  # (n_alpha, n_beta)

    if two_qubit_reduction and sum(num_particles) % 2 != 0:
        raise ValueError(
            "two-qubit reduction requires an even electron count; got "
            f"{sum(num_particles)} electrons. Set two_qubit_reduction=False."
        )

    # ``second_q_ops()`` returns ``(main_op, aux_ops)`` in qiskit-nature 0.6.x;
    # going through the hamiltonian avoids depending on that container shape.
    fermionic_op = problem.hamiltonian.second_q_op()

    if two_qubit_reduction:
        # Equivalent to the legacy
        #   QubitConverter(ParityMapper(), two_qubit_reduction=True)
        # but expressed through the 0.6 mapper API.
        mapper = ParityMapper(num_particles=num_particles)
    else:  # keep every qubit
        mapper = ParityMapper()
        warnings_.append("two-qubit reduction disabled: using the full qubit register")

    qubit_operator = mapper.map(fermionic_op)
    if not isinstance(qubit_operator, SparsePauliOp):  # pragma: no cover - safety net
        # ``use_pauli_sum_op`` above makes this unreachable in 0.6.2; it guards
        # against a build that ignores the setting.
        warnings_.append(
            f"mapper returned {type(qubit_operator).__name__}; converted to SparsePauliOp"
        )
        primitive = getattr(qubit_operator, "primitive", None)
        if primitive is None:
            raise TypeError(f"cannot convert {type(qubit_operator).__name__} to SparsePauliOp")
        qubit_operator = SparsePauliOp(
            [str(pauli) for pauli in primitive.paulis], np.asarray(primitive.coeffs)
        )

    # Qiskit Nature's ElectronicStructureProblem keeps the nuclear repulsion
    # outside the second-quantised operator, so it is re-added when the total
    # molecular energy is formed.
    nuclear_repulsion = float(getattr(problem, "nuclear_repulsion_energy", 0.0) or 0.0)

    expected_qubits = 2 * num_spatial_orbitals - (2 if two_qubit_reduction else 0)
    qubit_operator = _pad_to_width(qubit_operator, expected_qubits, warnings_)

    return HamiltonianData(
        qubit_operator=qubit_operator,
        fermionic_operator=fermionic_op,
        mapper=mapper,
        problem=problem,
        geometry=geometry,
        basis=basis,
        num_spatial_orbitals=num_spatial_orbitals,
        num_particles=(num_particles[0], num_particles[1]) if len(num_particles) == 2
        else (sum(num_particles), 0),
        num_qubits=int(qubit_operator.num_qubits),
        charge=int(charge),
        multiplicity=int(multiplicity),
        nuclear_repulsion_energy=nuclear_repulsion,
        electronic_energy_offset=nuclear_repulsion,
        two_qubit_reduction=bool(two_qubit_reduction),
        driver=driver_label,
        reference_energy=rhf_reference,
        warnings=warnings_,
    )


def _pad_to_width(
    operator: SparsePauliOp, qubit_count: int, warnings_: List[str]
) -> SparsePauliOp:
    """Extend Pauli strings to ``qubit_count`` qubits (identity padding).

    The parity mapper returns an operator whose width can be smaller than the
    physical register it was reduced from; padding with identities keeps every
    downstream stage (estimator, noise model) on a consistent qubit count.
    """
    if operator.num_qubits >= qubit_count:
        return operator
    return SparsePauliOp(
        [pauli.to_label().ljust(qubit_count, "I") for pauli in operator.paulis],
        operator.coeffs,
    )


def compare_with_reference_energy(
    hamiltonian: HamiltonianData, reference: float, atol: float = 1e-3
) -> Dict[str, Any]:
    """Sanity check the exact diagonalisation against a literature value."""
    exact = hamiltonian.exact_ground_state_energy()
    return {
        "exact_ground_state_energy": exact,
        "reference": float(reference),
        "absolute_error": abs(exact - float(reference)),
        "within_tolerance": abs(exact - float(reference)) <= atol,
    }


if __name__ == "__main__":  # pragma: no cover - manual smoke check
    geom = build_geometry("[H][H]", bond_length=0.7348)
    print(geom.xyz_block)
    ham = build_hamiltonian(geom)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        print(f"qubits: {ham.num_qubits}  pauli terms: {len(ham.qubit_operator)}")
        print(ham.qubit_operator)
    print("exact FCI:", ham.exact_ground_state_energy())
