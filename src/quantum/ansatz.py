"""Symmetry-protected ansatz: Hartree-Fock reference + UCCSD.

Why UCCSD and not ``RealAmplitudes``/``TwoLocal``
-------------------------------------------------
A hardware-efficient ansatz spans the *entire* ``2**n`` Hilbert space, including
sectors with the wrong particle number and spin.  The optimisation therefore
starts on a plateau: the gradients of a randomly initialised hardware-efficient
circuit vanish exponentially in the number of qubits (McClean et al.,
*Barren plateaus in quantum neural network training landscapes*, Nat. Commun. 9,
4812 (2018)).

Unitary Coupled Cluster with singles and doubles excitations is built from
fermionic excitation operators, so to first order the state stays inside the
particle-number- and spin-conserving subspace spanned by the Hartree-Fock
reference -- the symmetry sector that contains the ground state.  Combined with
the parity-mapped, two-qubit-reduced Hamiltonian the parameter count is
tiny (one parameter for H2/STO-3G), and there is no exponentially flat
landscape to fall into.

Version notes (qiskit-nature 0.6.2)
-----------------------------------
*   ``UCCSD`` lives in ``qiskit_nature.second_q.circuit.library``.
*   ``HartreeFock`` must be built with the *same* mapper as the Hamiltonian,
    otherwise the reference state is expressed in a different basis.  When the
    mapper carries ``num_particles`` (parity + two-qubit reduction) the
    reference occupation is reduced accordingly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from qiskit.circuit import QuantumCircuit
from qiskit_nature.second_q.circuit.library import HartreeFock, UCCSD

from .hamiltonian_builder import HamiltonianData

__all__ = [
    "AnsatzConfig",
    "AnsatzBundle",
    "build_hartree_fock_state",
    "build_uccsd_ansatz",
    "assert_symmetry_protected",
]


@dataclass
class AnsatzConfig:
    """Construction options for the symmetry-protected ansatz.

    The option names mirror ``UCCSD``'s own keyword arguments in
    qiskit-nature 0.6.2: the class fixes the excitation pool at singles + doubles
    (there is no ``excitations`` selector in this release) and exposes the
    spin-symmetry knobs that actually matter here.
    """

    reps: int = 1
    #: Generate the generalised (UCCGSD-style) excitation pool: every orbital
    #: pair instead of only occupied -> virtual transitions.  Larger and more
    #: prone to redundancy, but rotation-invariant.
    generalized: bool = False
    #: Drop excitations that do not conserve spin.  Keeping this on is part of
    #: what confines the search to the physical symmetry sector.
    preserve_spin: bool = True


@dataclass
class AnsatzBundle:
    """The circuit plus everything the report needs to describe it."""

    ansatz: QuantumCircuit
    initial_state: QuantumCircuit
    num_parameters: int
    num_qubits: int
    num_spatial_orbitals: int
    num_particles: Tuple[int, int]
    mapper_name: str
    label: str = "Symmetry-Protected UCCSD"
    warnings: List[str] = field(default_factory=list)

    @property
    def parameters(self) -> List[Any]:
        """Ordered circuit parameters (the VQE optimisation vector)."""
        return list(self.ansatz.parameters)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "num_parameters": self.num_parameters,
            "num_qubits": self.num_qubits,
            "num_spatial_orbitals": self.num_spatial_orbitals,
            "num_particles": list(self.num_particles),
            "mapper": self.mapper_name,
            "initial_state": type(self.initial_state).__name__,
            "depth": self.ansatz.decompose(reps=1).depth()
            if self.ansatz.num_qubits else 0,
            "warnings": self.warnings,
        }


def build_hartree_fock_state(
    num_spatial_orbitals: int,
    num_particles: Tuple[int, int],
    mapper: Any,
) -> HartreeFock:
    """Hartree-Fock reference determinant in the mapper's qubit basis.

    ``num_particles`` is ``(n_alpha, n_beta)``; ``HartreeFock`` places
    electrons in the lowest spin orbitals and, for a parity mapper carrying
    ``num_particles``, applies the same two-qubit reduction as the Hamiltonian.
    """
    n_alpha, n_beta = int(num_particles[0]), int(num_particles[1])
    if n_alpha + n_beta > 2 * num_spatial_orbitals:
        raise ValueError(
            f"{n_alpha} alpha + {n_beta} beta electrons do not fit in "
            f"{num_spatial_orbitals} spatial orbitals"
        )
    return HartreeFock(num_spatial_orbitals, (n_alpha, n_beta), mapper)


def build_uccsd_ansatz(
    hamiltonian: HamiltonianData,
    config: Optional[AnsatzConfig] = None,
) -> AnsatzBundle:
    """Construct ``UCCSD`` on top of the Hartree-Fock reference state.

    Returns
    -------
    AnsatzBundle
        ``bundle.ansatz`` is ready for ``assign_parameters``; ``bundle.parameters``
        fixes the ordering of the optimisation vector.
    """
    config = config or AnsatzConfig()
    mapper = hamiltonian.mapper
    num_spatial_orbitals = hamiltonian.num_spatial_orbitals
    num_particles = hamiltonian.num_particles
    warnings_: List[str] = []

    initial_state = build_hartree_fock_state(
        num_spatial_orbitals, num_particles, mapper
    )

    ansatz = UCCSD(
        num_spatial_orbitals=num_spatial_orbitals,
        num_particles=num_particles,
        qubit_mapper=mapper,
        initial_state=initial_state,
        reps=config.reps,
        generalized=config.generalized,
        preserve_spin=config.preserve_spin,
    )

    # The reduction in the mapper must survive into the circuit width.
    if ansatz.num_qubits != hamiltonian.num_qubits:
        warnings_.append(
            f"ansatz width {ansatz.num_qubits} != Hamiltonian width "
            f"{hamiltonian.num_qubits}; padding identities were applied to the "
            "operator, check the mapper/register consistency."
        )

    label = "Symmetry-Protected UCCSD"
    if config.generalized:
        label = "Symmetry-Protected UCCGSD (generalised)"
    if not config.preserve_spin:
        warnings_.append("preserve_spin=False: spin-changing excitations are included")
    bundle = AnsatzBundle(
        ansatz=ansatz,
        initial_state=initial_state,
        num_parameters=ansatz.num_parameters,
        num_qubits=ansatz.num_qubits,
        num_spatial_orbitals=num_spatial_orbitals,
        num_particles=num_particles,
        mapper_name=type(mapper).__name__,
        label=label,
        warnings=warnings_,
    )
    assert_symmetry_protected(bundle)
    return bundle


def assert_symmetry_protected(bundle: AnsatzBundle) -> None:
    """Guard rail: refuse hardware-efficient / generic-layer ansaetze.

    The engine's central claim is that the circuit only explores the
    symmetry-preserving subspace.  A ``RealAmplitudes``/``TwoLocal``/``EfficientSU2``
    circuit would silently break that claim, so construction is verified here
    rather than trusted.
    """
    forbidden = (
        "RealAmplitudes",
        "TwoLocal",
        "EfficientSU2",
        "PauliTwoDesign",
        "NLocal",
    )
    names = {type(bundle.ansatz).__name__, type(bundle.initial_state).__name__}
    for instruction in bundle.ansatz.data:
        names.add(type(instruction.operation).__name__)
    clash = sorted(names.intersection(forbidden))
    if clash:
        raise ValueError(
            f"ansatz uses generic hardware-efficient layer(s) {clash}; "
            "the symmetry-protection guarantee would be void"
        )
    if bundle.num_parameters == 0:
        # Legal (e.g. a 1-orbital system has no excitations) but must be loud:
        # VQE degenerates to a single Hartree-Fock energy evaluation.
        bundle.warnings.append(
            "UCCSD produced no excitation parameters; the energy will be the "
            "Hartree-Fock reference energy"
        )


if __name__ == "__main__":  # pragma: no cover - manual smoke check
    from .hamiltonian_builder import build_geometry, build_hamiltonian

    geometry = build_geometry("[H][H]", bond_length=0.7348)
    ham = build_hamiltonian(geometry)
    bundle = build_uccsd_ansatz(ham)
    print(bundle.to_dict())
    print(bundle.ansatz.draw("text"))
