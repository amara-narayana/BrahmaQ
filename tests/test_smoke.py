"""Behavioural tests for the Q-TMCE engine.

Run with either runner::

    python -m pytest tests/test_smoke.py -q
    python tests/test_smoke.py          # no pytest required

These exercise the paths the orchestrator does *not* take during a nominal H2
run: the obstruction branch, the tamper branch, the barren-plateau branch, and
the exact report layout.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator, SparsePauliOp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.crypto.zkp_wrapper import LatticeCommitmentScheme  # noqa: E402
from src.main import format_report  # noqa: E402
from src.quantum.ansatz import build_uccsd_ansatz  # noqa: E402
from src.quantum.hamiltonian_builder import (  # noqa: E402
    build_geometry,
    build_hamiltonian,
)
from src.quantum.mitigation import (  # noqa: E402
    _evaluate,
    build_depolarizing_noise_model,
    extrapolate_to_zero_noise,
    fold_circuit,
)
from src.quantum.simulator import make_statevector_estimator  # noqa: E402
from src.quantum.simulator import (  # noqa: E402
    BarrenPlateauConfig,
    BarrenPlateauError,
    run_vqe,
)
from src.topology.classifier import TopologicalClassifier  # noqa: E402
from src.topology.filter import compute_persistence  # noqa: E402

H2 = [[0.0, 0.0, -0.3674], [0.0, 0.0, 0.3674]]
BENZENE_RING = [
    [0.0, 1.397, 0.0], [1.2097, 0.6985, 0.0], [1.2097, -0.6985, 0.0],
    [0.0, -1.397, 0.0], [-1.2097, -0.6985, 0.0], [-1.2097, 0.6985, 0.0],
]


# ---------------------------------------------------------------------------
# Topology
# ---------------------------------------------------------------------------
def test_h2_is_tractable() -> None:
    result = TopologicalClassifier().run(H2)
    assert result["tractable"] is True, result["reason"]
    assert result["metrics"]["beta_0"] == 1
    assert result["metrics"]["beta_1"] == 0
    assert "no topological obstructions" in result["reason"]


def test_ring_is_obstructed() -> None:
    result = TopologicalClassifier().run(BENZENE_RING)
    assert result["tractable"] is False, result["reason"]
    assert result["metrics"]["beta_1"] >= 1


def test_dissociated_geometry_is_obstructed() -> None:
    result = TopologicalClassifier().run([[0.0, 0.0, 0.0], [0.0, 0.0, 5.0]])
    assert result["tractable"] is False
    assert result["metrics"]["beta_0"] == 2


def test_persistence_handles_degenerate_clouds() -> None:
    single = compute_persistence([[0.0, 0.0, 0.0]])
    assert single.betti_numbers[0] == 0 and single.n_points == 1
    coincident = compute_persistence([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    assert coincident.betti_numbers[1] == 0


# ---------------------------------------------------------------------------
# Commitment scheme
# ---------------------------------------------------------------------------
def test_commitment_round_trip_and_tamper(tmp_path: Path) -> None:
    scheme = LatticeCommitmentScheme(keystore_path=tmp_path / "keystore.json")
    payload = {"energy_ha": -1.137274, "molecule": "H2", "num_qubits": 2}
    record = scheme.commit(payload)

    assert record.commitment_hash.startswith("0x") and len(record.commitment_hash) == 66
    assert scheme.verify(record.proof_id, payload) is True
    # Reveal-phase verification with the salt supplied out of band.
    assert scheme.verify(record.proof_id, payload, salt_hex=record.salt_hex) is True
    # Tampering with the published energy must fail.
    tampered = dict(payload, energy_ha=-1.137275)
    assert scheme.verify(record.proof_id, tampered) is False
    # A different salt must fail.
    assert scheme.verify(record.proof_id, payload, salt_hex="00" * 32) is False
    # Unknown proof ids must fail rather than raise.
    assert scheme.verify("deadbeef", payload) is False


def test_commitment_hides_salt_and_rejects_non_finite(tmp_path: Path) -> None:
    scheme = LatticeCommitmentScheme(keystore_path=tmp_path / "keystore.json")
    record = scheme.commit({"energy_ha": -1.0})
    assert "salt_hex" not in record.to_dict()
    assert record.to_dict(include_salt=True)["salt_hex"] == record.salt_hex
    try:
        scheme.commit({"energy_ha": float("nan")})
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("non-finite energies must not be certifiable")


def test_two_identical_payloads_commit_differently(tmp_path: Path) -> None:
    scheme = LatticeCommitmentScheme(keystore_path=tmp_path / "keystore.json")
    first = scheme.commit({"energy_ha": -1.0})
    second = scheme.commit({"energy_ha": -1.0})
    assert first.commitment_hash != second.commitment_hash


# ---------------------------------------------------------------------------
# ZNE building blocks
# ---------------------------------------------------------------------------
def _parameterised_circuit() -> QuantumCircuit:
    from qiskit.circuit import Parameter

    theta = Parameter("theta")
    circuit = QuantumCircuit(3)
    circuit.h(0)
    circuit.rx(theta, 0)
    circuit.cx(0, 1)
    circuit.cx(1, 2)
    return circuit


def test_fold_circuit_preserves_unitary() -> None:
    circuit = _parameterised_circuit()
    reference = Operator(circuit.assign_parameters({circuit.parameters[0]: 0.31}))
    for scale in (1, 3, 5):
        folded = fold_circuit(circuit, scale)
        assert folded.num_parameters == circuit.num_parameters
        value = Operator(folded.assign_parameters({folded.parameters[0]: 0.31}))
        assert value.equiv(reference), f"scale {scale} changed the unitary"
        assert len(folded.data) == scale * len(circuit.data), f"scale {scale} gate count"


def test_fold_circuit_rejects_even_and_zero_scales() -> None:
    circuit = _parameterised_circuit()
    for bad in (0, 2, 4, -3):
        try:
            fold_circuit(circuit, bad)
        except ValueError:
            continue
        raise AssertionError(f"scale {bad} should have been rejected")


# ---------------------------------------------------------------------------
# Hamiltonian construction (PySCF-free fallback driver)
# ---------------------------------------------------------------------------
# Literature anchors for H2 in STO-3G at R = 0.735 A: RHF -1.1167 Ha and FCI
# -1.1373 Ha.  The tolerances are loose enough for the 0.7348 A geometry used
# here yet tight enough that any error in the integrals (or in the
# chemists'->physicists' index conversion, or the parity mapping) shows up.
H2_RHF_REFERENCE = -1.1167
H2_FCI_REFERENCE = -1.1373


def _h2_hamiltonian() -> object:
    geometry = build_geometry("[H][H]", bond_length=0.7348)
    return build_hamiltonian(geometry, driver="minimal")


def test_h2_hamiltonian_matches_literature() -> None:
    ham = _h2_hamiltonian()
    assert ham.num_spatial_orbitals == 2
    assert ham.num_particles == (1, 1)
    # The specification's central claim about the mapping: 4 spin-orbitals in,
    # 2 qubits out via parity mapping + two-qubit reduction.
    assert ham.num_qubits == 2, ham.qubit_operator
    assert abs(ham.reference_energy - H2_RHF_REFERENCE) < 2e-3, ham.reference_energy
    exact = ham.exact_ground_state_energy()
    assert abs(exact - H2_FCI_REFERENCE) < 2e-3, exact
    # A 2-electron, 2-orbital singlet has exactly these five Pauli strings.
    labels = sorted(str(pauli) for pauli in ham.qubit_operator.paulis)
    assert labels == ["II", "IZ", "XX", "ZI", "ZZ"], labels


def test_h2_atomisation_limit() -> None:
    # Stretched H2 must tend to twice the STO-3G hydrogen atom energy
    # (-0.4666 Ha each).  This checks the two-electron integrals independently
    # of the equilibrium numbers above.
    ham = build_hamiltonian(build_geometry("[H][H]", bond_length=3.0), driver="minimal")
    assert abs(ham.exact_ground_state_energy() - 2 * -0.4666) < 5e-3


def test_uccsd_vqe_reaches_exact_ground_state() -> None:
    """End-to-end quantum check: UCCSD + VQE must find the FCI energy."""
    ham = _h2_hamiltonian()
    bundle = build_uccsd_ansatz(ham)
    assert bundle.label == "Symmetry-Protected UCCSD"
    assert bundle.num_qubits == 2
    result = run_vqe(bundle, ham, optimizer="L_BFGS_B", seed=3)
    assert result.converged, result.status
    assert result.barren_plateau is False
    assert result.initial_gradient_norm and result.initial_gradient_norm > 1e-3
    gap = abs(result.total_energy - ham.exact_ground_state_energy())
    assert gap < 1e-6, f"VQE missed the exact energy by {gap:.2e} Ha"


def test_folding_is_invisible_to_the_noiseless_energy() -> None:
    """Folding must not change the physics: with no errors attached, the
    expectation value at every noise scale equals the bare VQE value."""
    from qiskit_aer.noise import NoiseModel

    ham = _h2_hamiltonian()
    bundle = build_uccsd_ansatz(ham)
    result = run_vqe(bundle, ham, estimator=make_statevector_estimator(), seed=3)
    parameters = dict(zip(list(bundle.ansatz.parameters), result.optimal_parameters))

    for scale in (1, 3, 5):
        folded = fold_circuit(bundle.ansatz, scale).assign_parameters(parameters)
        value = _evaluate(NoiseModel(), None, 11, folded, ham.qubit_operator)
        assert abs(value - result.optimal_energy) < 1e-9, (scale, value)


def test_noise_scaling_is_monotonic_and_deterministic() -> None:
    """Exact (density-matrix) evaluation must be deterministic and must show
    the energy degrading as the folding scale - and therefore the noise - grows."""
    ham = _h2_hamiltonian()
    bundle = build_uccsd_ansatz(ham)
    result = run_vqe(bundle, ham, seed=3)
    parameters = dict(zip(list(bundle.ansatz.parameters), result.optimal_parameters))
    noise = build_depolarizing_noise_model(1e-3, 1e-2)

    def energy(scale: int) -> float:
        folded = fold_circuit(bundle.ansatz, scale).assign_parameters(parameters)
        return _evaluate(noise, None, 5, folded, ham.qubit_operator)

    first = [energy(s) for s in (1, 3, 5)]
    second = [energy(s) for s in (1, 3, 5)]
    assert first == second, "density-matrix evaluation must be deterministic"
    # More folding means more noise, so the energy must rise monotonically
    # towards the noiseless value it started from (-1.857... < -1.818...).
    assert first[0] < first[1] < first[2], first
    assert all(value > result.optimal_energy for value in first)


#: Real ZNE points captured from an engine run (H2, p1=1e-3, p2=1e-2,
#: exact evaluation).  The exponential fit through them diverges: with three
#: points it is exactly determined and its intercept lands ~0.8 Ha away.
MEASURED_SCALE_ENERGIES = [-1.098237, -1.025695, -0.960023]


def test_exponential_fit_rejected_on_real_zne_points() -> None:
    """The unguarded exponential fit turns 0.138 Ha of drift into 0.8 Ha of
    'correction'.  It must be refused in favour of the linear fit."""
    fit = extrapolate_to_zero_noise([1, 3, 5], MEASURED_SCALE_ENERGIES, "exponential")
    assert fit["method"] == "linear", fit
    assert "note" in fit
    assert abs(fit["extrapolated_value"] + 1.131645) < 1e-3, fit


def test_extrapolation_guard_rejects_non_monotonic_richardson() -> None:
    """A polynomial correction larger than the observed drift is unphysical."""
    # Energy that goes down then *up* between scales cannot come from a
    # monotonically growing error channel.
    fit = extrapolate_to_zero_noise([1, 3, 5], [-1.0, -1.1, -0.9], method="richardson")
    assert fit["method"] == "linear", fit
    assert "note" in fit


def test_richardson_accepted_on_monotonic_data() -> None:
    """On physical (monotone) data the richer fit is kept, not discarded."""
    fit = extrapolate_to_zero_noise([1, 3, 5], MEASURED_SCALE_ENERGIES, "richardson")
    assert fit["method"] == "richardson", fit
    assert fit["extrapolated_value"] < MEASURED_SCALE_ENERGIES[0]


def test_linear_extrapolation_recovers_noiseless_value() -> None:
    true_energy, coefficient = -1.137, 0.021
    scales = [1, 3, 5]
    energies = [true_energy + coefficient * s for s in scales]
    fit = extrapolate_to_zero_noise(scales, energies, method="linear")
    assert abs(fit["extrapolated_value"] - true_energy) < 1e-9
    assert fit["r_squared"] > 0.999


def test_exponential_extrapolation_falls_back_when_unphysical() -> None:
    # Energy *increases* away from lambda = 0: a decaying exponential is wrong,
    # so the fit must fall back to a linear extrapolation instead of inventing
    # a value.
    fit = extrapolate_to_zero_noise([1, 3, 5], [-1.1, -1.2, -1.3], method="exponential")
    assert fit["method"] == "linear"
    assert "note" in fit


# ---------------------------------------------------------------------------
# Barren-plateau detection
# ---------------------------------------------------------------------------
def _flat_problem() -> tuple[SimpleNamespace, SimpleNamespace]:
    """A 1-parameter circuit whose cost is a constant (a true flat landscape)."""
    from qiskit.circuit import Parameter

    theta = Parameter("t")
    circuit = QuantumCircuit(1)
    circuit.rx(theta, 0)

    bundle = SimpleNamespace(
        ansatz=circuit,
        initial_state=QuantumCircuit(1),
        num_parameters=1,
        num_qubits=1,
        parameters=[theta],
        warnings=[],
        label="test",
    )
    hamiltonian = SimpleNamespace(
        qubit_operator=2.0 * SparsePauliOp.from_list([("I", 1.0)]),  # constant cost
        electronic_energy_offset=0.0,
    )
    return bundle, hamiltonian


def test_barren_plateau_is_raised_for_flat_landscape() -> None:
    bundle, hamiltonian = _flat_problem()
    config = BarrenPlateauConfig(gradient_threshold=1e-7, patience=3, monitor_steps=5)
    try:
        run_vqe(bundle, hamiltonian, optimizer="L_BFGS_B", bp_config=config)
    except BarrenPlateauError as exc:
        assert exc.gradient_norm < 1e-7
        assert len(exc.gradient_history) >= 3
        return
    raise AssertionError("flat landscape must raise BarrenPlateauError")


def test_non_flat_landscape_does_not_raise() -> None:
    from qiskit.circuit import Parameter

    theta = Parameter("t")
    circuit = QuantumCircuit(1)
    circuit.ry(theta, 0)
    bundle = SimpleNamespace(
        ansatz=circuit, initial_state=QuantumCircuit(1), num_parameters=1,
        num_qubits=1, parameters=[theta], warnings=[], label="test",
    )
    hamiltonian = SimpleNamespace(
        qubit_operator=SparsePauliOp.from_list([("Z", 1.0)]),
        electronic_energy_offset=0.0,
    )
    result = run_vqe(
        bundle, hamiltonian, optimizer="L_BFGS_B",
        bp_config=BarrenPlateauConfig(monitor_steps=5), seed=1,
    )
    assert result.barren_plateau is False
    assert result.optimal_energy <= -0.999  # |0> has <Z> = -1
    # The per-iteration monitor must actually be wired up: a baseline record
    # plus at least one optimiser step.
    assert len(result.energy_history) >= 2, result.energy_history
    assert len(result.gradient_history) == len(result.energy_history)
    assert result.initial_gradient_norm is not None
    assert result.initial_gradient_norm > 1e-7, result.initial_gradient_norm


# ---------------------------------------------------------------------------
# Report layout (the specification's output contract)
# ---------------------------------------------------------------------------
def test_report_matches_specified_layout() -> None:
    passed = {
        "tractable": True,
        "reason": "Low persistence entropy, no topological obstructions.",
    }
    zne = SimpleNamespace(raw_total_energy=-1.135103, mitigated_total_energy=-1.137232)
    report = format_report(
        molecule_name="H2",
        bond_length=0.7348,
        topology=passed,
        ansatz_label="Symmetry-Protected UCCSD",
        optimizer_name="L_BFGS_B",
        iterations=6,
        convergence="SUCCESS",
        barren_plateau=False,
        zne=zne,
        fci_energy=-1.137274,
        error_reduction=85.5,
        commitment_hash="0x" + "ab" * 32,
    )
    expected = "\n".join([
        "=== Q-TMCE ENGINE REPORT ===",
        "",
        "Molecule: H2 (Bond Length: 0.7348 Angstrom)",
        "",
        "[1] TOPOLOGICAL FILTER",
        "Status: PASS (Tractable)",
        "Reason: Low persistence entropy, no topological obstructions.",
        "",
        "[2] QUANTUM SIMULATION",
        "Ansatz: Symmetry-Protected UCCSD",
        "Optimizer: L_BFGS_B",
        "Iterations: 6",
        "Convergence: SUCCESS",
        "Barren Plateau Detected: NO",
        "",
        "[3] ERROR MITIGATION",
        "Raw Estimate: -1.135103 Ha",
        "Mitigated Estimate: -1.137232 Ha",
        "Exact Classical (FCI): -1.137274 Ha",
        "Error Reduction: 85.50%",
        "",
        "[4] CRYPTOGRAPHIC CERTIFICATION",
        f"ZKP Commitment Hash: 0x{'ab' * 32}",
        "Status: PROOF GENERATED",
        "",
        "=== CONCLUSION ===",
        "",
        "The molecule is certified as computationally tractable.",
        "The ground state energy has been verified via symmetry-protection and error mitigation.",
        "A cryptographic proof of this result has been generated.",
        "",
        "===============================",
    ])
    assert report == expected, "\n--- got ---\n" + report


def test_report_states_what_did_not_run() -> None:
    report = format_report(
        molecule_name="H2", bond_length=0.7348,
        topology={"tractable": False, "reason": "Topological obstruction detected."},
        ansatz_label="n/a", optimizer_name="n/a", iterations=0,
        convergence="NOT RUN", barren_plateau=False, skipped_reason="molecule rejected",
    )
    assert "Status: FAIL (Obstructed)" in report
    assert "Status: NOT RUN (molecule rejected)" in report
    assert "Status: NOT GENERATED" in report
    assert "rejected by the topological filter" in report


if __name__ == "__main__":  # pragma: no cover - plain-python runner
    import inspect
    import tempfile
    import traceback

    failures = 0
    with tempfile.TemporaryDirectory() as tmp:
        for name, function in sorted(globals().items()):
            if not name.startswith("test_") or not callable(function):
                continue
            kwargs = {"tmp_path": Path(tmp)} if "tmp_path" in inspect.signature(function).parameters else {}
            try:
                function(**kwargs)
                print(f"PASS  {name}")
            except Exception:
                failures += 1
                print(f"FAIL  {name}")
                traceback.print_exc()
    print(f"\n{len([n for n in globals() if n.startswith('test_')]) - failures} passed, {failures} failed")
    raise SystemExit(1 if failures else 0)
