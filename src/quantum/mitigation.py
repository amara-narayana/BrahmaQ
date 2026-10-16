"""Zero-Noise Extrapolation (ZNE) error mitigation.

Method
------
1.  Build a depolarizing noise model with ``qiskit_aer.noise``.
2.  Evaluate the energy at the noise scales ``lambda in {1, 3, 5}``.  The scale
    is realised by *gate folding*: the circuit ``U`` is replaced by

        lambda = 1 :  U
        lambda = 3 :  U (U^dagger U)          = U
        lambda = 5 :  U (U^dagger U)(U^dagger U) = U

    so that the logical action is unchanged while the physical gate count --
    and therefore the accumulated noise -- grows linearly in ``lambda``.
3.  Fit ``E(lambda)`` (linear, polynomial/Richardson, or exponential) and
    extrapolate to ``lambda = 0``.

Because the folded circuits are logically identical to the original, the
extrapolated value estimates the *noiseless* energy of the same parameters.

Determinism
-----------
With ``shots=None`` and the density-matrix method, Aer returns analytic
expectation values that include the noise channel exactly -- no shot noise to
confound the fit.  ``shots`` can be set explicitly to emulate sampling instead
(``config.json: mitigation.shots``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

from .hamiltonian_builder import HamiltonianData

__all__ = [
    "ZNEResult",
    "build_depolarizing_noise_model",
    "fold_circuit",
    "extrapolate_to_zero_noise",
    "run_zne",
]


# ---------------------------------------------------------------------------
# Noise model
# ---------------------------------------------------------------------------
_ONE_QUBIT_GATES = ("id", "u1", "u2", "u3", "u", "p", "x", "y", "z",
                    "h", "s", "sdg", "sx", "sxdg", "t", "tdg", "rx", "ry", "rz")
_TWO_QUBIT_GATES = ("cx", "cnot", "cy", "cz", "ch", "swap", "iswap", "rxx",
                    "ryy", "rzz", "rzx", "ecr", "dcx")


def build_depolarizing_noise_model(
    single_qubit_error: float = 1e-3,
    two_qubit_error: float = 1e-2,
    readout_error: float = 0.0,
) -> Any:
    """Depolarizing noise model over the standard 1- and 2-qubit gate sets.

    ``single_qubit_error``/``two_qubit_error`` are the total depolarizing
    probabilities attached to each gate; ``readout_error`` (``0`` by default)
    would also attach a symmetric bit-flip channel to measurements.
    """
    from qiskit_aer.noise import NoiseModel, depolarizing_error

    noise_model = NoiseModel()
    if single_qubit_error > 0:
        noise_model.add_all_qubit_quantum_error(
            depolarizing_error(float(single_qubit_error), 1), list(_ONE_QUBIT_GATES)
        )
    if two_qubit_error > 0:
        noise_model.add_all_qubit_quantum_error(
            depolarizing_error(float(two_qubit_error), 2), list(_TWO_QUBIT_GATES)
        )
    if readout_error > 0:
        noise_model.add_all_qubit_readout_error(
            [[1.0 - readout_error, readout_error], [readout_error, 1.0 - readout_error]]
        )
    return noise_model


# ---------------------------------------------------------------------------
# Circuit folding
# ---------------------------------------------------------------------------
def fold_circuit(circuit: QuantumCircuit, scale: int) -> QuantumCircuit:
    """Realise noise scale ``scale`` by folding ``U -> U (U^dag U)^k``.

    ``scale`` must be an odd integer (``2k + 1``).  The folded circuit has the
    same unitary as ``circuit`` but ``scale`` times as many gates, and it keeps
    the original ``Parameter`` objects so parameters can still be assigned.
    """
    if scale < 1 or scale % 2 == 0:
        raise ValueError(f"noise scale must be an odd integer >= 1, got {scale!r}")
    if scale == 1:
        return circuit.copy()
    folded = circuit.copy()
    inverse = circuit.inverse()
    for _ in range((scale - 1) // 2):
        folded = folded.compose(inverse).compose(circuit)
    return folded


# ---------------------------------------------------------------------------
# Extrapolation
# ---------------------------------------------------------------------------
def extrapolate_to_zero_noise(
    scales: Sequence[float],
    energies: Sequence[float],
    method: str = "linear",
) -> Dict[str, Any]:
    """Fit ``E(lambda)`` and return the ``lambda -> 0`` intercept.

    ``method`` is one of

    * ``"linear"``     -- ordinary least squares on ``(lambda, E)`` (default;
      the right choice for three equally spaced scales),
    * ``"richardson"`` -- polynomial of degree ``n - 1`` through all points,
    * ``"exponential"``-- ``a + b exp(-c lambda)``; falls back to a linear fit
      when the curve fit does not converge or the intercept is unphysical.
    """
    x = np.asarray(scales, dtype=float)
    y = np.asarray(energies, dtype=float)
    if x.size != y.size:
        raise ValueError("scales and energies must have the same length")
    if x.size < 2:
        raise ValueError("at least two noise scales are required to extrapolate")
    if np.any(np.diff(x) <= 0):
        raise ValueError(f"noise scales must be strictly increasing, got {list(x)}")

    drift = abs(float(y[-1] - y[0]))

    if method == "linear" or (method == "exponential" and x.size < 3):
        slope, intercept = np.polyfit(x, y, 1)
        predicted = slope * x + intercept
        return {
            "method": "linear",
            "requested_method": method,
            "extrapolated_value": float(intercept),
            "coefficients": {"slope": float(slope), "intercept": float(intercept)},
            "fit_energies": predicted.tolist(),
            "r_squared": _r_squared(y, predicted),
        }

    if method == "richardson" and x.size == 3:
        # A quadratic through three points is exactly determined; the guard
        # below still applies, since an exactly-fitted polynomial can run away.
        coefficients = np.polyfit(x, y, 2)
        predicted = np.polyval(coefficients, x)
        value = float(np.polyval(coefficients, 0.0))
        if abs(value - y[0]) > 1.5 * drift + 1e-9:
            fallback = extrapolate_to_zero_noise(x, y, method="linear")
            fallback["note"] = (
                "richardson fit rejected: extrapolated correction exceeds the "
                f"observed noise-induced drift ({drift:.3e} Ha)"
            )
            return fallback
        return {
            "method": "richardson",
            "requested_method": method,
            "extrapolated_value": value,
            "coefficients": [float(c) for c in coefficients],
            "fit_energies": predicted.tolist(),
            "r_squared": _r_squared(y, predicted),
        }

    if method == "richardson":
        coefficients = np.polyfit(x, y, x.size - 1)
        predicted = np.polyval(coefficients, x)
        value = float(np.polyval(coefficients, 0.0))
        if abs(value - y[0]) > 1.5 * drift + 1e-9:
            fallback = extrapolate_to_zero_noise(x, y, method="linear")
            fallback["note"] = (
                "richardson fit rejected: extrapolated correction exceeds the "
                f"observed noise-induced drift ({drift:.3e} Ha)"
            )
            return fallback
        return {
            "method": "richardson",
            "requested_method": method,
            "extrapolated_value": value,
            "coefficients": [float(c) for c in coefficients],
            "fit_energies": predicted.tolist(),
            "r_squared": _r_squared(y, predicted),
        }

    if method == "exponential":
        from scipy.optimize import curve_fit

        def model(lam: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
            return a + b * np.exp(-c * lam)

        try:
            amplitude_guess = float(y[0] - y[-1]) or 1e-3
            popt, _ = curve_fit(
                model, x, y,
                p0=(float(y[-1]), amplitude_guess, 1.0),
                maxfev=20000,
            )
            predicted = model(x, *popt)
            result: Dict[str, Any] = {
                "method": "exponential",
                "requested_method": method,
                "extrapolated_value": float(popt[0]),
                "coefficients": {"asymptote": float(popt[0]),
                                 "amplitude": float(popt[1]),
                                 "rate": float(popt[2])},
                "fit_energies": predicted.tolist(),
                "r_squared": _r_squared(y, predicted),
            }
            # A noise-induced energy shift must decay with lambda, and ZNE must
            # not invent a correction larger than the drift it observed.  With
            # three points the exponential is exactly determined, so this guard
            # is what keeps an unstable fit out of the report.
            if popt[2] <= 0 or abs(result["extrapolated_value"] - y[0]) > 1.5 * drift + 1e-9:
                fallback = extrapolate_to_zero_noise(x, y, method="linear")
                fallback["note"] = "exponential fit rejected as unphysical; linear fit used"
                return fallback
            return result
        except Exception as exc:  # pragma: no cover - depends on data
            fallback = extrapolate_to_zero_noise(x, y, method="linear")
            fallback["note"] = (
                f"exponential fit failed ({type(exc).__name__}: {exc}); linear fit used"
            )
            return fallback

    raise ValueError(
        f"unknown extrapolation method {method!r}; "
        "use 'linear', 'richardson' or 'exponential'"
    )


def _r_squared(observed: np.ndarray, predicted: np.ndarray) -> float:
    residual = float(np.sum((observed - predicted) ** 2))
    total = float(np.sum((observed - observed.mean()) ** 2))
    if total <= 0:
        return 1.0 if residual <= 0 else 0.0
    return 1.0 - residual / total


# ---------------------------------------------------------------------------
# ZNE driver
# ---------------------------------------------------------------------------
@dataclass
class ZNEResult:
    """Noisy scale points, the fit, and the mitigated estimate."""

    raw_estimate: float
    mitigated_estimate: float
    scales: List[int]
    energies: List[float]
    fit: Dict[str, Any]
    energy_offset: float
    mitigated_total_energy: float
    raw_total_energy: float
    noiseless_estimate: Optional[float] = None
    reoptimized: bool = False
    per_scale_parameters: List[List[float]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def error_reduction_percent(self, reference: float) -> float:
        """Fraction of the residual error removed by mitigation (percent).

        ``raw`` is the single-scale (lambda = 1) noisy estimate, so this is
        exactly the improvement ZNE buys over running the circuit as-is.
        ``reference`` is the exact (FCI) energy.
        """
        raw_gap = abs(self.raw_total_energy - reference)
        mitigated_gap = abs(self.mitigated_total_energy - reference)
        if raw_gap <= 0:
            return 0.0
        return 100.0 * (raw_gap - mitigated_gap) / raw_gap

    def to_dict(self) -> Dict[str, Any]:
        return {
            "raw_estimate": self.raw_estimate,
            "mitigated_estimate": self.mitigated_estimate,
            "raw_total_energy": self.raw_total_energy,
            "mitigated_total_energy": self.mitigated_total_energy,
            "energy_offset": self.energy_offset,
            "scales": self.scales,
            "energies": self.energies,
            "fit": self.fit,
            "noiseless_estimate": self.noiseless_estimate,
            "reoptimized": self.reoptimized,
            "per_scale_parameters": self.per_scale_parameters,
            "warnings": self.warnings,
        }


def _aer_estimator(noise_model: Any, shots: Optional[int], seed: int) -> Any:
    """Aer primitive that carries the noise model into expectation values.

    Used for the sampled route (``shots`` set) and by the optional re-optimising
    branch, which needs a primitive rather than a bare number.
    """
    from qiskit_aer.primitives import Estimator as AerEstimator

    backend_options: Dict[str, Any] = {"noise_model": noise_model, "seed_simulator": seed}
    run_options: Dict[str, Any] = {"seed_simulator": seed}
    if shots is not None:
        backend_options["method"] = "density_matrix"
        run_options["shots"] = shots
    return AerEstimator(
        backend_options=backend_options,
        # The default transpilation optimises the circuit, which can cancel the
        # U^dag U folds and make every noise scale look identical.  Translation
        # without optimisation keeps the noise scaling intact.
        transpile_options={"optimization_level": 0},
        run_options=run_options,
        approximation=False,
    )


def run_zne(
    hamiltonian: HamiltonianData,
    ansatz: Any,
    optimal_parameters: Sequence[float],
    scales: Sequence[int] = (1, 3, 5),
    single_qubit_error: float = 1e-3,
    two_qubit_error: float = 1e-2,
    readout_error: float = 0.0,
    shots: Optional[int] = None,
    seed: int = 42,
    extrapolation: str = "linear",
    noise_model: Optional[Any] = None,
    noiseless_estimate: Optional[float] = None,
    reoptimize_with: Optional[str] = None,
    optimizer_options: Optional[Dict[str, Any]] = None,
) -> ZNEResult:
    """Execute the ZNE protocol at the supplied noise scales.

    The parameters found by the noiseless VQE are held fixed while the folded
    circuit is evaluated under the depolarizing model (the standard ZNE
    setting).  Setting ``reoptimize_with="SPSA"`` instead re-optimises at every
    noise scale and folds the converged energies, which is the noisy-VQE
    variant.
    """
    operator: SparsePauliOp = hamiltonian.qubit_operator
    noise_model = noise_model or build_depolarizing_noise_model(
        single_qubit_error, two_qubit_error, readout_error
    )
    scales = [int(s) for s in scales]
    warnings_: List[str] = []
    energies: List[float] = []
    per_scale_parameters: List[List[float]] = []

    # Bind parameters through the ansatz's own Parameter objects: the folded
    # circuit holds the same objects, and this keeps binding independent of any
    # ordering that ``QuantumCircuit.parameters`` might report.
    parameters = list(getattr(ansatz, "parameters", []))
    phi = [float(v) for v in np.asarray(optimal_parameters, dtype=float).reshape(-1)]
    if len(parameters) != len(phi):
        raise ValueError(
            f"got {len(phi)} parameter values for {len(parameters)} circuit parameters"
        )

    uncovered = _uncovered_gates(ansatz, noise_model)
    if uncovered:
        warnings_.append(
            "noise model has no error registered for "
            f"{sorted(uncovered)}; those gates run noiselessly"
        )

    for scale in scales:
        folded = fold_circuit(ansatz, scale)
        phi = [float(v) for v in np.asarray(optimal_parameters, dtype=float).reshape(-1)]

        if reoptimize_with:
            # Local imports: mitigation is normally downstream of the simulator,
            # and this branch reverses that dependence.
            from .ansatz import AnsatzBundle
            from .simulator import BarrenPlateauConfig, run_vqe

            bundle = AnsatzBundle(
                ansatz=folded,
                initial_state=folded,
                num_parameters=int(folded.num_parameters),
                num_qubits=int(folded.num_qubits),
                num_spatial_orbitals=hamiltonian.num_spatial_orbitals,
                num_particles=hamiltonian.num_particles,
                mapper_name=type(hamiltonian.mapper).__name__,
                label=f"folded x{scale}",
            )
            try:
                noisy = run_vqe(
                    bundle,
                    hamiltonian,
                    optimizer=reoptimize_with,
                    initial_point=phi,
                    estimator=_aer_estimator(noise_model, shots, seed),
                    bp_config=BarrenPlateauConfig(monitor_steps=0),
                    optimizer_options=optimizer_options,
                    seed=seed,
                )
                energy = noisy.optimal_energy
                phi = [float(v) for v in np.asarray(noisy.optimal_parameters).reshape(-1)]
                warnings_.extend(noisy.warnings)
            except Exception as exc:  # pragma: no cover - noisy path
                warnings_.append(
                    f"re-optimisation at scale {scale} failed "
                    f"({type(exc).__name__}: {exc}); using the noiseless parameters"
                )
                energy = _evaluate(
                    noise_model, shots, seed,
                    folded.assign_parameters(dict(zip(parameters, phi))), operator,
                )
        else:
            energy = _evaluate(
                noise_model, shots, seed,
                folded.assign_parameters(dict(zip(parameters, phi))), operator,
            )

        energies.append(float(energy))
        per_scale_parameters.append(phi)

    fit = extrapolate_to_zero_noise(scales, energies, method=extrapolation)
    offset = hamiltonian.electronic_energy_offset
    raw = float(energies[scales.index(1)]) if 1 in scales else float(energies[0])
    if 1 not in scales:
        warnings_.append("no unsuppressed (lambda = 1) point: raw estimate taken from the first scale")

    return ZNEResult(
        raw_estimate=raw,
        mitigated_estimate=float(fit["extrapolated_value"]),
        scales=scales,
        energies=energies,
        fit=fit,
        energy_offset=offset,
        mitigated_total_energy=float(fit["extrapolated_value"] + offset),
        raw_total_energy=float(raw + offset),
        noiseless_estimate=noiseless_estimate,
        reoptimized=bool(reoptimize_with),
        per_scale_parameters=per_scale_parameters,
        warnings=warnings_,
    )


#: Above this width the exact density-matrix path is impractical, so the
#: sampled Aer primitive is used even when ``shots is None``.
_EXACT_MAX_QUBITS = 12


def _uncovered_gates(circuit: QuantumCircuit, noise_model: Any) -> set:
    """Gate names used by ``circuit`` that the noise model leaves noiseless."""
    try:
        covered = set(noise_model.basis_gates)  # gates carrying any error
    except Exception:  # pragma: no cover - API variation
        return set()
    used = {
        instruction.operation.name
        for instruction in circuit.data
        if not getattr(instruction.operation, "_directive", False)
    }
    return {name for name in used if name not in covered}


def _expectation_density_matrix(
    noise_model: Any, circuit: QuantumCircuit, operator: SparsePauliOp
) -> float:
    """Exact ``Tr[rho H]`` from density-matrix evolution (no sampling).

    ``AerEstimator`` cannot do this: with ``shots=None`` and
    ``approximation=False`` it silently falls back to 1024 shots, and with
    ``approximation=True`` it ignores the noise model altogether.  Simulating
    the (small) circuit and taking the trace directly keeps the ZNE points free
    of shot noise, which is what the extrapolation fit relies on.
    """
    from qiskit import transpile
    from qiskit_aer import AerSimulator

    backend = AerSimulator(method="density_matrix", noise_model=noise_model)
    # ``optimization_level=0`` is essential: it translates the ansatz's custom
    # instructions into basis gates (Aer cannot assemble ``EvolvedOps``
    # directly) *without* running an optimisation pass that could cancel the
    # U^dag U folds and silently undo the noise scaling.
    probe = transpile(circuit, backend, optimization_level=0)
    probe.save_density_matrix()
    # ``shots=1`` only satisfies the sampler; saved quantum states are exact.
    result = backend.run(probe, shots=1).result()
    saved = result.data()["density_matrix"]
    density = np.asarray(getattr(saved, "data", saved), dtype=complex)
    matrix = np.asarray(operator.to_matrix(sparse=False), dtype=complex)
    if density.shape != matrix.shape:
        raise RuntimeError(
            f"density matrix {density.shape} does not match operator {matrix.shape}"
        )
    return float(np.real(np.trace(density @ matrix)))


def _evaluate(
    noise_model: Any,
    shots: Optional[int],
    seed: int,
    bound_circuit: QuantumCircuit,
    operator: SparsePauliOp,
) -> float:
    """Expectation value of ``operator`` for a parameter-bound noisy circuit.

    ``shots is None`` (the default) requests the exact density-matrix route;
    an integer emulates finite sampling through the Aer primitive instead.
    """
    if shots is None and operator.num_qubits <= _EXACT_MAX_QUBITS:
        return _expectation_density_matrix(noise_model, bound_circuit, operator)
    estimator = _aer_estimator(noise_model, shots, seed)
    result = estimator.run([bound_circuit], [operator]).result()
    return float(np.real(result.values[0]))
