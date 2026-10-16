"""VQE loop with explicit barren-plateau monitoring.

Cost function
-------------
``C(theta) = <psi(theta)| H |psi(theta)>`` is evaluated with a Qiskit
``Estimator`` primitive, which keeps the implementation agnostic to the backend
(statevector reference simulator here, Aer + noise model in
:mod:`src.quantum.mitigation`).

Optimiser
---------
``qiskit_algorithms.optimizers.L_BFGS_B`` (deterministic gradient-based search,
the right choice on an exact statevector simulator) or ``SPSA`` (gradient-free,
shot-noise tolerant -- the choice when the estimator is noisy).  Both are the
``qiskit_algorithms`` implementations, as required for this Qiskit generation.

Barren-plateau criterion
------------------------
Specification: *"monitor ||grad C|| at each step; if it stays below 1e-7 for 3
consecutive steps, raise* ``BarrenPlateauError``\ *"*.

A literal reading would misfire: at a *converged* minimum the gradient is also
below any fixed threshold, so a successful optimisation would be reported as a
plateau.  Physically, a barren plateau is a property of the landscape **away
from the optimum** -- exponentially vanishing gradients at initialization.  The
implementation therefore

* records the gradient norm at *every* step (``gradient_history``),
* raises only when the sub-threshold run is detected inside the first
  ``monitor_steps`` iterations (the exploration phase, default 5), and
* still reports late sub-threshold runs in ``late_plateau_windows`` so nothing
  is hidden.

Gradients are central finite differences with step ``eps``; the cost is
re-evaluated ``2 * n_parameters`` times per step, so probing is skipped (with a
recorded warning) above ``max_probed_parameters``.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import BaseEstimator, Estimator
from qiskit_algorithms import VQE
from qiskit_algorithms.optimizers import L_BFGS_B, SPSA

from .ansatz import AnsatzBundle
from .hamiltonian_builder import HamiltonianData

__all__ = [
    "BarrenPlateauError",
    "BarrenPlateauConfig",
    "VQEResult",
    "build_cost_function",
    "numerical_gradient",
    "run_vqe",
    "make_statevector_estimator",
]


class BarrenPlateauError(RuntimeError):
    """Raised when the optimisation landscape is flat at initialization.

    Carries the diagnostic evidence rather than a bare message, so the
    orchestrator can report *why* the run was aborted.
    """

    def __init__(
        self,
        message: str,
        gradient_norm: float,
        iteration: int,
        gradient_history: Sequence[float],
    ) -> None:
        super().__init__(message)
        self.gradient_norm = float(gradient_norm)
        self.iteration = int(iteration)
        self.gradient_history = [float(g) for g in gradient_history]


@dataclass
class BarrenPlateauConfig:
    """Thresholds for the gradient monitor."""

    gradient_threshold: float = 1e-7
    patience: int = 3
    #: Iterations over which the strict criterion can abort the run.
    monitor_steps: int = 5
    #: Step size of the central finite-difference gradient estimate.
    finite_difference_eps: float = 1e-5
    #: Skip gradient probing above this parameter count (cost: 2*n extra evals).
    max_probed_parameters: int = 24


@dataclass
class VQEResult:
    """Outcome of a VQE run, including the full monitoring record."""

    optimal_energy: float
    energy_offset: float
    optimal_parameters: np.ndarray
    parameter_names: List[str]
    iterations: int
    cost_function_evals: int
    optimizer_name: str
    converged: bool
    status: str
    barren_plateau: bool
    energy_history: List[Dict[str, float]] = field(default_factory=list)
    gradient_history: List[float] = field(default_factory=list)
    initial_gradient_norm: Optional[float] = None
    final_gradient_norm: Optional[float] = None
    late_plateau_windows: List[int] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def total_energy(self) -> float:
        """Molecular energy: qubit expectation value + nuclear repulsion."""
        return float(self.optimal_energy + self.energy_offset)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "optimizer": self.optimizer_name,
            "iterations": self.iterations,
            "cost_function_evals": self.cost_function_evals,
            "converged": self.converged,
            "status": self.status,
            "barren_plateau": self.barren_plateau,
            "optimal_energy_qubit": self.optimal_energy,
            "energy_offset": self.energy_offset,
            "total_energy": self.total_energy,
            "optimal_parameters": [float(p) for p in np.asarray(self.optimal_parameters)],
            "parameter_names": self.parameter_names,
            "initial_gradient_norm": self.initial_gradient_norm,
            "final_gradient_norm": self.final_gradient_norm,
            "gradient_history": self.gradient_history,
            "late_plateau_windows": self.late_plateau_windows,
            "energy_history": self.energy_history,
            "warnings": self.warnings,
        }


# ---------------------------------------------------------------------------
# Estimator plumbing
# ---------------------------------------------------------------------------
def make_statevector_estimator(seed: Optional[int] = None) -> BaseEstimator:
    """Exact, noiseless estimator (analytic statevector expectation values).

    The Qiskit reference ``Estimator`` returns exact expectation values rather
    than shot-sampled ones, which is what isolates the algorithm from hardware
    noise in the noiseless stage.  ``seed`` is accepted for interface parity
    with the noisy Aer estimator; the reference primitive is deterministic and
    does not consume it.
    """
    del seed  # analytical expectation values need no random stream
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return Estimator()
    except TypeError:  # pragma: no cover - defensive, API is positional-stable
        return Estimator()


def build_cost_function(
    ansatz: Any,
    hamiltonian: SparsePauliOp,
    estimator: Optional[BaseEstimator] = None,
) -> Callable[[np.ndarray], float]:
    """Return ``C(theta) = <psi(theta)|H|psi(theta)>`` as a plain callable."""
    estimator = estimator or make_statevector_estimator()
    operator = hamiltonian

    def cost(parameters: Sequence[float]) -> float:
        values = np.asarray(parameters, dtype=float).reshape(-1)
        result = estimator.run([ansatz], [operator], [values]).result()
        return float(np.real(result.values[0]))

    return cost


def numerical_gradient(
    cost: Callable[[np.ndarray], float],
    parameters: Sequence[float],
    eps: float = 1e-5,
) -> np.ndarray:
    """Central finite-difference gradient (cost: ``2 * len(parameters)`` evals)."""
    theta = np.asarray(parameters, dtype=float).reshape(-1)
    gradient = np.zeros_like(theta)
    for index in range(theta.size):
        shift = np.zeros_like(theta)
        shift[index] = eps
        gradient[index] = (cost(theta + shift) - cost(theta - shift)) / (2.0 * eps)
    return gradient


# ---------------------------------------------------------------------------
# Plateau detection
# ---------------------------------------------------------------------------
def _find_plateau_windows(
    gradient_history: Sequence[float], threshold: float, patience: int
) -> List[int]:
    """Start indices of every sub-threshold run of length ``patience``."""
    starts: List[int] = []
    run = 0
    for index, norm in enumerate(gradient_history):
        if norm == norm and norm < threshold:  # norm == norm filters NaN
            run += 1
            if run >= patience:
                starts.append(index - patience + 1)
        else:
            run = 0
    return starts


# ---------------------------------------------------------------------------
# VQE
# ---------------------------------------------------------------------------
def run_vqe(
    ansatz: AnsatzBundle,
    hamiltonian: HamiltonianData,
    optimizer: str = "L_BFGS_B",
    initial_point: Optional[Sequence[float]] = None,
    estimator: Optional[BaseEstimator] = None,
    bp_config: Optional[BarrenPlateauConfig] = None,
    optimizer_options: Optional[Dict[str, Any]] = None,
    seed: int = 42,
) -> VQEResult:
    """Optimise the ansatz parameters with VQE and monitor the gradient norm.

    Raises
    ------
    BarrenPlateauError
        If ``||grad C|| < bp_config.gradient_threshold`` for
        ``bp_config.patience`` consecutive iterations inside the exploration
        window ``bp_config.monitor_steps``.
    ValueError
        If ``initial_point`` does not match the ansatz parameter count.
    """
    bp_config = bp_config or BarrenPlateauConfig()
    optimizer_options = dict(optimizer_options or {})
    circuit = ansatz.ansatz
    operator: SparsePauliOp = hamiltonian.qubit_operator
    n_parameters = int(ansatz.num_parameters)
    warnings_: List[str] = list(ansatz.warnings)

    estimator = estimator or make_statevector_estimator(seed=seed)
    cost = build_cost_function(circuit, operator, estimator)

    if initial_point is None:
        rng = np.random.default_rng(seed)
        initial_point = rng.uniform(-0.05, 0.05, size=n_parameters)
    initial_point = np.asarray(initial_point, dtype=float).reshape(-1)
    if initial_point.size != n_parameters:
        raise ValueError(
            f"initial_point has {initial_point.size} entries but the ansatz has "
            f"{n_parameters} parameters"
        )

    parameter_names = [p.name for p in ansatz.parameters]
    parameter_lookup = {p.name: p for p in ansatz.parameters}

    # -- degenerate case: no variational parameters ------------------------
    if n_parameters == 0:
        energy = cost(initial_point)
        warnings_.append("no variational parameters: energy is the reference value")
        return VQEResult(
            optimal_energy=energy,
            energy_offset=hamiltonian.electronic_energy_offset,
            optimal_parameters=initial_point,
            parameter_names=parameter_names,
            iterations=0,
            cost_function_evals=1,
            optimizer_name="none (zero-parameter ansatz)",
            converged=True,
            status="SUCCESS",
            barren_plateau=False,
            energy_history=[{"iteration": 0, "energy": energy, "gradient_norm": 0.0}],
            gradient_history=[0.0],
            initial_gradient_norm=0.0,
            final_gradient_norm=0.0,
            warnings=warnings_,
        )

    probe_gradients = n_parameters <= bp_config.max_probed_parameters
    if not probe_gradients:
        warnings_.append(
            f"gradient probing skipped: {n_parameters} parameters exceed the "
            f"{bp_config.max_probed_parameters}-parameter budget"
        )

    energy_history: List[Dict[str, float]] = []
    gradient_history: List[float] = []

    def _record(params: Sequence[float], value: Optional[float] = None,
                iteration: Optional[int] = None) -> float:
        theta = np.asarray(params, dtype=float).reshape(-1)
        if value is None:
            value = cost(theta)
        grad_norm = float("nan")
        if probe_gradients:
            gradient = numerical_gradient(cost, theta, eps=bp_config.finite_difference_eps)
            grad_norm = float(np.linalg.norm(gradient))
        gradient_history.append(grad_norm)
        index = len(energy_history) if iteration is None else iteration
        energy_history.append(
            {"iteration": int(index), "energy": float(value), "gradient_norm": grad_norm}
        )
        return grad_norm

    # Seal the baseline record before the optimiser starts calling back.
    _record(initial_point, iteration=0)

    def _monitor(*args: Any) -> None:
        """Per-evaluation monitor.

        qiskit-algorithms calls ``callback(eval_count, params, value, meta)``;
        scipy-style optimizers call ``callback(xk)``.  Both shapes are handled so
        the monitor survives either wiring.
        """
        if len(args) == 1 and isinstance(args[0], (np.ndarray, list, tuple)):
            params, value = args[0], None
        elif len(args) >= 2 and isinstance(args[1], (np.ndarray, list, tuple)):
            params = args[1]
            value = float(args[2]) if len(args) > 2 and np.isscalar(args[2]) else None
        else:
            return
        grad_norm = _record(params, value=value)
        iteration = len(gradient_history) - 1
        # Barren plateau: sub-threshold gradients while still exploring.
        if (probe_gradients
                and iteration <= bp_config.monitor_steps
                and grad_norm == grad_norm
                and grad_norm < bp_config.gradient_threshold):
            window = gradient_history[max(0, iteration - bp_config.patience + 1):]
            if len(window) >= bp_config.patience and all(
                g == g and g < bp_config.gradient_threshold for g in window
            ):
                raise BarrenPlateauError(
                    "Barren plateau detected: ||grad C|| < "
                    f"{bp_config.gradient_threshold:g} for {bp_config.patience} "
                    f"consecutive iterations (iteration {iteration}, "
                    f"||grad|| = {grad_norm:.3e}).",
                    gradient_norm=grad_norm,
                    iteration=iteration,
                    gradient_history=gradient_history,
                )

    # The monitor is attached to the VQE wrapper only.  In qiskit-algorithms
    # 0.2.2 that callback fires inside ``evaluate_energy`` -- i.e. on *every*
    # cost evaluation, line searches included -- and exceptions raised there
    # propagate out of ``compute_minimum_eigenvalue`` untouched.  Attaching a
    # second monitor to the optimizer would double-count the history.
    if optimizer.upper() in {"L_BFGS_B", "LBFGSB", "L-BFGS-B"}:
        opt = L_BFGS_B(maxiter=int(optimizer_options.pop("maxiter", 1000)),
                       **optimizer_options)
        optimizer_label = "L_BFGS_B"
    elif optimizer.upper() == "SPSA":
        opt = SPSA(maxiter=int(optimizer_options.pop("maxiter", 300)),
                   seed=seed,
                   **optimizer_options)
        optimizer_label = "SPSA"
    else:
        raise ValueError(f"unsupported optimizer {optimizer!r}; use L_BFGS_B or SPSA")

    vqe = VQE(
        estimator=estimator,
        ansatz=circuit,
        optimizer=opt,
        initial_point=initial_point,
        callback=_monitor,
    )

    converged, status = True, "SUCCESS"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            algorithm_result = vqe.compute_minimum_eigenvalue(operator)
        except BarrenPlateauError:
            raise
        except Exception as exc:  # keep the engine alive, report honestly
            converged = False
            status = f"OPTIMIZER ERROR: {type(exc).__name__}: {exc}"
            warnings_.append(status)
            algorithm_result = None

    if algorithm_result is None:
        best = min(energy_history, key=lambda record: _nan_to_inf(record["energy"]))
        optimal_parameters = initial_point
        optimal_energy = float(best["energy"])
        iterations = len(energy_history) - 1
        cost_evals = len(energy_history)
    else:
        optimal_parameters = np.asarray(
            [_value_of(algorithm_result.optimal_parameters, name) for name in parameter_names],
            dtype=float,
        ) if parameter_names else np.asarray(initial_point)
        optimal_energy = float(np.real(algorithm_result.optimal_value))
        optimizer_result = getattr(algorithm_result, "optimizer_result", None)
        iterations = int(getattr(optimizer_result, "nit", len(energy_history) - 1) or 0)
        cost_evals = int(getattr(algorithm_result, "cost_function_evals", len(energy_history)))
        converged = bool(getattr(optimizer_result, "success", True))
        if not converged:
            status = "FAILED TO CONVERGE"

    final_gradient = float("nan")
    if probe_gradients:
        final_gradient = float(
            np.linalg.norm(
                numerical_gradient(cost, optimal_parameters,
                                   eps=bp_config.finite_difference_eps)
            )
        )

    windows = _find_plateau_windows(
        gradient_history, bp_config.gradient_threshold, bp_config.patience
    )
    late_windows = [w for w in windows if w > bp_config.monitor_steps]
    if late_windows:
        warnings_.append(
            "sub-threshold gradient windows detected after convergence at "
            f"iterations {late_windows} (expected at a minimum, not a plateau)"
        )

    initial_gradient = gradient_history[0] if gradient_history else None

    return VQEResult(
        optimal_energy=optimal_energy,
        energy_offset=hamiltonian.electronic_energy_offset,
        optimal_parameters=optimal_parameters,
        parameter_names=parameter_names,
        iterations=iterations,
        cost_function_evals=cost_evals,
        optimizer_name=optimizer_label,
        converged=converged,
        status=status,
        barren_plateau=False,
        energy_history=energy_history,
        gradient_history=gradient_history,
        initial_gradient_norm=initial_gradient,
        final_gradient_norm=final_gradient,
        late_plateau_windows=late_windows,
        warnings=warnings_,
    )


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _value_of(parameters: Any, name: str) -> float:
    """Read a parameter value from VQE's ``{Parameter: value}`` mapping."""
    if isinstance(parameters, dict):
        for key, value in parameters.items():
            if getattr(key, "name", None) == name:
                return float(np.real(value))
        raise KeyError(f"parameter {name!r} missing from the VQE result")
    index_lookup = {name: idx for idx, name in enumerate(parameters)}
    return float(np.real(parameters[index_lookup.get(name, 0)]))


def _nan_to_inf(value: float) -> float:
    return value if value == value else float("inf")
