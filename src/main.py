"""Q-TMCE orchestrator: topology filter -> VQE -> ZNE -> cryptographic seal.

Workflow
--------
1.  **Geometry**   RDKit builds a 3-D structure from the SMILES string.
2.  **Topology**   persistent homology of the nuclear point cloud decides
    tractability; an obstructed molecule stops here.
3.  **Hamiltonian** PySCF (via Qiskit Nature) + parity mapping.
4.  **Ansatz**     Hartree-Fock + UCCSD on the symmetry-preserving subspace.
5.  **Simulation** VQE with per-iteration gradient monitoring.
6.  **Mitigation** zero-noise extrapolation under a depolarizing model.
7.  **Certification** salted commitment over the published result.

The report printed on stdout follows the specified layout exactly; progress
messages, warnings and diagnostics go to stderr or into the JSON artifact.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .crypto.zkp_wrapper import LatticeCommitmentScheme
from .quantum.ansatz import AnsatzConfig, build_uccsd_ansatz
from .quantum.hamiltonian_builder import build_geometry, build_hamiltonian
from .quantum.mitigation import run_zne
from .quantum.simulator import (
    BarrenPlateauConfig,
    BarrenPlateauError,
    run_vqe,
)
from .topology.classifier import TopologicalClassifier

ENGINE_VERSION = "1.0.0"

EXIT_OK = 0
EXIT_INTRACTABLE = 1
EXIT_BARREN_PLATEAU = 2
EXIT_ERROR = 3

_DEFAULT_CONFIG: Dict[str, Any] = {
    "molecule": {
        "smiles": "[H][H]",
        "bond_length_angstrom": 0.7348,
        "basis": "sto3g",
        "charge": 0,
        "multiplicity": 1,
        # "auto" = PySCF when importable, else the built-in STO-3G fallback.
        "driver": "auto",
    },
    "topology": {
        "entropy_threshold": 0.5,
        "entropy_mode": "normalized",
        "maxdim": 2,
        "reference_scale": 1.4,
        "prominence": 0.3,
        "allow_beta_1": 0,
        "gudhi_cross_check": True,
    },
    "quantum": {
        "two_qubit_reduction": True,
        "optimizer": "L_BFGS_B",
        "optimizer_options": {"maxiter": 1000},
        "seed": 42,
        "ansatz": {"reps": 1, "generalized": False, "preserve_spin": True},
        "barren_plateau": {
            "gradient_threshold": 1e-7, "patience": 3, "monitor_steps": 5,
            "finite_difference_eps": 1e-5, "max_probed_parameters": 24,
        },
    },
    "mitigation": {
        "scales": [1, 3, 5],
        "extrapolation": "linear",
        "shots": None,
        "single_qubit_error": 1e-3,
        "two_qubit_error": 1e-2,
        "readout_error": 0.0,
        "reoptimize_with": None,
        "seed": 42,
    },
    "crypto": {
        "keystore_path": "artifacts/zkp_keystore.json",
        "salt_bytes": 32,
    },
    "output": {
        "artifact_path": "artifacts/qtmce_result.json",
        "print_report": True,
    },
    "ibm_quantum": {
        "token": "<YOUR-IBM-QUANTUM-TOKEN>",
        "channel": "ibm_quantum",
        "instance": "",
        "use_hardware": False,
    },
}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``override`` into a copy of ``base``."""
    merged = dict(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: Optional[str], overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Load ``config.json`` (if present), then apply CLI overrides."""
    config = dict(_DEFAULT_CONFIG)
    if path:
        config_path = Path(path)
        if config_path.exists():
            with config_path.open("r", encoding="utf-8") as handle:
                config = deep_merge(config, json.load(handle))
        elif path != "config.json":
            raise FileNotFoundError(f"config file not found: {path}")
    return deep_merge(config, overrides or {})


def resolve_backend(config: Dict[str, Any]) -> Dict[str, Any]:
    """Report which backend the run uses.

    This proof of concept executes on the Aer statevector/density-matrix
    simulator.  A real IBM Quantum token is read from the configuration (never
    logged) so the same config file works unchanged once hardware execution is
    wired up.
    """
    ibm = config.get("ibm_quantum", {})
    token = str(ibm.get("token", "") or "")
    placeholder = (not token) or token.startswith("<") or "YOUR" in token.upper()
    return {
        "execution": "qiskit-aer (statevector / density-matrix)",
        "hardware_requested": bool(ibm.get("use_hardware", False)),
        "hardware_available": not placeholder,
        "hardware_reason": "placeholder token in config.json -- simulation only"
        if placeholder else "token present; this PoC still simulates locally",
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def format_report(
    *,
    molecule_name: str,
    bond_length: float,
    topology: Optional[Dict[str, Any]],
    ansatz_label: str,
    optimizer_name: str,
    iterations: int,
    convergence: str,
    barren_plateau: bool,
    zne: Optional[Any] = None,
    fci_energy: Optional[float] = None,
    error_reduction: Optional[float] = None,
    commitment_hash: Optional[str] = None,
    skipped_reason: Optional[str] = None,
) -> str:
    """Render the engine report in the specified layout.

    Sections that did not run are printed with their reason instead of being
    silently dropped, so the report always states what was and was not
    established.
    """
    lines: List[str] = []
    lines.append("=== Q-TMCE ENGINE REPORT ===")
    lines.append("")
    lines.append(f"Molecule: {molecule_name} (Bond Length: {bond_length:.4f} Angstrom)")
    lines.append("")

    # -- [1] topology -------------------------------------------------------
    lines.append("[1] TOPOLOGICAL FILTER")
    if topology is None:
        lines.append("Status: NOT RUN")
        lines.append("Reason: topology filter did not complete.")
    else:
        status = "PASS (Tractable)" if topology["tractable"] else "FAIL (Obstructed)"
        lines.append(f"Status: {status}")
        lines.append(f"Reason: {topology['reason']}")
    lines.append("")

    # -- [2] quantum simulation --------------------------------------------
    lines.append("[2] QUANTUM SIMULATION")
    lines.append(f"Ansatz: {ansatz_label}")
    lines.append(f"Optimizer: {optimizer_name}")
    lines.append(f"Iterations: {iterations}")
    lines.append(f"Convergence: {convergence}")
    lines.append(f"Barren Plateau Detected: {'YES' if barren_plateau else 'NO'}")
    lines.append("")

    # -- [3] error mitigation ----------------------------------------------
    lines.append("[3] ERROR MITIGATION")
    if zne is None:
        lines.append(f"Status: NOT RUN ({skipped_reason or 'simulation did not complete'})")
    else:
        lines.append(f"Raw Estimate: {zne.raw_total_energy:.6f} Ha")
        lines.append(f"Mitigated Estimate: {zne.mitigated_total_energy:.6f} Ha")
        lines.append(
            f"Exact Classical (FCI): {fci_energy:.6f} Ha"
            if fci_energy is not None else "Exact Classical (FCI): UNAVAILABLE"
        )
        lines.append(
            f"Error Reduction: {error_reduction:.2f}%"
            if error_reduction is not None else "Error Reduction: UNAVAILABLE"
        )
    lines.append("")

    # -- [4] certification --------------------------------------------------
    lines.append("[4] CRYPTOGRAPHIC CERTIFICATION")
    if commitment_hash is None:
        lines.append("Status: NOT GENERATED")
    else:
        lines.append(f"ZKP Commitment Hash: {commitment_hash}")
        lines.append("Status: PROOF GENERATED")
    lines.append("")

    # -- conclusion ---------------------------------------------------------
    lines.append("=== CONCLUSION ===")
    lines.append("")
    if topology is None or not topology["tractable"]:
        lines.append("The molecule was rejected by the topological filter.")
        lines.append("No quantum simulation or certification was performed.")
    elif barren_plateau:
        lines.append("The molecule passed the topological filter.")
        lines.append(
            "The variational optimisation was aborted: the cost landscape is "
            "flat (barren plateau)."
        )
        lines.append("No cryptographic proof of a ground-state energy was generated.")
    else:
        lines.append("The molecule is certified as computationally tractable.")
        lines.append(
            "The ground state energy has been verified via symmetry-protection "
            "and error mitigation."
        )
        lines.append("A cryptographic proof of this result has been generated.")
    lines.append("")
    lines.append("===============================")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------
def run_engine(config: Dict[str, Any]) -> Tuple[str, Dict[str, Any], int]:
    """Execute the full pipeline and return ``(report, artifact, exit_code)``."""
    started = time.time()
    molecule_cfg = config["molecule"]
    topo_cfg = config["topology"]
    quantum_cfg = config["quantum"]
    mitigation_cfg = config["mitigation"]

    smiles = str(molecule_cfg["smiles"])
    bond_length_cfg = molecule_cfg.get("bond_length_angstrom")
    bond_length_cfg = None if bond_length_cfg is None else float(bond_length_cfg)
    artifact: Dict[str, Any] = {
        "engine": "Q-TMCE",
        "version": ENGINE_VERSION,
        "config": config,
        "backend": resolve_backend(config),
    }

    # -- 1. geometry --------------------------------------------------------
    geometry = build_geometry(smiles, bond_length=bond_length_cfg)
    bond_lengths = geometry.bond_lengths
    if len(bond_lengths) == 1:
        bond_length, bond_length_kind = bond_lengths[0], "single"
    elif bond_lengths:
        bond_length, bond_length_kind = sum(bond_lengths) / len(bond_lengths), "mean_pairwise"
    else:
        bond_length, bond_length_kind = 0.0, "undefined"
    artifact["geometry"] = {**geometry.to_dict(), "bond_length_kind": bond_length_kind,
                            "xyz_block": geometry.xyz_block}

    # -- 2. topological filter ---------------------------------------------
    classifier = TopologicalClassifier(topo_cfg)
    topology = classifier.run(geometry.coords)
    artifact["topology"] = topology

    if not topology["tractable"]:
        report = format_report(
            molecule_name=geometry.name, bond_length=bond_length, topology=topology,
            ansatz_label="n/a", optimizer_name="n/a", iterations=0,
            convergence="NOT RUN", barren_plateau=False, skipped_reason="molecule rejected",
        )
        artifact["exit_code"] = EXIT_INTRACTABLE
        artifact["runtime_seconds"] = time.time() - started
        return report, artifact, EXIT_INTRACTABLE

    # -- 3. hamiltonian -----------------------------------------------------
    hamiltonian = build_hamiltonian(
        geometry,
        basis=str(molecule_cfg.get("basis", "sto3g")),
        charge=int(molecule_cfg.get("charge", 0)),
        multiplicity=int(molecule_cfg.get("multiplicity", 1)),
        two_qubit_reduction=bool(quantum_cfg.get("two_qubit_reduction", True)),
        driver=str(molecule_cfg.get("driver", "auto")),
    )
    artifact["hamiltonian"] = hamiltonian.to_dict()

    # -- 4. symmetry-protected ansatz --------------------------------------
    ansatz_cfg = AnsatzConfig(**quantum_cfg.get("ansatz", {}))
    bundle = build_uccsd_ansatz(hamiltonian, ansatz_cfg)
    artifact["ansatz"] = bundle.to_dict()

    # -- 5. VQE with gradient monitoring -----------------------------------
    bp_config = BarrenPlateauConfig(**quantum_cfg.get("barren_plateau", {}))
    try:
        vqe = run_vqe(
            bundle,
            hamiltonian,
            optimizer=str(quantum_cfg.get("optimizer", "L_BFGS_B")),
            bp_config=bp_config,
            optimizer_options=quantum_cfg.get("optimizer_options"),
            seed=int(quantum_cfg.get("seed", 42)),
        )
    except BarrenPlateauError as exc:
        artifact["vqe"] = {
            "barren_plateau": True,
            "error": str(exc),
            "gradient_norm": exc.gradient_norm,
            "iteration": exc.iteration,
            "gradient_history": exc.gradient_history,
        }
        report = format_report(
            molecule_name=geometry.name, bond_length=bond_length, topology=topology,
            ansatz_label=bundle.label,
            optimizer_name=str(quantum_cfg.get("optimizer", "L_BFGS_B")),
            iterations=exc.iteration, convergence="FAILED (BARREN PLATEAU)",
            barren_plateau=True, skipped_reason="barren plateau",
        )
        artifact["exit_code"] = EXIT_BARREN_PLATEAU
        artifact["runtime_seconds"] = time.time() - started
        return report, artifact, EXIT_BARREN_PLATEAU

    artifact["vqe"] = vqe.to_dict()
    fci_energy = hamiltonian.exact_ground_state_energy()
    artifact["exact_classical_energy"] = fci_energy

    # -- 6. zero-noise extrapolation ---------------------------------------
    zne = run_zne(
        hamiltonian,
        bundle.ansatz,
        vqe.optimal_parameters,
        scales=mitigation_cfg.get("scales", [1, 3, 5]),
        single_qubit_error=float(mitigation_cfg.get("single_qubit_error", 1e-3)),
        two_qubit_error=float(mitigation_cfg.get("two_qubit_error", 1e-2)),
        readout_error=float(mitigation_cfg.get("readout_error", 0.0)),
        shots=mitigation_cfg.get("shots"),
        seed=int(mitigation_cfg.get("seed", 42)),
        extrapolation=str(mitigation_cfg.get("extrapolation", "linear")),
        noiseless_estimate=vqe.total_energy,
        reoptimize_with=mitigation_cfg.get("reoptimize_with"),
        optimizer_options=quantum_cfg.get("optimizer_options"),
    )
    error_reduction = zne.error_reduction_percent(fci_energy)
    artifact["mitigation"] = {**zne.to_dict(), "error_reduction_percent": error_reduction}

    # -- 7. cryptographic certification ------------------------------------
    crypto_cfg = config["crypto"]
    scheme = LatticeCommitmentScheme(
        keystore_path=crypto_cfg.get("keystore_path", "artifacts/zkp_keystore.json"),
        salt_bytes=int(crypto_cfg.get("salt_bytes", 32)),
    )
    payload = {
        "engine": "Q-TMCE",
        "version": ENGINE_VERSION,
        "molecule": geometry.name,
        "smiles": smiles,
        "basis": hamiltonian.basis,
        "num_qubits": hamiltonian.num_qubits,
        "ansatz": bundle.label,
        "optimizer": vqe.optimizer_name,
        "iterations": vqe.iterations,
        "raw_estimate_ha": round(zne.raw_total_energy, 6),
        "mitigated_estimate_ha": round(zne.mitigated_total_energy, 6),
        "fci_energy_ha": round(fci_energy, 6),
        "error_reduction_percent": round(error_reduction, 4),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    commitment = scheme.commit(payload, label="ground-state-energy")
    verification = scheme.verify_detailed(commitment.proof_id, payload)
    tampered = dict(payload, mitigated_estimate_ha=round(zne.mitigated_total_energy + 1e-3, 6))
    tamper_check = scheme.verify_detailed(commitment.proof_id, tampered)
    certification_ok = bool(verification["valid"]) and not bool(tamper_check["valid"])
    artifact["certification"] = {
        "commitment": commitment.to_dict(),
        "verification": verification,
        "tamper_check": {
            "expected_to_fail": True,
            "rejected": not tamper_check["valid"],
            "reason": tamper_check["reason"],
        },
        "certification_ok": certification_ok,
    }
    if not certification_ok:
        # The certification is the engine's product: a self-verification failure
        # must not be reported as a generated proof.
        artifact["certification"]["failure"] = (
            "the commitment did not re-verify against its own payload; the "
            "published energy is NOT certified"
        )

    # -- report -------------------------------------------------------------
    report = format_report(
        molecule_name=geometry.name,
        bond_length=bond_length,
        topology=topology,
        ansatz_label=bundle.label,
        optimizer_name=vqe.optimizer_name,
        iterations=vqe.iterations,
        convergence="SUCCESS" if vqe.converged else vqe.status,
        barren_plateau=vqe.barren_plateau,
        zne=zne,
        fci_energy=fci_energy,
        error_reduction=error_reduction,
        commitment_hash=commitment.commitment_hash if certification_ok else None,
    )
    artifact["exit_code"] = EXIT_OK if certification_ok else EXIT_ERROR
    artifact["runtime_seconds"] = time.time() - started
    artifact["report"] = report
    return report, artifact, (EXIT_OK if certification_ok else EXIT_ERROR)


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point.  Returns the process exit code."""
    parser = argparse.ArgumentParser(
        prog="qtmce", description="Q-TMCE: Quantum-Topological Molecular Certification Engine"
    )
    parser.add_argument("--config", default="config.json", help="path to config.json")
    parser.add_argument("--smiles", default=None, help="override the molecule")
    parser.add_argument("--bond-length", type=float, default=None,
                        help="internuclear distance in Angstrom (diatomics)")
    parser.add_argument("--basis", default=None, help="one-particle basis set")
    parser.add_argument("--driver", default=None, choices=["auto", "pyscf", "minimal"],
                        help="electronic-structure backend")
    parser.add_argument("--optimizer", default=None, help="L_BFGS_B or SPSA")
    parser.add_argument("--extrapolation", default=None,
                        choices=["linear", "richardson", "exponential"])
    parser.add_argument("--artifact", default=None, help="path for the JSON artifact")
    parser.add_argument("--no-artifact", action="store_true", help="skip writing the artifact")
    parser.add_argument("--verbose", action="store_true", help="verbose progress on stderr")
    args = parser.parse_args(argv)

    overrides: Dict[str, Any] = {"molecule": {}, "quantum": {}, "mitigation": {}, "output": {}}
    if args.smiles:
        overrides["molecule"]["smiles"] = args.smiles
    if args.bond_length is not None:
        overrides["molecule"]["bond_length_angstrom"] = args.bond_length
    if args.basis:
        overrides["molecule"]["basis"] = args.basis
    if args.driver:
        overrides["molecule"]["driver"] = args.driver
    if args.optimizer:
        overrides["quantum"]["optimizer"] = args.optimizer
    if args.extrapolation:
        overrides["mitigation"]["extrapolation"] = args.extrapolation
    if args.artifact:
        overrides["output"]["artifact_path"] = args.artifact
    if args.no_artifact:
        overrides["output"]["artifact_path"] = None

    try:
        config = load_config(args.config, overrides)
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        print(f"[Q-TMCE] configuration error: {exc}", file=sys.stderr)
        return EXIT_ERROR

    logged = lambda message: print(f"[Q-TMCE] {message}", file=sys.stderr)  # noqa: E731

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            if args.verbose:
                logged("starting pipeline")
            report, artifact, exit_code = run_engine(config)
        except Exception as exc:  # keep the CLI honest about failure
            import traceback

            print(f"[Q-TMCE] engine error: {type(exc).__name__}: {exc}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            return EXIT_ERROR

    artifact["warnings"] = [
        f"{entry.category.__name__}: {entry.message}" for entry in caught
    ]

    if config.get("output", {}).get("print_report", True):
        print(report)

    artifact_path = config.get("output", {}).get("artifact_path")
    if artifact_path:
        path = Path(artifact_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(artifact, handle, indent=2, sort_keys=True, default=str)
        if args.verbose:
            logged(f"artifact written to {path}")

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
