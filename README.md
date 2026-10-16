# Q-TMCE — Quantum-Topological Molecular Certification Engine

A hybrid pipeline that **filters** a molecule by algebraic topology, **simulates**
it with symmetry-protected VQE on Qiskit, **mitigates** device noise with
zero-noise extrapolation, and **seals** the result in a cryptographic
commitment.

```
SMILES ──RDKit──► 3-D point cloud
                      │
                      ├─ Vietoris–Rips filtration ──► persistence entropy, Betti numbers
                      │                                   │  (tractable?)
                      ▼                                   ▼
                 HF integrals ──► parity mapping ──► 2 qubits
                      │
                      └─ Hartree–Fock + UCCSD ──► VQE ──► ZNE ──► commitment
```

## Install

> **On Kali/Debian, or a fresh Linux box, use
> [`RUN_ON_KALI_LINUX.txt`](RUN_ON_KALI_LINUX.txt) instead** — a from-scratch,
> copy-paste walkthrough covering the PEP 668 `externally-managed-environment`
> trap, the absent `python3.10` in apt, and the one pinned package that has no
> Linux wheel. The summary below is the short version.

Python **3.10.12** is required by the pinned stack (`requirements.txt`,
`.python-version`).

```bash
python -m venv .venv
. .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

PySCF is required by the production driver but **has no Windows wheel on PyPI**
and no conda-forge build compatible with these pins (see
[Drivers](#choosing-a-driver) below). On Windows the engine still runs: it falls
back to a bundled STO-3G integral engine.

## Run

```bash
python -m src.main                     # H2, the reference molecule
python -m src.main --verbose           # progress on stderr
python -m src.main --extrapolation richardson   # best measured mitigation
python tests/test_smoke.py             # 23 behavioural tests, no pytest needed
```

## Interactive Web Dashboard

Launch the rich web interface:

```bash
python run_ui.py                       # starts local server and opens http://127.0.0.1:8080/
# or manually:
python -m src.web.server --port 8080
```

The dashboard includes:
* **3D Point Cloud & Vietoris-Rips Filtration**: Dynamic 3D atom view and interactive filtration radius slider to visualize persistent simplices.
* **Topological Invariants & Barcode**: Betti numbers ($\beta_0, \beta_1, \beta_2$), normalized entropy, and GUDHI SimplexTree verification.
* **Symmetry-Protected VQE Convergence**: Optimization trajectory chart tracking energy and gradient norm $\|\nabla C\|$ with barren plateau detection.
* **ZNE Error Mitigation Curves**: Gate folding ($\lambda = 1, 3, 5$) with Linear, Richardson, and Exponential zero-noise extrapolation.
* **Cryptographic Tamper-Proof Sandbox**: Interactive verification and tamper simulation over the HMAC-SHA256 commitment.
* **Benchmark Presets & CLI Report**: One-click switching between $H_2$, $HeH^+$, and Benzene ring obstruction, plus live CLI reports and JSON export.

Output: the report on stdout, plus
[`artifacts/qtmce_result.json`](artifacts/qtmce_result.json) (full diagnostics)
and `artifacts/zkp_keystore.json` (the simulated secret store holding the
commitment salts).

## Choosing a driver

`molecule.driver` (or `--driver`) selects the electronic-structure backend:

| Value | Backend | Notes |
| --- | --- | --- |
| `auto` (default) | PySCF when importable, else the fallback | warned in the artifact either way |
| `pyscf` | `qiskit_nature.second_q.drivers.PySCFDriver` | any basis set, open shell, any element |
| `minimal` | [`src/quantum/minimal_basis_driver.py`](src/quantum/minimal_basis_driver.py) | STO-3G 1s on H/He, closed shell, no external chemistry package |

The fallback exists because PySCF is genuinely unavailable on some hosts:
PyPI publishes no Windows wheel for any PySCF 2.x (the sdist needs CMake + a
BLAS), and conda-forge builds it only for Python 3.11, whose numpy-2 ABI is
incompatible with `numpy==1.25.2` that qiskit 0.45.2 needs. On Linux/macOS,
`pyscf` installs from PyPI normally and `auto` will prefer it.

## Verified results

H2, STO-3G, R = 0.7348 Å, `qiskit==0.45.2` / `qiskit-nature==0.6.2` /
`qiskit-aer==0.12.2` / `qiskit-algorithms==0.2.2` on Python 3.10.12:

| Quantity | Engine | Reference |
| --- | --- | --- |
| Bond length | 0.7348 Å | 0.735 Å (config) |
| Qubit count (4 spin-orbitals) | **2** | spec: parity + two-qubit reduction |
| RHF energy | −1.11701 Ha | −1.1167 Ha (literature) |
| FCI energy | −1.13731 Ha | −1.1373 Ha (literature) |
| Dissociation limit (R = 3 Å) | −0.93363 Ha | 2 × E(H) = −0.93316 Ha |
| VQE (UCCSD, L_BFGS_B) | −1.1373060 Ha | exact −1.1373060 Ha (1.9e−12) |
| Initial ‖∇C‖ / final ‖∇C‖ | 0.4757 / 3.3e−6 | no plateau |
| Noisy ⟨H⟩ at scales 1, 3, 5 | −1.8184, −1.7459, −1.6802 | monotone; deterministic |
| ZNE mitigated (linear, default) | −1.131645 Ha | 85.5 % of the noise error removed |
| ZNE mitigated (richardson) | −1.137084 Ha | 99.4 % removed |

Folding was checked to be physics-preserving (every scale reproduces the VQE
value to 1e−14 with no noise attached) *and* to actually scale the noise
(transpiled gates 21 → 63 → 105, two-qubit gates 4 → 12 → 20).

**Mitigation quality depends on the fit.** With exactly three scales the
exponential fit is fully determined and diverges (it wants to subtract 0.8 Ha
from 0.14 Ha of drift), so it is rejected by the physicality guard and the
linear fit is used. Richardson extrapolation — the canonical ZNE fit — is
accepted and recovers 99.4 % of the error. `linear` is the default because the
specification asks for a linear fit; switch it with
`mitigation.extrapolation: "richardson"` when accuracy matters more than
spec-faithfulness.

## Layout

| Path | Role |
| --- | --- |
| [`src/topology/filter.py`](src/topology/filter.py) | Vietoris–Rips filtration, persistence diagrams, entropy, Betti numbers, gudhi cross-check |
| [`src/topology/classifier.py`](src/topology/classifier.py) | Tractability decision (JSON result) |
| [`src/quantum/hamiltonian_builder.py`](src/quantum/hamiltonian_builder.py) | RDKit geometry → integrals → parity-mapped Hamiltonian |
| [`src/quantum/minimal_basis_driver.py`](src/quantum/minimal_basis_driver.py) | PySCF-free STO-3G integrals + RHF fallback driver |
| [`src/quantum/ansatz.py`](src/quantum/ansatz.py) | Hartree–Fock + UCCSD, with a guard against hardware-efficient layers |
| [`src/quantum/simulator.py`](src/quantum/simulator.py) | VQE loop, per-evaluation gradient monitoring, `BarrenPlateauError` |
| [`src/quantum/mitigation.py`](src/quantum/mitigation.py) | Depolarizing model, gate folding ×1/×3/×5, extrapolation to λ=0 |
| [`src/crypto/zkp_wrapper.py`](src/crypto/zkp_wrapper.py) | Salted HMAC commitment + `verify()` |
| [`src/main.py`](src/main.py) | Orchestrator and report |
| [`config.json`](config.json) | Every threshold, plus the IBM Quantum token slot |

## Deviations from the specification (and why)

The specified API surface does not exist in the pinned versions, so five
substitutions were necessary. Each is marked in the source.

1. **`qiskit_nature.drivers.PyMolecule` → `PySCFDriver`.** `PyMolecule` is not
   importable from `qiskit_nature.drivers` in 0.6.2 (the legacy driver module
   was retired); the supported entry point is
   `qiskit_nature.second_q.drivers.PySCFDriver`, which needs `pyscf`.
2. **`QubitConverter(..., two_qubit_reduction=True)` →
   `ParityMapper(num_particles=...)`.** Identical reduction (H2/STO-3G: 4 → 2
   qubits, asserted by the tests); `QubitConverter` is soft-deprecated in 0.6.x.
   `qiskit_nature.settings.use_pauli_sum_op = False` is set so the mapper returns
   the modern `SparsePauliOp` instead of the opflow `PauliSumOp` that
   `qiskit-algorithms` 0.2 no longer consumes.
3. **`QubitOperator` → `SparsePauliOp`.** `qiskit.opflow` was dropped by
   `qiskit-algorithms` 0.2.x, so the engine carries the operator type that
   `qiskit_algorithms.VQE` accepts.
4. **A bundled fallback driver was added** ([`minimal_basis_driver.py`](src/quantum/minimal_basis_driver.py))
   so the engine runs on hosts where PySCF cannot be installed. It is selected
   only when PySCF is missing, and it reports itself in the artifact.
5. **The "ZKP" is a commitment scheme, not a zero-knowledge proof.** Salted
   HMAC-SHA256 gives auditing (tamper-evident publication) but *not* zero
   knowledge, and it is not lattice-based. Every output labels it honestly.

Further refinements, all configurable in `config.json`:

* **Persistence entropy is normalised** (`E / log n_bars`, in `[0,1]`) so the
  `> 0.5` threshold means the same thing for H2 and for a 20-atom molecule. Set
  `topology.entropy_mode: "raw"` for the unnormalised value.
* **The barren-plateau rule is scoped.** Gradients are recorded at every cost
  evaluation, but the abort triggers only inside the exploration window
  (`barren_plateau.monitor_steps`). At a *converged* minimum the gradient is
  also ≈0; treating that as a plateau would abort every successful run. Late
  sub-threshold windows are still reported in `vqe.late_plateau_windows`.
* **ZNE evaluates exactly by default** (`mitigation.shots: null` → density-matrix
  trace, no shot noise). `AerEstimator` cannot do this: with `shots=None` it
  silently samples 1024 shots, and with `approximation=True` it ignores the
  noise model. Set `mitigation.shots` to an integer to emulate sampling instead.
  Both paths transpile with `optimization_level=0`, because an optimisation pass
  can cancel the `U†U` folds and silently flatten every noise scale.
* **`mitigation.reoptimize_with: "SPSA"`** re-optimises at each noise scale
  (noisy VQE) instead of holding the noiseless optimum fixed.

## Honest scope

* "Symmetry-protected" means the UCCSD ansatz is built from particle- and
  spin-conserving fermionic excitations. It is *not* a formal guarantee of
  plateau-freedom, and it is not the symmetry-protected topological order of
  condensed-matter physics.
* The Betti numbers reported for `k ≥ 1` count bars with persistence above
  `topology.prominence`; the Rips complex of a finite point cloud has no
  essential `H_k` classes for `k ≥ 1`. `gudhi` independently computes the exact
  Betti numbers of `VR(L_ref)` as a cross-check, and the two are compared on
  `beta_0` (the only quantity with the same definition in both).
* `beta_0` is evaluated at `topology.reference_scale` (≈ a covalent reach), so
  `beta_0 > 1` flags a fragmented geometry rather than a mathematical invariant.
* `gudhi`'s `SimplexTree.compute_persistence()` **segfaults** on a complex
  holding only 0-simplices; the cross-check therefore uses
  `persistence(persistence_dim_max=True)` and short-circuits the degenerate
  case. `persistence_dim_max=True` is also required for correctness — without it
  the top homology dimension is skipped and a molecular ring reports `beta_1=0`.
* The restricted closed-shell fallback driver is not size-consistent: its RHF
  energy is poor at stretched geometries even though its FCI is correct.
* The fallback driver covers STO-3G 1s functions on H and He only. Anything else
  raises an error naming PySCF.
* ZNE assumes the noise grows with the folding scale, which holds for
  depolarizing noise but not for every error channel; the extrapolated value is
  an estimate, not a bound.
* The engine runs on the Aer simulator. `config.json` carries the IBM Quantum
  token slot, but hardware execution is deliberately not wired up; the token is
  never logged.
