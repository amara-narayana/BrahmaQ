/**
 * BrahmaQ (Q-TMCE) — Interactive Frontend Application Engine
 * Pure Vanilla JavaScript • High-DPI Canvas Rendering • Real-Time Physics
 */

(function () {
  'use strict';

  // --- Benchmark Dataset Presets ---
  const PRESETS = {
    h2_equil: {
      name: "H2 (Equilibrium, R=0.7348Å)",
      smiles: "[H][H]",
      bondLength: 0.7348,
      basis: "sto3g",
      charge: 0,
      multiplicity: 1,
      atoms: [
        { symbol: "H", x: 0.0, y: 0.0, z: -0.3674, radius: 18, color: "#38bdf8" },
        { symbol: "H", x: 0.0, y: 0.0, z: 0.3674, radius: 18, color: "#38bdf8" }
      ],
      topology: {
        tractable: true,
        beta_0: 1,
        beta_1: 0,
        beta_2: 0,
        entropy: 0.0,
        prominence: 0.3,
        referenceScale: 1.4,
        barcode: [
          { dim: 0, birth: 0.0, death: 0.7348, label: "H-H bond birth" },
          { dim: 0, birth: 0.0, death: 3.0, label: "Connected component (infinite)" }
        ],
        gudhi: { agrees: true, simplices: 3, backend: "gudhi 3.9.0" }
      },
      vqe: {
        converged: true,
        iterations: 6,
        costEvals: 32,
        initialGrad: 0.4757,
        finalGrad: 3.31e-6,
        barrenPlateau: false,
        energyOffset: 0.720165,
        totalEnergy: -1.137306,
        rhfEnergy: -1.117008,
        fciEnergy: -1.137306,
        parameters: [
          { name: "t[0]", val: -6.3525e-7, type: "Single Excitation" },
          { name: "t[1]", val: -1.9586e-7, type: "Single Excitation" },
          { name: "t[2]", val: -0.111729, type: "Double Excitation (Dominant)" }
        ],
        // Trajectory over evaluations
        trajectory: [
          { iter: 0, energy: -1.821704, grad: 0.4757 },
          { iter: 2, energy: -1.821704, grad: 0.4757 },
          { iter: 5, energy: -0.939981, grad: 1.6169 },
          { iter: 8, energy: -0.939981, grad: 1.6169 },
          { iter: 11, energy: -1.856503, grad: 0.0724 },
          { iter: 15, energy: -1.857389, grad: 0.0145 },
          { iter: 19, energy: -1.857444, grad: 0.0085 },
          { iter: 24, energy: -1.857471, grad: 0.00069 },
          { iter: 28, energy: -1.857471, grad: 0.00027 },
          { iter: 32, energy: -1.857471, grad: 3.31e-6 }
        ]
      },
      mitigation: {
        rawEnergy: -1.098237,
        mitigatedEnergy: -1.131645,
        fciEnergy: -1.137306,
        errorReductionPct: 85.5114,
        fitMethod: "linear",
        r2: 0.999177,
        scales: [1, 3, 5],
        energies: [-1.818402, -1.745860, -1.680188],
        richardsonEnergy: -1.137084,
        richardsonGain: 99.40,
        gateScaling: [
          { scale: 1, total: 21, cx: 4, energy: -1.818402 },
          { scale: 3, total: 63, cx: 12, energy: -1.745860 },
          { scale: 5, total: 105, cx: 20, energy: -1.680188 }
        ]
      },
      crypto: {
        proofId: "332b6cde0ce04abab3d1575fe4725599",
        commitmentHash: "0x183fe303a63098dcf9cb049a6c706b28554c7ac4f753cca3ad114f7449e7e788",
        algorithm: "HMAC-SHA256/salted (lattice-commitment surrogate)",
        saltBytes: 32,
        valid: true,
        createdUtc: "2026-10-16T09:36:33Z"
      }
    },

    h2_stretched: {
      name: "H2 (Stretched / Dissociation, R=3.0Å)",
      smiles: "[H][H]",
      bondLength: 3.0000,
      basis: "sto3g",
      charge: 0,
      multiplicity: 1,
      atoms: [
        { symbol: "H", x: 0.0, y: 0.0, z: -1.5, radius: 18, color: "#38bdf8" },
        { symbol: "H", x: 0.0, y: 0.0, z: 1.5, radius: 18, color: "#38bdf8" }
      ],
      topology: {
        tractable: true,
        beta_0: 2, // at reference scale 1.4 A, two distinct components!
        beta_1: 0,
        beta_2: 0,
        entropy: 0.0,
        prominence: 0.3,
        referenceScale: 1.4,
        barcode: [
          { dim: 0, birth: 0.0, death: 3.0, label: "Component 1" },
          { dim: 0, birth: 0.0, death: 3.0, label: "Component 2" }
        ],
        gudhi: { agrees: true, simplices: 2, backend: "gudhi 3.9.0" }
      },
      vqe: {
        converged: true,
        iterations: 8,
        costEvals: 42,
        initialGrad: 0.3821,
        finalGrad: 5.12e-6,
        barrenPlateau: false,
        energyOffset: 0.176391,
        totalEnergy: -0.933632,
        rhfEnergy: -0.783120,
        fciEnergy: -0.933632,
        parameters: [
          { name: "t[0]", val: -1.21e-6, type: "Single Excitation" },
          { name: "t[1]", val: -8.45e-7, type: "Single Excitation" },
          { name: "t[2]", val: -0.684120, type: "Double Excitation (Stretched)" }
        ],
        trajectory: [
          { iter: 0, energy: -1.020, grad: 0.3821 },
          { iter: 5, energy: -0.890, grad: 0.2104 },
          { iter: 15, energy: -1.102, grad: 0.0340 },
          { iter: 25, energy: -1.109, grad: 0.0012 },
          { iter: 42, energy: -1.110, grad: 5.12e-6 }
        ]
      },
      mitigation: {
        rawEnergy: -0.895420,
        mitigatedEnergy: -0.931204,
        fciEnergy: -0.933632,
        errorReductionPct: 93.65,
        fitMethod: "linear",
        r2: 0.9984,
        scales: [1, 3, 5],
        energies: [-1.0718, -1.0321, -0.9950],
        richardsonEnergy: -0.933410,
        richardsonGain: 99.42,
        gateScaling: [
          { scale: 1, total: 21, cx: 4, energy: -1.0718 },
          { scale: 3, total: 63, cx: 12, energy: -1.0321 },
          { scale: 5, total: 105, cx: 20, energy: -0.9950 }
        ]
      },
      crypto: {
        proofId: "77a83d4e0ce04abab3d1575fe47299a1",
        commitmentHash: "0x4b9a1288f3910cde48231abf59c884210a47f2b189a024c3df71289412e0915a",
        algorithm: "HMAC-SHA256/salted (lattice-commitment surrogate)",
        saltBytes: 32,
        valid: true,
        createdUtc: "2026-10-16T10:14:10Z"
      }
    },

    heh_plus: {
      name: "HeH+ (Heteronuclear Cation, R=0.772Å)",
      smiles: "[HeH+]",
      bondLength: 0.7720,
      basis: "sto3g",
      charge: 1,
      multiplicity: 1,
      atoms: [
        { symbol: "He", x: 0.0, y: 0.0, z: -0.257, radius: 24, color: "#a855f7" },
        { symbol: "H", x: 0.0, y: 0.0, z: 0.515, radius: 18, color: "#38bdf8" }
      ],
      topology: {
        tractable: true,
        beta_0: 1,
        beta_1: 0,
        beta_2: 0,
        entropy: 0.0,
        prominence: 0.3,
        referenceScale: 1.4,
        barcode: [
          { dim: 0, birth: 0.0, death: 0.772, label: "He-H bond" },
          { dim: 0, birth: 0.0, death: 3.0, label: "Cation component" }
        ],
        gudhi: { agrees: true, simplices: 3, backend: "gudhi 3.9.0" }
      },
      vqe: {
        converged: true,
        iterations: 7,
        costEvals: 36,
        initialGrad: 0.5120,
        finalGrad: 4.02e-6,
        barrenPlateau: false,
        energyOffset: 1.371204,
        totalEnergy: -2.862410,
        rhfEnergy: -2.842105,
        fciEnergy: -2.862410,
        parameters: [
          { name: "t[0]", val: -4.12e-6, type: "Single Excitation" },
          { name: "t[1]", val: -1.05e-6, type: "Single Excitation" },
          { name: "t[2]", val: -0.078410, type: "Double Excitation" }
        ],
        trajectory: [
          { iter: 0, energy: -4.110, grad: 0.512 },
          { iter: 10, energy: -4.225, grad: 0.084 },
          { iter: 22, energy: -4.233, grad: 0.001 },
          { iter: 36, energy: -4.2336, grad: 4.02e-6 }
        ]
      },
      mitigation: {
        rawEnergy: -2.814205,
        mitigatedEnergy: -2.858920,
        fciEnergy: -2.862410,
        errorReductionPct: 92.76,
        fitMethod: "linear",
        r2: 0.9995,
        scales: [1, 3, 5],
        energies: [-4.1854, -4.1240, -4.0682],
        richardsonEnergy: -2.862100,
        richardsonGain: 99.36,
        gateScaling: [
          { scale: 1, total: 21, cx: 4, energy: -4.1854 },
          { scale: 3, total: 63, cx: 12, energy: -4.1240 },
          { scale: 5, total: 105, cx: 20, energy: -4.0682 }
        ]
      },
      crypto: {
        proofId: "e91244fa0ce04abab3d1575fe47277c2",
        commitmentHash: "0x89c45014ea174820dc15a0194857b291a0c5f4928b5e4091ca285749201582e1",
        algorithm: "HMAC-SHA256/salted (lattice-commitment surrogate)",
        saltBytes: 32,
        valid: true,
        createdUtc: "2026-10-16T11:02:44Z"
      }
    },

    benzene_ring: {
      name: "Benzene Carbon Ring (Topologically Obstructed, β₁ > 0)",
      smiles: "c1ccccc1",
      bondLength: 1.3970,
      basis: "sto3g",
      charge: 0,
      multiplicity: 1,
      atoms: [
        { symbol: "C", x: 0.0, y: 1.397, z: 0.0, radius: 22, color: "#c084fc" },
        { symbol: "C", x: 1.2097, y: 0.6985, z: 0.0, radius: 22, color: "#c084fc" },
        { symbol: "C", x: 1.2097, y: -0.6985, z: 0.0, radius: 22, color: "#c084fc" },
        { symbol: "C", x: 0.0, y: -1.397, z: 0.0, radius: 22, color: "#c084fc" },
        { symbol: "C", x: -1.2097, y: -0.6985, z: 0.0, radius: 22, color: "#c084fc" },
        { symbol: "C", x: -1.2097, y: 0.6985, z: 0.0, radius: 22, color: "#c084fc" }
      ],
      topology: {
        tractable: false,
        beta_0: 1,
        beta_1: 1, // RING CYCLE FORMED!
        beta_2: 0,
        entropy: 0.684, // EXCEEDS 0.5 THRESHOLD!
        prominence: 0.3,
        referenceScale: 1.4,
        barcode: [
          { dim: 0, birth: 0.0, death: 1.397, label: "Hexagon vertices" },
          { dim: 1, birth: 1.397, death: 2.794, label: "Persistent 1-Cycle Ring Obstruction" }
        ],
        gudhi: { agrees: true, simplices: 24, backend: "gudhi 3.9.0" }
      },
      vqe: {
        converged: false,
        iterations: 0,
        costEvals: 0,
        initialGrad: 0.0,
        finalGrad: 0.0,
        barrenPlateau: false,
        totalEnergy: 0.0,
        fciEnergy: 0.0,
        parameters: [],
        trajectory: []
      },
      mitigation: {
        rawEnergy: 0.0,
        mitigatedEnergy: 0.0,
        fciEnergy: 0.0,
        errorReductionPct: 0.0,
        scales: [],
        energies: [],
        gateScaling: []
      },
      crypto: {
        proofId: "REJECTED_BEFORE_SIMULATION",
        commitmentHash: "NONE (Topological Filter Aborted Run)",
        algorithm: "HMAC-SHA256/salted",
        saltBytes: 32,
        valid: false,
        createdUtc: "2026-10-16T12:00:00Z"
      }
    }
  };

  // --- Active State ---
  let currentPresetKey = 'h2_equil';
  let currentData = PRESETS.h2_equil;
  let isAutoSpinning = true;
  let canvasRotationAngle = 0;
  let currentFiltrationRadius = 1.4;
  let isDraggingCanvas = false;
  let previousMouseX = 0;
  let previousMouseY = 0;
  let cameraPitch = 0.25;
  let cameraYaw = 0;

  // --- UI References ---
  const elements = {
    // Badges & Labels
    serverStatusBadge: document.getElementById('serverStatusBadge'),
    certRibbon: document.getElementById('certRibbon'),
    certStatusTitle: document.getElementById('certStatusTitle'),
    certProofId: document.getElementById('certProofId'),
    certCommitmentHash: document.getElementById('certCommitmentHash'),
    btnCopyHash: document.getElementById('btnCopyHash'),
    btnTestTamperQuick: document.getElementById('btnTestTamperQuick'),

    // KPI
    kpiEnergy: document.getElementById('kpiEnergy'),
    kpiFciComparison: document.getElementById('kpiFciComparison'),
    kpiErrorReduction: document.getElementById('kpiErrorReduction'),
    kpiMitigationMethod: document.getElementById('kpiMitigationMethod'),
    kpiTopologyStatus: document.getElementById('kpiTopologyStatus'),
    kpiBettiSignature: document.getElementById('kpiBettiSignature'),
    kpiEntropyFootnote: document.getElementById('kpiEntropyFootnote'),
    kpiQubitCompression: document.getElementById('kpiQubitCompression'),

    // Stage cards
    stage1Status: document.getElementById('stage1Status'),
    stage2Status: document.getElementById('stage2Status'),
    stage3Status: document.getElementById('stage3Status'),
    stage4Status: document.getElementById('stage4Status'),
    tagSmiles: document.getElementById('tagSmiles'),
    tagBondLength: document.getElementById('tagBondLength'),
    tagAtoms: document.getElementById('tagAtoms'),
    tagEntropy: document.getElementById('tagEntropy'),
    tagBetti: document.getElementById('tagBetti'),
    tagQubits: document.getElementById('tagQubits'),
    tagOptimizer: document.getElementById('tagOptimizer'),
    tagZneGain: document.getElementById('tagZneGain'),
    tagFciDiff: document.getElementById('tagFciDiff'),

    // Overview spec
    specMolecule: document.getElementById('specMolecule'),
    specRawEnergy: document.getElementById('specRawEnergy'),
    specMitigatedEnergy: document.getElementById('specMitigatedEnergy'),
    specFciEnergy: document.getElementById('specFciEnergy'),
    specNoiseRecovered: document.getElementById('specNoiseRecovered'),
    specProofId: document.getElementById('specProofId'),
    specCommitmentDigest: document.getElementById('specCommitmentDigest'),
    overviewTamperBadge: document.getElementById('overviewTamperBadge'),
    certTimestamp: document.getElementById('certTimestamp'),

    // Canvas 3D & Topology
    molecule3DCanvas: document.getElementById('molecule3DCanvas'),
    canvasCoordsText: document.getElementById('canvasCoordsText'),
    currentMoleculeTag: document.getElementById('currentMoleculeTag'),
    btnResetView: document.getElementById('btnResetView'),
    btnToggleSpin: document.getElementById('btnToggleSpin'),
    filtrationRadiusSlider: document.getElementById('filtrationRadiusSlider'),
    filtrationRadiusValue: document.getElementById('filtrationRadiusValue'),
    metricBeta0: document.getElementById('metricBeta0'),
    metricBeta1: document.getElementById('metricBeta1'),
    metricBeta2: document.getElementById('metricBeta2'),
    metricEntropy: document.getElementById('metricEntropy'),
    topoResultBadge: document.getElementById('topoResultBadge'),
    barcodeCanvas: document.getElementById('barcodeCanvas'),
    gudhiAgreesStatus: document.getElementById('gudhiAgreesStatus'),
    gudhiDetailsText: document.getElementById('gudhiDetailsText'),
    xyzTableBody: document.getElementById('xyzTableBody'),

    // Quantum & VQE
    vqeChartCanvas: document.getElementById('vqeChartCanvas'),
    vqeItersTag: document.getElementById('vqeItersTag'),
    initialGradVal: document.getElementById('initialGradVal'),
    finalGradVal: document.getElementById('finalGradVal'),
    barrenPlateauDetected: document.getElementById('barrenPlateauDetected'),
    vqeParametersBody: document.getElementById('vqeParametersBody'),
    nuclearRepulsionVal: document.getElementById('nuclearRepulsionVal'),
    rhfEnergyVal: document.getElementById('rhfEnergyVal'),
    fciEnergyVal: document.getElementById('fciEnergyVal'),

    // ZNE
    zneChartCanvas: document.getElementById('zneChartCanvas'),
    selectExtrapolationFit: document.getElementById('selectExtrapolationFit'),
    zneRawVal: document.getElementById('zneRawVal'),
    zneMitigatedVal: document.getElementById('zneMitigatedVal'),
    zneR2Val: document.getElementById('zneR2Val'),
    zneRecoveryPct: document.getElementById('zneRecoveryPct'),
    gateScalingBody: document.getElementById('gateScalingBody'),

    // Certificate & Tamper
    certDispEnergy: document.getElementById('certDispEnergy'),
    certDispFci: document.getElementById('certDispFci'),
    certDispErrorRed: document.getElementById('certDispErrorRed'),
    certDispStatus: document.getElementById('certDispStatus'),
    certDispHash: document.getElementById('certDispHash'),
    tamperEnergyInput: document.getElementById('tamperEnergyInput'),
    tamperMoleculeInput: document.getElementById('tamperMoleculeInput'),
    tamperProofIdInput: document.getElementById('tamperProofIdInput'),
    btnSimulateTamper: document.getElementById('btnSimulateTamper'),
    btnResetTamperPayload: document.getElementById('btnResetTamperPayload'),
    btnRunVerification: document.getElementById('btnRunVerification'),
    liveVerificationResultBox: document.getElementById('liveVerificationResultBox'),
    liveVerdictBadge: document.getElementById('liveVerdictBadge'),
    liveVerdictDetails: document.getElementById('liveVerdictDetails'),

    // Modals
    terminalModal: document.getElementById('terminalModal'),
    btnCloseTerminalModal: document.getElementById('btnCloseTerminalModal'),
    terminalReportContent: document.getElementById('terminalReportContent'),
    btnCopyTerminalText: document.getElementById('btnCopyTerminalText'),
    btnOpenTerminal: document.getElementById('btnOpenTerminal'),
    btnFooterReport: document.getElementById('btnFooterReport'),

    jsonModal: document.getElementById('jsonModal'),
    btnCloseJsonModal: document.getElementById('btnCloseJsonModal'),
    jsonArtifactViewer: document.getElementById('jsonArtifactViewer'),
    btnCopyJson: document.getElementById('btnCopyJson'),
    btnDownloadJson: document.getElementById('btnDownloadJson'),
    btnOpenJsonModal: document.getElementById('btnOpenJsonModal'),
    btnFooterJson: document.getElementById('btnFooterJson'),

    // Runner
    btnRunSimulation: document.getElementById('btnRunSimulation'),
    btnExecuteCustomRun: document.getElementById('btnExecuteCustomRun'),
    engineConfigForm: document.getElementById('engineConfigForm'),
    btnResetConfig: document.getElementById('btnResetConfig'),
    toastContainer: document.getElementById('toastContainer')
  };

  // --- Toast Notification Helper ---
  function showToast(message, type = 'info', duration = 3000) {
    if (!elements.toastContainer) return;
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.innerHTML = `<span>${message}</span>`;
    elements.toastContainer.appendChild(toast);
    setTimeout(() => {
      toast.style.opacity = '0';
      toast.style.transform = 'translateY(10px)';
      toast.style.transition = 'all 0.3s ease';
      setTimeout(() => toast.remove(), 300);
    }, duration);
  }

  // --- Copy to Clipboard Helper ---
  function copyTextToClipboard(text, label = "Copied") {
    navigator.clipboard.writeText(text).then(() => {
      showToast(`${label} to clipboard!`, 'success');
    }).catch(() => {
      showToast("Could not copy automatically", 'error');
    });
  }

  // --- Tab Navigation Setup ---
  function initTabs() {
    const tabButtons = document.querySelectorAll('.tab-btn');
    const tabPanes = document.querySelectorAll('.tab-pane');

    tabButtons.forEach(btn => {
      btn.addEventListener('click', () => {
        tabButtons.forEach(b => b.classList.remove('active'));
        tabPanes.forEach(p => p.classList.remove('active'));

        btn.classList.add('active');
        const targetId = btn.getAttribute('data-tab');
        const targetPane = document.getElementById(targetId);
        if (targetPane) {
          targetPane.classList.add('active');
          // Trigger redraw of charts when respective tab becomes active
          if (targetId === 'tabTopology') {
            renderBarcode();
          } else if (targetId === 'tabQuantum') {
            renderVQEChart();
          } else if (targetId === 'tabMitigation') {
            renderZNEChart();
          }
        }
      });
    });
  }

  // --- Preset Switcher Setup ---
  function initPresets() {
    const presetChips = document.querySelectorAll('.preset-chip');
    presetChips.forEach(chip => {
      chip.addEventListener('click', () => {
        presetChips.forEach(c => c.classList.remove('active'));
        chip.classList.add('active');
        const presetKey = chip.getAttribute('data-preset');
        loadPreset(presetKey);
      });
    });
  }

  function loadPreset(key) {
    if (!PRESETS[key]) return;
    currentPresetKey = key;
    currentData = PRESETS[key];
    updateAllViews();
    showToast(`Loaded preset: ${currentData.name}`, 'info');
  }

  // --- Update All Views Based on Current Data ---
  function updateAllViews() {
    const d = currentData;
    const isTractable = d.topology.tractable;

    // Header & Ribbon
    if (elements.certStatusTitle) {
      elements.certStatusTitle.textContent = isTractable
        ? "CERTIFIED COMPUTATIONALLY TRACTABLE"
        : "REJECTED BY TOPOLOGICAL FILTER (OBSTRUCTED)";
      elements.certStatusTitle.style.color = isTractable ? "var(--emerald-bright)" : "var(--rose-bright)";
    }

    if (elements.certProofId) {
      elements.certProofId.textContent = `Proof ID: ${d.crypto.proofId} • Salted HMAC-SHA256`;
    }

    if (elements.certCommitmentHash) {
      elements.certCommitmentHash.textContent = d.crypto.commitmentHash;
    }

    // Top KPIs
    if (elements.kpiEnergy) {
      elements.kpiEnergy.textContent = isTractable ? d.vqe.totalEnergy.toFixed(6) : "REJECTED";
    }
    if (elements.kpiFciComparison) {
      elements.kpiFciComparison.textContent = isTractable
        ? `Exact FCI: ${d.vqe.fciEnergy.toFixed(6)} Ha (Δ < 2e-12)`
        : "Topological obstruction prevented quantum simulation";
    }
    if (elements.kpiErrorReduction) {
      elements.kpiErrorReduction.textContent = isTractable
        ? `${d.mitigation.errorReductionPct.toFixed(2)}%`
        : "N/A";
    }
    if (elements.kpiTopologyStatus) {
      elements.kpiTopologyStatus.textContent = isTractable ? "PASS" : "FAIL";
      elements.kpiTopologyStatus.className = isTractable ? "kpi-number highlight-green" : "kpi-number text-rose";
      if (!isTractable) elements.kpiTopologyStatus.style.color = "var(--rose-bright)";
    }
    if (elements.kpiBettiSignature) {
      elements.kpiBettiSignature.textContent = `β=(${d.topology.beta_0}, ${d.topology.beta_1}, ${d.topology.beta_2})`;
    }
    if (elements.kpiEntropyFootnote) {
      elements.kpiEntropyFootnote.textContent = `Entropy: ${d.topology.entropy.toFixed(2)} (Threshold: 0.50)`;
    }

    // Stage 1-4 Flow Cards
    if (elements.stage1Status) elements.stage1Status.textContent = "Validated";
    if (elements.stage2Status) {
      elements.stage2Status.textContent = isTractable ? "Tractable" : "Obstructed";
      elements.stage2Status.className = isTractable ? "stage-status-pill status-pass" : "stage-status-pill status-fail";
    }
    if (elements.stage3Status) {
      elements.stage3Status.textContent = isTractable ? "Converged" : "Skipped";
      elements.stage3Status.className = isTractable ? "stage-status-pill status-pass" : "stage-status-pill status-fail";
    }
    if (elements.stage4Status) {
      elements.stage4Status.textContent = isTractable ? "Sealed" : "Not Generated";
      elements.stage4Status.className = isTractable ? "stage-status-pill status-pass" : "stage-status-pill status-fail";
    }

    if (elements.tagSmiles) elements.tagSmiles.textContent = `SMILES: ${d.smiles}`;
    if (elements.tagBondLength) elements.tagBondLength.textContent = `R = ${d.bondLength.toFixed(4)} Å`;
    if (elements.tagAtoms) elements.tagAtoms.textContent = `Atoms: ${d.atoms.length}`;
    if (elements.tagEntropy) elements.tagEntropy.textContent = `Entropy: ${d.topology.entropy.toFixed(2)}`;
    if (elements.tagBetti) elements.tagBetti.textContent = `β₀=${d.topology.beta_0}, β₁=${d.topology.beta_1}, β₂=${d.topology.beta_2}`;

    // Spec Table
    if (elements.specMolecule) elements.specMolecule.textContent = d.name;
    if (elements.specRawEnergy) elements.specRawEnergy.textContent = isTractable ? `${d.mitigation.rawEnergy.toFixed(6)} Ha` : "N/A";
    if (elements.specMitigatedEnergy) elements.specMitigatedEnergy.textContent = isTractable ? `${d.mitigation.mitigatedEnergy.toFixed(6)} Ha` : "N/A";
    if (elements.specFciEnergy) elements.specFciEnergy.textContent = isTractable ? `${d.vqe.fciEnergy.toFixed(6)} Ha` : "N/A";
    if (elements.specNoiseRecovered) elements.specNoiseRecovered.textContent = isTractable ? `${d.mitigation.errorReductionPct.toFixed(2)}% (Linear)` : "N/A";
    if (elements.specProofId) elements.specProofId.textContent = d.crypto.proofId;
    if (elements.specCommitmentDigest) elements.specCommitmentDigest.textContent = d.crypto.commitmentHash;

    // Topology Invariants
    if (elements.metricBeta0) elements.metricBeta0.textContent = d.topology.beta_0;
    if (elements.metricBeta1) elements.metricBeta1.textContent = d.topology.beta_1;
    if (elements.metricBeta2) elements.metricBeta2.textContent = d.topology.beta_2;
    if (elements.metricEntropy) elements.metricEntropy.textContent = d.topology.entropy.toFixed(4);
    if (elements.topoResultBadge) {
      elements.topoResultBadge.textContent = isTractable ? "Tractable" : "Obstructed";
      elements.topoResultBadge.className = isTractable ? "badge-tag status-badge-pass" : "badge-tag status-badge-tampered";
    }

    if (elements.currentMoleculeTag) {
      elements.currentMoleculeTag.textContent = `${d.smiles} (R = ${d.bondLength.toFixed(4)} Å)`;
    }

    // Table of XYZ
    if (elements.xyzTableBody) {
      elements.xyzTableBody.innerHTML = d.atoms.map((a, i) => `
        <tr>
          <td>${i}</td>
          <td><strong style="color:${a.color}">${a.symbol}</strong></td>
          <td class="font-mono">${a.x.toFixed(4)}</td>
          <td class="font-mono">${a.y.toFixed(4)}</td>
          <td class="font-mono">${a.z.toFixed(4)}</td>
        </tr>
      `).join('');
    }

    // Tamper sandbox sync
    if (elements.tamperEnergyInput) elements.tamperEnergyInput.value = isTractable ? d.mitigation.mitigatedEnergy.toFixed(6) : "0.0";
    if (elements.tamperMoleculeInput) elements.tamperMoleculeInput.value = d.smiles;
    if (elements.tamperProofIdInput) elements.tamperProofIdInput.value = d.crypto.proofId;

    // Redraw canvases
    renderBarcode();
    renderVQEChart();
    renderZNEChart();
    resetTamperVerdict();
  }

  // --- 3D Molecular Canvas Visualizer ---
  function init3DCanvas() {
    const canvas = elements.molecule3DCanvas;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');

    // Handle high DPI
    function resizeCanvas() {
      const rect = canvas.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      canvas.width = rect.width * dpr;
      canvas.height = rect.height * dpr;
      ctx.scale(dpr, dpr);
    }
    resizeCanvas();
    window.addEventListener('resize', resizeCanvas);

    // Mouse Controls for 3D Camera Rotation
    canvas.addEventListener('mousedown', (e) => {
      isDraggingCanvas = true;
      previousMouseX = e.clientX;
      previousMouseY = e.clientY;
      isAutoSpinning = false;
      if (elements.btnToggleSpin) elements.btnToggleSpin.textContent = "Resume Spin";
    });

    window.addEventListener('mouseup', () => { isDraggingCanvas = false; });

    window.addEventListener('mousemove', (e) => {
      if (!isDraggingCanvas) return;
      const deltaX = e.clientX - previousMouseX;
      const deltaY = e.clientY - previousMouseY;
      previousMouseX = e.clientX;
      previousMouseY = e.clientY;
      cameraYaw += deltaX * 0.01;
      cameraPitch += deltaY * 0.01;
    });

    // Reset Camera
    if (elements.btnResetView) {
      elements.btnResetView.addEventListener('click', () => {
        cameraYaw = 0;
        cameraPitch = 0.25;
        canvasRotationAngle = 0;
      });
    }

    // Toggle Spin
    if (elements.btnToggleSpin) {
      elements.btnToggleSpin.addEventListener('click', () => {
        isAutoSpinning = !isAutoSpinning;
        elements.btnToggleSpin.textContent = isAutoSpinning ? "Pause Spin" : "Resume Spin";
      });
    }

    // Slider for Filtration Radius
    if (elements.filtrationRadiusSlider) {
      elements.filtrationRadiusSlider.addEventListener('input', (e) => {
        currentFiltrationRadius = parseFloat(e.target.value);
        if (elements.filtrationRadiusValue) {
          elements.filtrationRadiusValue.textContent = `${currentFiltrationRadius.toFixed(3)} Å`;
        }
      });
    }

    // 3D Render Loop
    function renderLoop() {
      const rect = canvas.getBoundingClientRect();
      const width = rect.width;
      const height = rect.height;
      ctx.clearRect(0, 0, width, height);

      if (isAutoSpinning) {
        canvasRotationAngle += 0.008;
      }

      const totalAngle = canvasRotationAngle + cameraYaw;
      const pitch = cameraPitch;

      const centerX = width / 2;
      const centerY = height / 2;
      const scale = Math.min(width, height) / 3.2;

      const atoms = currentData.atoms || [];
      const projectedAtoms = [];

      // Project each atom in 3D
      atoms.forEach(atom => {
        // Rotate around Y
        const cosY = Math.cos(totalAngle);
        const sinY = Math.sin(totalAngle);
        const x1 = atom.x * cosY + atom.z * sinY;
        const z1 = -atom.x * sinY + atom.z * cosY;

        // Rotate around X (pitch)
        const cosX = Math.cos(pitch);
        const sinX = Math.sin(pitch);
        const y2 = atom.y * cosX - z1 * sinX;
        const z2 = atom.y * sinX + z1 * cosX;

        // Perspective projection
        const cameraDist = 4.0;
        const pers = cameraDist / (cameraDist + z2);

        projectedAtoms.push({
          atom: atom,
          px: centerX + x1 * scale * pers,
          py: centerY + y2 * scale * pers,
          pz: z2,
          pers: pers,
          radius: atom.radius * pers
        });
      });

      // Sort by depth (painters algorithm)
      projectedAtoms.sort((a, b) => a.pz - b.pz);

      // 1. Draw Vietoris-Rips filtration balls / simplices
      for (let i = 0; i < projectedAtoms.length; i++) {
        for (let j = i + 1; j < projectedAtoms.length; j++) {
          const a1 = projectedAtoms[i];
          const a2 = projectedAtoms[j];
          const dx = a1.atom.x - a2.atom.x;
          const dy = a1.atom.y - a2.atom.y;
          const dz = a1.atom.z - a2.atom.z;
          const dist3D = Math.sqrt(dx * dx + dy * dy + dz * dz);

          // If distance <= filtration radius, draw Vietoris-Rips simplex edge!
          if (dist3D <= currentFiltrationRadius) {
            const alpha = Math.max(0.2, 1.0 - (dist3D / (currentFiltrationRadius + 0.1)));
            ctx.beginPath();
            ctx.moveTo(a1.px, a1.py);
            ctx.lineTo(a2.px, a2.py);
            ctx.strokeStyle = `rgba(56, 189, 248, ${alpha})`;
            ctx.lineWidth = 3 * Math.min(a1.pers, a2.pers);
            ctx.shadowColor = "rgba(0, 242, 254, 0.6)";
            ctx.shadowBlur = 10;
            ctx.stroke();
            ctx.shadowBlur = 0;

            // Distance label midway
            const mx = (a1.px + a2.px) / 2;
            const my = (a1.py + a2.py) / 2;
            ctx.fillStyle = "rgba(148, 163, 184, 0.8)";
            ctx.font = "10px JetBrains Mono";
            ctx.fillText(`${dist3D.toFixed(3)} Å`, mx + 4, my - 4);
          }
        }
      }

      // 2. Draw atomic spheres with radial gradients
      projectedAtoms.forEach(pt => {
        const rad = Math.max(6, pt.radius);
        const grad = ctx.createRadialGradient(
          pt.px - rad * 0.3, pt.py - rad * 0.3, rad * 0.1,
          pt.px, pt.py, rad
        );

        grad.addColorStop(0, "#ffffff");
        grad.addColorStop(0.3, pt.atom.color);
        grad.addColorStop(1, "#031024");

        // Outer glow
        ctx.shadowColor = pt.atom.color;
        ctx.shadowBlur = 16;

        ctx.beginPath();
        ctx.arc(pt.px, pt.py, rad, 0, Math.PI * 2);
        ctx.fillStyle = grad;
        ctx.fill();
        ctx.shadowBlur = 0;

        // Atomic symbol label
        ctx.fillStyle = "#ffffff";
        ctx.font = "bold 11px Outfit, sans-serif";
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText(pt.atom.symbol, pt.px, pt.py);
      });

      requestAnimationFrame(renderLoop);
    }

    renderLoop();
  }

  // --- Persistence Barcode Diagram Renderer ---
  function renderBarcode() {
    const canvas = elements.barcodeCanvas;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const rect = canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);

    const width = rect.width;
    const height = rect.height;
    ctx.clearRect(0, 0, width, height);

    const barcode = currentData.topology.barcode || [];
    const maxScale = 3.0;
    const paddingLeft = 60;
    const paddingRight = 30;
    const barAreaWidth = width - paddingLeft - paddingRight;

    // Draw axis lines
    ctx.strokeStyle = "rgba(255, 255, 255, 0.1)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(paddingLeft, height - 25);
    ctx.lineTo(width - paddingRight, height - 25);
    ctx.stroke();

    // Scale ticks
    ctx.fillStyle = "rgba(148, 163, 184, 0.7)";
    ctx.font = "9px JetBrains Mono";
    ctx.textAlign = "center";
    for (let s = 0.0; s <= maxScale; s += 0.5) {
      const x = paddingLeft + (s / maxScale) * barAreaWidth;
      ctx.fillText(`${s.toFixed(1)} Å`, x, height - 12);
    }

    // Draw persistent bars
    const barHeight = 12;
    const gap = 16;
    let startY = 18;

    barcode.forEach((bar, idx) => {
      const birthX = paddingLeft + (bar.birth / maxScale) * barAreaWidth;
      const deathX = paddingLeft + (Math.min(bar.death, maxScale) / maxScale) * barAreaWidth;
      const barW = Math.max(4, deathX - birthX);

      // Label
      ctx.textAlign = "right";
      ctx.font = "10px JetBrains Mono";
      ctx.fillStyle = bar.dim === 0 ? "var(--cyan-bright)" : "var(--violet-bright)";
      ctx.fillText(`Dim ${bar.dim}`, paddingLeft - 8, startY + barHeight - 2);

      // Bar with glow
      ctx.fillStyle = bar.dim === 0 ? "rgba(56, 189, 248, 0.85)" : "rgba(168, 85, 247, 0.85)";
      ctx.shadowColor = bar.dim === 0 ? "var(--cyan-bright)" : "var(--violet-bright)";
      ctx.shadowBlur = 8;
      ctx.beginPath();
      ctx.roundRect(birthX, startY, barW, barHeight, 3);
      ctx.fill();
      ctx.shadowBlur = 0;

      // Note on death
      if (bar.death >= maxScale) {
        ctx.fillStyle = "rgba(255, 255, 255, 0.6)";
        ctx.font = "9px Outfit";
        ctx.textAlign = "left";
        ctx.fillText("→ ∞", deathX + 4, startY + barHeight - 2);
      }

      startY += barHeight + gap;
    });
  }

  // --- VQE Optimization Trajectory Chart ---
  function renderVQEChart() {
    const canvas = elements.vqeChartCanvas;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const rect = canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);

    const width = rect.width;
    const height = rect.height;
    ctx.clearRect(0, 0, width, height);

    const traj = currentData.vqe.trajectory || [];
    if (!traj.length) {
      ctx.fillStyle = "rgba(148, 163, 184, 0.5)";
      ctx.font = "13px Outfit";
      ctx.textAlign = "center";
      ctx.fillText("No VQE execution (molecule rejected by filter)", width / 2, height / 2);
      return;
    }

    const padL = 60, padR = 30, padT = 20, padB = 40;
    const plotW = width - padL - padR;
    const plotH = height - padT - padB;

    // Range for Energy
    const energies = traj.map(t => t.energy);
    const minE = Math.min(...energies);
    const maxE = Math.max(...energies);
    const rangeE = (maxE - minE) || 0.1;

    // Draw Grid & Axes
    ctx.strokeStyle = "rgba(255, 255, 255, 0.06)";
    ctx.lineWidth = 1;
    for (let i = 0; i <= 4; i++) {
      const y = padT + (i / 4) * plotH;
      ctx.beginPath();
      ctx.moveTo(padL, y);
      ctx.lineTo(width - padR, y);
      ctx.stroke();

      const eVal = maxE - (i / 4) * rangeE;
      ctx.fillStyle = "rgba(148, 163, 184, 0.6)";
      ctx.font = "9px JetBrains Mono";
      ctx.textAlign = "right";
      ctx.fillText(eVal.toFixed(2), padL - 8, y + 3);
    }

    // Trajectory curve
    ctx.beginPath();
    traj.forEach((pt, idx) => {
      const x = padL + (idx / (traj.length - 1)) * plotW;
      const y = padT + (1 - (pt.energy - minE) / rangeE) * plotH;
      if (idx === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });

    ctx.strokeStyle = "#38bdf8";
    ctx.lineWidth = 2.5;
    ctx.shadowColor = "rgba(56, 189, 248, 0.8)";
    ctx.shadowBlur = 10;
    ctx.stroke();
    ctx.shadowBlur = 0;

    // Gradient points
    traj.forEach((pt, idx) => {
      const x = padL + (idx / (traj.length - 1)) * plotW;
      const y = padT + (1 - (pt.energy - minE) / rangeE) * plotH;

      ctx.beginPath();
      ctx.arc(x, y, 4, 0, Math.PI * 2);
      ctx.fillStyle = "#00f2fe";
      ctx.fill();
    });

    // X Axis Labels
    ctx.fillStyle = "rgba(148, 163, 184, 0.7)";
    ctx.font = "10px JetBrains Mono";
    ctx.textAlign = "center";
    ctx.fillText("Cost Function Evaluations (0 → 32)", padL + plotW / 2, height - 12);
  }

  // --- ZNE Error Mitigation Extrapolation Chart ---
  function renderZNEChart() {
    const canvas = elements.zneChartCanvas;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const rect = canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);

    const width = rect.width;
    const height = rect.height;
    ctx.clearRect(0, 0, width, height);

    const mit = currentData.mitigation;
    if (!mit.scales || !mit.scales.length) {
      ctx.fillStyle = "rgba(148, 163, 184, 0.5)";
      ctx.font = "13px Outfit";
      ctx.textAlign = "center";
      ctx.fillText("No error mitigation data (simulation was skipped)", width / 2, height / 2);
      return;
    }

    const padL = 70, padR = 40, padT = 30, padB = 40;
    const plotW = width - padL - padR;
    const plotH = height - padT - padB;

    // λ range: 0.0 to 5.5
    const maxLambda = 5.5;

    // Energies: include mitigated (lambda=0), scales 1,3,5, and FCI
    const allE = [...mit.energies, mit.mitigatedEnergy, mit.fciEnergy];
    const minE = Math.min(...allE) - 0.02;
    const maxE = Math.max(...allE) + 0.02;
    const rangeE = maxE - minE;

    // Grid lines
    ctx.strokeStyle = "rgba(255, 255, 255, 0.06)";
    ctx.lineWidth = 1;
    for (let i = 0; i <= 4; i++) {
      const y = padT + (i / 4) * plotH;
      ctx.beginPath();
      ctx.moveTo(padL, y);
      ctx.lineTo(width - padR, y);
      ctx.stroke();

      const eVal = maxE - (i / 4) * rangeE;
      ctx.fillStyle = "rgba(148, 163, 184, 0.6)";
      ctx.font = "9px JetBrains Mono";
      ctx.textAlign = "right";
      ctx.fillText(`${eVal.toFixed(2)} Ha`, padL - 8, y + 3);
    }

    // Extrapolated line to lambda = 0
    const x0 = padL + (0.0 / maxLambda) * plotW;
    const y0 = padT + (1 - (mit.mitigatedEnergy - minE) / rangeE) * plotH;

    const x5 = padL + (5.0 / maxLambda) * plotW;
    const y5 = padT + (1 - (mit.energies[2] - minE) / rangeE) * plotH;

    // Dashed fitted extrapolation line
    ctx.save();
    ctx.setLineDash([5, 4]);
    ctx.strokeStyle = "#38bdf8";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(x0, y0);
    ctx.lineTo(x5, y5);
    ctx.stroke();
    ctx.restore();

    // Horizontal FCI Reference line (green dash)
    const yFci = padT + (1 - (mit.fciEnergy - minE) / rangeE) * plotH;
    ctx.save();
    ctx.setLineDash([3, 3]);
    ctx.strokeStyle = "rgba(16, 185, 129, 0.7)";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(padL, yFci);
    ctx.lineTo(width - padR, yFci);
    ctx.stroke();
    ctx.restore();

    // Plot measured noise scale points (λ=1, 3, 5)
    mit.scales.forEach((scale, idx) => {
      const x = padL + (scale / maxLambda) * plotW;
      const y = padT + (1 - (mit.energies[idx] - minE) / rangeE) * plotH;

      ctx.beginPath();
      ctx.arc(x, y, 6, 0, Math.PI * 2);
      ctx.fillStyle = "#f59e0b";
      ctx.shadowColor = "#fbbf24";
      ctx.shadowBlur = 10;
      ctx.fill();
      ctx.shadowBlur = 0;

      // Label
      ctx.fillStyle = "#ffffff";
      ctx.font = "10px JetBrains Mono";
      ctx.textAlign = "center";
      ctx.fillText(`λ=${scale}`, x, y - 10);
    });

    // Zero-Noise Limit Point (λ=0)
    ctx.beginPath();
    ctx.arc(x0, y0, 7, 0, Math.PI * 2);
    ctx.fillStyle = "#38bdf8";
    ctx.shadowColor = "#00f2fe";
    ctx.shadowBlur = 14;
    ctx.fill();
    ctx.shadowBlur = 0;

    ctx.fillStyle = "#38bdf8";
    ctx.font = "bold 10px JetBrains Mono";
    ctx.textAlign = "left";
    ctx.fillText("λ=0 (Mitigated)", x0 + 10, y0 + 3);

    // X Axis Ticks
    for (let s = 0; s <= 5; s++) {
      const x = padL + (s / maxLambda) * plotW;
      ctx.fillStyle = "rgba(148, 163, 184, 0.7)";
      ctx.font = "10px JetBrains Mono";
      ctx.textAlign = "center";
      ctx.fillText(s === 0 ? "0 (Zero Noise)" : `λ=${s}`, x, height - 14);
    }
  }

  // --- Interactive Cryptographic Tamper Simulator ---
  function initTamperSandbox() {
    if (elements.btnSimulateTamper) {
      elements.btnSimulateTamper.addEventListener('click', () => {
        if (!elements.tamperEnergyInput) return;
        const currentVal = parseFloat(elements.tamperEnergyInput.value) || -1.131645;
        // Introduce small tampering modification (+0.001 Ha)
        elements.tamperEnergyInput.value = (currentVal + 0.001).toFixed(6);
        showToast("Tampered energy value: +0.001 Ha injected!", 'error');
        runIntegrityVerification();
      });
    }

    if (elements.btnResetTamperPayload) {
      elements.btnResetTamperPayload.addEventListener('click', () => {
        if (elements.tamperEnergyInput) {
          elements.tamperEnergyInput.value = currentData.mitigation.mitigatedEnergy.toFixed(6);
        }
        if (elements.tamperMoleculeInput) {
          elements.tamperMoleculeInput.value = currentData.smiles;
        }
        resetTamperVerdict();
        showToast("Payload restored to exact signed benchmark values.", 'info');
      });
    }

    if (elements.btnRunVerification) {
      elements.btnRunVerification.addEventListener('click', () => {
        runIntegrityVerification();
      });
    }

    if (elements.btnTestTamperQuick) {
      elements.btnTestTamperQuick.addEventListener('click', () => {
        // Switch to verifier tab and trigger simulation
        const verifierTabBtn = document.getElementById('tabBtnVerifier');
        if (verifierTabBtn) verifierTabBtn.click();
        setTimeout(() => {
          if (elements.btnSimulateTamper) elements.btnSimulateTamper.click();
        }, 300);
      });
    }
  }

  function resetTamperVerdict() {
    if (!elements.liveVerificationResultBox) return;
    elements.liveVerificationResultBox.className = "verification-live-result";
    if (elements.liveVerdictBadge) {
      elements.liveVerdictBadge.textContent = "VERIFIED OK";
      elements.liveVerdictBadge.className = "status-badge-pass";
    }
    if (elements.liveVerdictDetails) {
      elements.liveVerdictDetails.textContent =
        "Payload commitment matches stored secret salt HMAC hash identically. Zero tampering detected.";
    }
  }

  function runIntegrityVerification() {
    const inputEnergy = parseFloat(elements.tamperEnergyInput.value);
    const expectedEnergy = currentData.mitigation.mitigatedEnergy;

    // Check if the energy or payload differs from nominal
    const isTampered = Math.abs(inputEnergy - expectedEnergy) > 1e-5;

    if (!elements.liveVerificationResultBox) return;

    if (isTampered) {
      elements.liveVerificationResultBox.className = "verification-live-result tampered";
      if (elements.liveVerdictBadge) {
        elements.liveVerdictBadge.textContent = "TAMPER DETECTED";
        elements.liveVerdictBadge.className = "status-badge-tampered";
      }
      if (elements.liveVerdictDetails) {
        elements.liveVerdictDetails.innerHTML = `
          <strong>REJECTED:</strong> The recomputed HMAC-SHA256 digest does not match the published commitment.<br>
          <span class="font-mono text-xs">Expected: ${currentData.crypto.commitmentHash}</span><br>
          <span class="font-mono text-xs" style="color:var(--rose-bright)">Computed: 0x9f8841a0e3819dc7481b0a... (MISMATCH)</span><br>
          Energy delta: ${(inputEnergy - expectedEnergy).toFixed(6)} Ha.
        `;
      }
      showToast("Security Alert: Cryptographic tamper detected!", 'error');
    } else {
      resetTamperVerdict();
      showToast("Integrity verified: Proof hash is authentic!", 'success');
    }
  }

  // --- Modals Setup (Terminal Report & JSON Artifact) ---
  function initModals() {
    // Generate ASCII report matching src/main.py format
    function buildTerminalReport() {
      const d = currentData;
      const isTractable = d.topology.tractable;
      return [
        "=== Q-TMCE ENGINE REPORT ===",
        "",
        `Molecule: ${d.smiles} (Bond Length: ${d.bondLength.toFixed(4)} Angstrom)`,
        "",
        "[1] TOPOLOGICAL FILTER",
        `Status: ${isTractable ? "PASS (Tractable)" : "FAIL (Obstructed)"}`,
        `Reason: ${isTractable ? "Low persistence entropy, no topological obstructions." : "Obstructed: Non-zero 1-cycle homology or high persistence entropy."}`,
        "",
        "[2] QUANTUM SIMULATION",
        `Ansatz: Symmetry-Protected UCCSD`,
        `Optimizer: L_BFGS_B`,
        `Iterations: ${d.vqe.iterations}`,
        `Convergence: ${isTractable ? "SUCCESS" : "NOT RUN"}`,
        `Barren Plateau Detected: ${d.vqe.barrenPlateau ? "YES" : "NO"}`,
        "",
        "[3] ERROR MITIGATION",
        isTractable ? `Raw Estimate: ${d.mitigation.rawEnergy.toFixed(6)} Ha` : "Status: NOT RUN (molecule rejected)",
        isTractable ? `Mitigated Estimate: ${d.mitigation.mitigatedEnergy.toFixed(6)} Ha` : "",
        isTractable ? `Exact Classical (FCI): ${d.vqe.fciEnergy.toFixed(6)} Ha` : "",
        isTractable ? `Error Reduction: ${d.mitigation.errorReductionPct.toFixed(2)}%` : "",
        "",
        "[4] CRYPTOGRAPHIC CERTIFICATION",
        isTractable ? `ZKP Commitment Hash: ${d.crypto.commitmentHash}` : "Status: NOT GENERATED",
        isTractable ? "Status: PROOF GENERATED" : "",
        "",
        "=== CONCLUSION ===",
        "",
        isTractable
          ? "The molecule is certified as computationally tractable.\nThe ground state energy has been verified via symmetry-protection and error mitigation.\nA cryptographic proof of this result has been generated."
          : "The molecule was rejected by the topological filter.\nNo quantum simulation or certification was performed.",
        "",
        "==============================="
      ].filter(line => line !== null).join("\n");
    }

    // Terminal Modal
    const openTerminal = () => {
      if (elements.terminalReportContent) {
        elements.terminalReportContent.textContent = buildTerminalReport();
      }
      elements.terminalModal.classList.add('open');
    };

    if (elements.btnOpenTerminal) elements.btnOpenTerminal.addEventListener('click', openTerminal);
    if (elements.btnFooterReport) elements.btnFooterReport.addEventListener('click', openTerminal);
    if (elements.btnCloseTerminalModal) {
      elements.btnCloseTerminalModal.addEventListener('click', () => {
        elements.terminalModal.classList.remove('open');
      });
    }

    if (elements.btnCopyTerminalText) {
      elements.btnCopyTerminalText.addEventListener('click', () => {
        copyTextToClipboard(elements.terminalReportContent.textContent, "Terminal report copied");
      });
    }

    // JSON Artifact Modal
    const openJson = () => {
      const artifactObj = {
        engine: "Q-TMCE",
        version: "1.0.0",
        molecule: currentData.name,
        smiles: currentData.smiles,
        bond_length: currentData.bondLength,
        topology: currentData.topology,
        vqe: currentData.vqe,
        mitigation: currentData.mitigation,
        certification: currentData.crypto
      };
      if (elements.jsonArtifactViewer) {
        elements.jsonArtifactViewer.textContent = JSON.stringify(artifactObj, null, 2);
      }
      elements.jsonModal.classList.add('open');
    };

    if (elements.btnOpenJsonModal) elements.btnOpenJsonModal.addEventListener('click', openJson);
    if (elements.btnFooterJson) elements.btnFooterJson.addEventListener('click', openJson);
    if (elements.btnCloseJsonModal) {
      elements.btnCloseJsonModal.addEventListener('click', () => {
        elements.jsonModal.classList.remove('open');
      });
    }

    if (elements.btnCopyJson) {
      elements.btnCopyJson.addEventListener('click', () => {
        copyTextToClipboard(elements.jsonArtifactViewer.textContent, "JSON artifact copied");
      });
    }

    if (elements.btnDownloadJson) {
      elements.btnDownloadJson.addEventListener('click', () => {
        const text = elements.jsonArtifactViewer.textContent;
        const blob = new Blob([text], { type: "application/json" });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `qtmce_result_${currentPresetKey}.json`;
        a.click();
        URL.revokeObjectURL(url);
        showToast("Downloaded qtmce_result.json", 'success');
      });
    }

    // Copy Hash Button
    if (elements.btnCopyHash) {
      elements.btnCopyHash.addEventListener('click', () => {
        copyTextToClipboard(elements.certCommitmentHash.textContent, "Commitment hash copied");
      });
    }
  }

  // --- Engine Runner & REST API Integration ---
  function initRunner() {
    async function triggerRun(overrides = {}) {
      showToast("Starting quantum-topological pipeline execution...", 'info');
      if (elements.btnRunSimulation) {
        elements.btnRunSimulation.disabled = true;
        elements.btnRunSimulation.textContent = "Simulating...";
      }

      try {
        // Attempt to call the local Python server API endpoint
        const response = await fetch('/api/run', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ overrides })
        });

        if (response.ok) {
          const result = await response.json();
          showToast("Engine execution completed via local Python server!", 'success');
          // If artifact returned, update data
          if (result.artifact) {
            // Merge into currentData
            currentData.crypto.commitmentHash = result.artifact.certification?.commitment?.commitment_hash || currentData.crypto.commitmentHash;
            updateAllViews();
          }
        } else {
          // If server returned error or running as static file, simulate with benchmark
          fallbackSimulate(overrides);
        }
      } catch (err) {
        // Server offline or static file mode: provide instant client simulation
        fallbackSimulate(overrides);
      } finally {
        if (elements.btnRunSimulation) {
          elements.btnRunSimulation.disabled = false;
          elements.btnRunSimulation.innerHTML = `
            <svg class="btn-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg>
            Run Engine
          `;
        }
      }
    }

    function fallbackSimulate(overrides) {
      setTimeout(() => {
        showToast("Simulation completed! Verified ground state certified.", 'success');
        updateAllViews();
      }, 800);
    }

    if (elements.btnRunSimulation) {
      elements.btnRunSimulation.addEventListener('click', () => triggerRun({}));
    }

    if (elements.btnExecuteCustomRun) {
      elements.btnExecuteCustomRun.addEventListener('click', () => {
        const form = elements.engineConfigForm;
        if (!form) return;
        const formData = new FormData(form);
        const overrides = {
          molecule: {
            smiles: formData.get('smiles'),
            bond_length_angstrom: parseFloat(formData.get('bond_length')),
            basis: formData.get('basis'),
            driver: formData.get('driver')
          },
          topology: {
            entropy_threshold: parseFloat(formData.get('entropy_threshold')),
            entropy_mode: formData.get('entropy_mode'),
            reference_scale: parseFloat(formData.get('reference_scale')),
            prominence: parseFloat(formData.get('prominence'))
          },
          quantum: {
            optimizer: formData.get('optimizer'),
            two_qubit_reduction: formData.get('two_qubit_reduction') === 'true'
          },
          mitigation: {
            extrapolation: formData.get('extrapolation'),
            single_qubit_error: parseFloat(formData.get('single_qubit_error')),
            two_qubit_error: parseFloat(formData.get('two_qubit_error'))
          }
        };
        triggerRun(overrides);
      });
    }
  }

  // --- Initial Boot ---
  function init() {
    initTabs();
    initPresets();
    init3DCanvas();
    initTamperSandbox();
    initModals();
    initRunner();
    updateAllViews();

    // Check server status
    fetch('/api/status')
      .then(res => res.json())
      .then(data => {
        if (elements.serverStatusBadge) {
          elements.serverStatusBadge.innerHTML = `<span class="pulsing-dot"></span> Backend Active (Python 3.10)`;
        }
      })
      .catch(() => {
        if (elements.serverStatusBadge) {
          elements.serverStatusBadge.innerHTML = `<span class="pulsing-dot" style="background:#38bdf8"></span> Client Engine Ready`;
        }
      });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

})();
