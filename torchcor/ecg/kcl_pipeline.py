"""
torchcor/ecg/kcl_pipeline.py
=============================
Full KCL whole-torso cardiac simulation pipeline:

  KCL anatomy VTU  +  CARP mesh
        │
        ▼  kcl_extract.py  (one-time)
  Simulation-ready CARP sub-meshes
        │
        ▼  Monodomain  (torchcor.simulator)
  Transmembrane potential  Vm(t)  on biventricular mesh
        │
        ▼  ECG Forward solve  (lead field method, torchcor.ecg.test)
  12-lead ECG  +  clinical layout plot

KCL CARP tag convention
-----------------------
  1 – 25 : anatomy
  26 – 33: ICD hardware  (excluded)
  34     : RV wall   (bidomain)
  35     : LV wall   (bidomain)
  24     : RV blood pool  (isotropic)
  25     : LV blood pool  (isotropic)

Usage
-----
    python -m torchcor.ecg.kcl_pipeline
    python -m torchcor.ecg.kcl_pipeline --device cuda:0 --T 600 --dt 0.05
    python -m torchcor.ecg.kcl_pipeline --skip-extract --skip-sim
"""

from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import torch

warnings.filterwarnings("ignore", message="Sparse CSR tensor support is in beta state")

# ── Default paths ──────────────────────────────────────────────────────────────
_KCL_DIR        = Path(r"C:\Users\bulli\cardiac_data\kcl_torso\KCL_torso3")
DEFAULT_SIM_DIR = _KCL_DIR / "sim"
DEFAULT_RESULT  = DEFAULT_SIM_DIR / "results"

# ── Conductivity table (S/m) — KCL Qian et al. 2022 ──────────────────────────
#  Key: CARP region tag
#  Value: isotropic scalar (torso) or (il, it, el, et) tuple (myocardium)
CONDUCTIVITY: dict[int, float | tuple] = {
    1:  0.247,   # Inner body / bulk soft tissue
    2:  0.117,   # Skin
    3:  0.050,   # Bones
    4:  0.167,   # Kidneys
    5:  0.167,   # Liver
    6:  0.100,   # Stomach
    7:  0.100,   # Spleen
    8:  0.071,   # Lungs
    9:  0.667,   # SVC blood pool
    10: 0.667,   # Superior vena cava
    11: 0.667,   # Pulmonary artery pool
    12: 0.667,   # Pulmonary artery
    13: 0.667,   # Aorta
    14: 0.667,   # Aorta blood pool
    15: 0.667,   # SVC plane
    16: 0.667,   # Mitral valve
    17: 0.667,   # Pulmonary valve
    18: 0.667,   # Aortic valve
    19: 0.667,   # Tricuspid valve
    20: 0.250,   # L. atrial wall
    21: 0.667,   # L. atrial blood pool
    22: 0.667,   # R. atrial blood pool
    23: 0.250,   # R. atrial wall
    24: 0.667,   # RV blood pool
    25: 0.667,   # LV blood pool
    # 34 = RV wall, 35 = LV wall → bidomain below
    # 26-33 = ICD hardware → excluded by kcl_extract
}

# ==============================================================================
#  Ventricular myocardium bidomain conductivity presets  (all values in S/m)
#
#  Literature reference table  (1 mS/cm = 0.1 S/m):
#
#  Preset               σiL    σiT    σeL    σeT   Notes
#  ─────────────────────────────────────────────────────────────────────────────
#  clerc_1976          0.174  0.019  0.625  0.236  Calf RV trabeculae, cable
#                                                   analysis.  Foundational
#                                                   dataset; Ri≈9.2, Re≈2.6.
#  roberts_scher_1982  0.344  0.060  0.117  0.080  Canine LV in situ; produces
#                                                   σiL > σeL — opposite to
#                                                   Clerc.  Used in some CARP
#                                                   benchmarks.
#  roth_1997           0.300  0.030  0.300  0.120  Normalised dimensionless
#   = potse_2006                                    framework (Ri=10, Re=2.5).
#   (ACTIVE — default)                              Adopted verbatim by Potse
#                                                   (2006, 2018).  Most widely
#                                                   cited in ECG simulation.
#  stinstra_2005       0.160  0.005  0.210  0.060  Microstructural FEM
#                                                   homogenisation (rat). Lowest
#                                                   σiT in the literature.
#  hooks_2007          0.260  0.026  0.260  0.250  Ovine LV intramural mapping.
#                                                   Establishes 3-D ORTHOTROPIC
#                                                   conduction (fiber:laminar:
#                                                   normal ≈ 4:2:1).  σiT here
#                                                   is the laminar-transverse
#                                                   value; strict cross-laminar
#                                                   σiN ≈ 0.008 S/m.
#  patel_roth_2015     0.240  0.035  0.240  0.200  6-conductivity consensus
#                                                   set (Patel & Roth 2015,
#                                                   Ann. Biomed. Eng.).  Derived
#                                                   from Hooks 4:2:1 ratios +
#                                                   Clerc/Roberts constraint.
#                                                   Validated against ischaemia
#                                                   epicardial potential data.
#                                                   Most modern widely-adopted
#                                                   set; includes σiN=0.008,
#                                                   σeN=0.110 S/m (not yet used
#                                                   here — requires ortho mesh).
#  niederer_2011_mono  0.133  0.018    —      —    Canonical monodomain solver-
#                                                   benchmark cuboid; not a
#                                                   bidomain set.
#  ─────────────────────────────────────────────────────────────────────────────
#  PREVIOUS "demo" values (0.5272 / 0.2076 / 1.0732 / 0.4227) had NO literature
#  provenance — they were unvalidated placeholders roughly 2× too high.
# ==============================================================================

# Active preset: Roth (1997) / Potse (2006) — normalised dimensionless framework
# (Ri=10, Re=2.5; most widely cited set in ECG simulation literature)
SIGMA_I_LONG  = 0.300   # S/m   σiL  intracellular longitudinal (fiber)
SIGMA_I_TRANS = 0.030   # S/m   σiT  intracellular transverse
SIGMA_E_LONG  = 0.300   # S/m   σeL  extracellular longitudinal (fiber)
SIGMA_E_TRANS = 0.120   # S/m   σeT  extracellular transverse

# Convenience aliases used throughout the file
MILLI_VENTRICULAR_ENDO_IL = SIGMA_I_LONG
MILLI_VENTRICULAR_ENDO_IT = SIGMA_I_TRANS
MILLI_VENTRICULAR_EPI_EL  = SIGMA_E_LONG
MILLI_VENTRICULAR_EPI_ET  = SIGMA_E_TRANS

# Blood-pool conductivity in bidomain context (isotropic, high)
SIGMA_BLOOD = 0.667


def _device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


# ==============================================================================
#  Step 1 – Extract simulation-ready CARP meshes
# ==============================================================================

def step_extract(sim_dir: Path, force: bool = False) -> None:
    from torchcor.ecg.kcl_extract import extract, DEFAULT_CARP_PREFIX, DEFAULT_VTU
    extract(
        carp_prefix=DEFAULT_CARP_PREFIX,
        vtu_path=DEFAULT_VTU,
        out_dir=sim_dir,
        force=force,
    )


# ==============================================================================
#  Step 2 – Monodomain simulation on biventricular mesh
# ==============================================================================

def step_simulate(
    sim_dir: Path,
    result_dir: Path,
    device: torch.device,
    dtype: torch.dtype,
    T: float = 600.0,
    dt: float = 0.05,
    snapshot_interval: int = 1,
    force: bool = False,
) -> torch.Tensor:
    """
    Run monodomain simulation on the extracted biventricular mesh.

    Returns
    -------
    Vm : (n_snapshots, n_nodes) tensor  (or loaded from cache)
    """
    import torchcor as tc
    from torchcor.simulator import Monodomain
    from torchcor.ionic import TenTusscherPanfilov

    vm_cache = result_dir / "Vm.pt"
    if vm_cache.exists() and not force:
        print(f"Loading cached Vm from {vm_cache}")
        return torch.load(str(vm_cache), map_location=device)

    result_dir.mkdir(parents=True, exist_ok=True)
    vent_dir   = sim_dir / "ventricles"
    pacing_dir = vent_dir / "pacing"

    print("\n" + "=" * 60)
    print(f"[Monodomain] T={T} ms, dt={dt} ms, device={device}")
    print("=" * 60)

    tc.set_device(str(device))

    # ── Ionic models: three TenTusscher-Panfilov 2006 cell types ─────────
    # im_endo absorbs all regions at load_mesh time (region_ids=None).
    # im_mid / im_epi use non-existent tags so load_mesh gives them empty
    # node_indices; these are overridden after transmural depth is computed.
    im_endo = TenTusscherPanfilov(cell_type="ENDO",  dt=dt, dtype=dtype,
                                  region_ids=None)
    im_mid  = TenTusscherPanfilov(cell_type="MCELL", dt=dt, dtype=dtype,
                                  region_ids=[99997])
    im_epi  = TenTusscherPanfilov(cell_type="EPI",   dt=dt, dtype=dtype,
                                  region_ids=[99998])

    simulator = Monodomain(ionic_models=[im_endo, im_mid, im_epi],
                           T=T, dt=dt, dtype=dtype)
    simulator.load_mesh(path=str(vent_dir), unit_conversion=1000)  # µm → mm

    # ── Transmural layer assignment ───────────────────────────────────────
    # Compute EPI/MID/ENDO depth using:
    #   (a) COM-ray method  – radial distance from blood-pool centre-of-mass
    #   (b) Euclidean distance from endo/epi boundary surfaces
    # Result is cached to disk so re-runs are instant.
    print("\n[Layer Assignment] Computing transmural EPI/MID/ENDO depth...",
          flush=True)
    from torchcor.ecg.kcl_layers import assign_wall_layers

    nodes_np = simulator.nodes.cpu().numpy()
    elems_np = simulator.elems.Tt.data.cpu().numpy()
    regs_np  = simulator.regions.cpu().numpy()

    endo_idx, mid_idx, epi_idx, sep_node_idx = assign_wall_layers(
        nodes_np, elems_np, regs_np,
        cache_dir=vent_dir,   # cache depth + layer arrays here
    )

    # Blood-pool nodes (tags 24/25) — assign to ENDO model to preserve
    # the original single-model behaviour for passive fluid regions.
    # sep_node_idx is already correctly distributed across endo_idx/mid_idx
    # by assign_wall_layers (LV-face and RV-face → ENDO, mid-septum → MCELL).
    bp_mask    = np.isin(regs_np, [24, 25])
    bp_nodes   = np.unique(elems_np[bp_mask].reshape(-1))
    endo_with_bp = torch.tensor(
        np.unique(np.concatenate([endo_idx, bp_nodes])),
        device=device, dtype=torch.long,
    )

    # Override node indices BEFORE solve() is called.
    im_endo.node_indices = endo_with_bp
    im_mid.node_indices  = torch.tensor(mid_idx, device=device, dtype=torch.long)
    im_epi.node_indices  = torch.tensor(epi_idx, device=device, dtype=torch.long)

    print(
        f"  Node indices set — ENDO+BP: {len(endo_with_bp):,}  "
        f"MID: {len(mid_idx):,}  EPI: {len(epi_idx):,}  "
        f"SEPTUM (for info): {len(sep_node_idx):,}",
        flush=True,
    )

    # Always use all four conductivity parameters (il, it, el, et).
    # Calling add_conductivity with only il/it sets sigma_il=None internally,
    # which breaks any subsequent bidomain call. Use el=et=il=it for isotropic.
    #
    # Tags 34, 35 = ventricular walls (bidomain — must come first)
    simulator.add_conductivity(
        [34, 35],
        il=MILLI_VENTRICULAR_ENDO_IL,
        it=MILLI_VENTRICULAR_ENDO_IT,
        el=MILLI_VENTRICULAR_EPI_EL,
        et=MILLI_VENTRICULAR_EPI_ET,
    )
    # Tags 24, 25 = blood pools (isotropic — use 4-param form to avoid nullifying sigma_il)
    simulator.add_conductivity(
        [24, 25],
        il=SIGMA_BLOOD, it=SIGMA_BLOOD,
        el=SIGMA_BLOOD, et=SIGMA_BLOOD,
    )

    # Pacing: LV endocardium at t=0, RV endocardium at t=5 ms
    lv_vtx = pacing_dir / "LV_endo.vtx"
    rv_vtx = pacing_dir / "RV_endo.vtx"

    if lv_vtx.exists():
        simulator.add_stimulus(str(lv_vtx), start=0.0, duration=2.0, intensity=80.0)
        print(f"  LV pacing: {lv_vtx.name}")
    else:
        print(f"  WARNING: {lv_vtx} not found — no LV stimulus applied")

    if rv_vtx.exists():
        simulator.add_stimulus(str(rv_vtx), start=5.0, duration=2.0, intensity=80.0)
        print(f"  RV pacing: {rv_vtx.name}")
    else:
        print(f"  WARNING: {rv_vtx} not found — no RV stimulus applied")

    t0 = time.time()
    Vm = simulator.solve(
        a_tol=1e-5,
        r_tol=1e-5,
        max_iter=200,
        snapshot_interval=snapshot_interval,
        verbose=True,
        result_path=str(result_dir / "monodomain"),
    )
    dt_wall = time.time() - t0
    print(f"\nSimulation done in {dt_wall:.1f} s  ({dt_wall/60:.1f} min)")

    # Compute activation / repolarisation maps
    ATs = simulator.compute_activation_map(
        Vm=Vm, snapshot_interval=snapshot_interval, threshold=0)
    RTs = simulator.compute_repolarization_map(
        Vm=Vm, snapshot_interval=snapshot_interval, threshold=-70)
    print(f"  AT  range: {ATs.min().item():.1f} – {ATs.max().item():.1f} ms")
    print(f"  RT  range: {RTs.min().item():.1f} – {RTs.max().item():.1f} ms")

    # Save AT / RT maps for visualisation
    np.save(str(result_dir / "activation_times.npy"),    ATs.cpu().numpy())
    np.save(str(result_dir / "repolarization_times.npy"), RTs.cpu().numpy())
    print(f"  Saved AT/RT maps → {result_dir}")

    # Save Vm snapshot series
    torch.save(Vm, str(vm_cache))
    print(f"  Saved Vm ({Vm.shape}) → {vm_cache}")

    # Write VTK visualisation files (every 10 snapshots)
    try:
        simulator.vm_to_vtk(Vm=Vm, step=10)
    except Exception as e:
        print(f"  (VTK export skipped: {e})")

    return Vm


# ==============================================================================
#  Step 3 – ECG forward solve  (lead field method)
# ==============================================================================

def step_ecg(
    sim_dir: Path,
    result_dir: Path,
    Vm: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
    dt_ms: float = 1.0,   # snapshot spacing in ms
    force: bool = False,
) -> dict[str, torch.Tensor]:
    """
    Compute 12-lead ECG using the reciprocal lead field method.

    Parameters
    ----------
    Vm     : (n_time, n_heart_nodes) — transmembrane potential in mV
    dt_ms  : time between snapshots (for axis labelling)

    Returns
    -------
    ecg : dict { 'I', 'II', …, 'V6' : (n_time,) tensor }
    """
    from torchcor.ecg.test import ECGSolver

    ecg_cache = result_dir / "ecg_12lead.pt"
    if ecg_cache.exists() and not force:
        print(f"Loading cached ECG from {ecg_cache}")
        return torch.load(str(ecg_cache), map_location=device)

    torso_dir = sim_dir / "torso"
    vent_dir  = sim_dir / "ventricles"
    elec_vtx  = sim_dir / "electrodes.vtx"
    h2t_file  = sim_dir / "heart_to_torso.npy"

    for f in (torso_dir, vent_dir, elec_vtx, h2t_file):
        if not Path(str(f)).exists():
            raise FileNotFoundError(
                f"Required file/dir missing: {f}\n"
                "Run step_extract first (or kcl_extract.py)."
            )

    heart_to_torso = np.load(str(h2t_file))
    print(f"\n{'='*60}")
    print(f"[ECG Forward]  torso nodes: (loading…)  heart nodes: {len(heart_to_torso)}")
    print("=" * 60)

    solver = ECGSolver(
        torso_mesh_dir=str(torso_dir),
        heart_mesh_dir=str(vent_dir),
        heart_to_torso_node=heart_to_torso,
        device=device,
        dtype=dtype,
    )

    # ── Torso conductivities (isotropic, S/m) ─────────────────────────
    for tag, g in CONDUCTIVITY.items():
        solver.set_torso_conductivity([tag], g=float(g))

    # ── Heart conductivities (bidomain, S/m) ──────────────────────────
    # Blood pools: high isotropic (il=el=sigma_blood, it=et=sigma_blood)
    solver.set_heart_conductivity(
        [24, 25],
        il=SIGMA_BLOOD, it=SIGMA_BLOOD,
        el=SIGMA_BLOOD, et=SIGMA_BLOOD,
    )
    # Ventricular walls: anisotropic bidomain
    solver.set_heart_conductivity(
        [34, 35],
        il=MILLI_VENTRICULAR_ENDO_IL, it=MILLI_VENTRICULAR_ENDO_IT,
        el=MILLI_VENTRICULAR_EPI_EL,  et=MILLI_VENTRICULAR_EPI_ET,
    )

    # ── Assemble stiffness matrices ────────────────────────────────────
    print("\nAssembling FEM stiffness matrices…")
    t0 = time.time()
    solver.build()
    print(f"  Assembly time: {time.time()-t0:.1f} s")

    # ── Load electrode node indices ────────────────────────────────────
    _load_electrodes_kcl(solver, elec_vtx)

    # ── Precompute lead fields (one CG solve per electrode) ────────────
    print("\nPrecomputing lead fields…")
    t0 = time.time()
    solver.precompute_lead_fields(I=1.0, a_tol=1e-8, r_tol=1e-8, max_iter=5000)
    print(f"  Lead field time: {time.time()-t0:.1f} s")

    # ── Compute 12-lead ECG ────────────────────────────────────────────
    print("\nComputing 12-lead ECG…")
    Vm_dev = Vm.to(device=device, dtype=dtype)
    ecg = solver.compute_12lead(Vm_dev)

    # Save
    torch.save({k: v.cpu() for k, v in ecg.items()}, str(ecg_cache))
    print(f"  Saved ECG → {ecg_cache}")

    # Plot (PNG + CSV)
    _plot_ecg_12lead(ecg, dt_ms=dt_ms,
                     filepath=result_dir / "ecg_12lead.png")
    _save_ecg_csv(ecg, dt_ms=dt_ms,
                  filepath=result_dir / "ecg_12lead.csv")

    # Interactive HTML (Plotly)
    try:
        from torchcor.ecg.kcl_visualize import _save_ecg_plotly
        n_t   = next(iter(ecg.values())).shape[0]
        t_ms  = np.arange(n_t, dtype=np.float32) * dt_ms
        ecg_np = {k: v.cpu().numpy() for k, v in ecg.items()}
        _save_ecg_plotly(ecg_np, t_ms, result_dir)
    except Exception as e:
        print(f"  [warn] ECG HTML not saved: {e}")

    return ecg


def _load_electrodes_kcl(solver, elec_vtx: Path) -> None:
    """
    Parse the KCL-format electrodes.vtx file (written by kcl_extract.py)
    and populate solver.electrodes.

    File format:
        10          ← number of electrodes
        1234  # V1
        5678  # V2
        ...
    """
    LABELS = ["V1", "V2", "V3", "V4", "V5", "V6", "RA", "LA", "RL", "LL"]
    with open(elec_vtx) as f:
        n = int(f.readline().strip())
        node_ids = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            node_id = int(line.split()[0])
            node_ids.append(node_id)

    if len(node_ids) != len(LABELS):
        raise ValueError(
            f"electrodes.vtx has {len(node_ids)} entries; expected {len(LABELS)}"
        )
    solver.electrodes = dict(zip(LABELS, node_ids))
    print(f"  Loaded {len(solver.electrodes)} electrodes, ground = {solver.ground}")


# ==============================================================================
#  Step 4 – Body Surface Potential Map (BSPM)  —  full forward solve
# ==============================================================================

def step_bspm(
    sim_dir: Path,
    result_dir: Path,
    device: torch.device,
    dtype: torch.dtype,
    n_frames: int = None,
    force: bool = False,
) -> None:
    """
    Compute the body surface potential map (BSPM) for a set of time frames.

    For each frame t, solves the full forward bioelectric problem on the torso:

        K_torso  phi_e(t)  =  -K_i_torso  Vm(t)

    and extracts phi_e at all torso surface nodes.

    n_frames = None (default) → one frame per ms (all snapshots in Vm.pt).

    Outputs (written to result_dir)
    --------------------------------
    bspm.pt                : float32 tensor [N_frames × N_surface]
    bspm_frame_ms.npy      : float32 array  [N_frames]   frame times in ms
    torso_surface_nodes.npy: int64 array    [N_surface]  torso node indices
    """
    from torchcor.ecg.test import ECGSolver, solve_singular_system
    import pyvista as pv

    bspm_cache = result_dir / "bspm.pt"
    if bspm_cache.exists() and not force:
        print(f"BSPM cache found: {bspm_cache}  (use --force-bspm to recompute)")
        return

    torso_dir  = sim_dir / "torso"
    vent_dir   = sim_dir / "ventricles"
    elec_vtx   = sim_dir / "electrodes.vtx"
    h2t_file   = sim_dir / "heart_to_torso.npy"
    vm_path    = result_dir / "Vm.pt"

    for f in (torso_dir, vent_dir, elec_vtx, h2t_file, vm_path):
        if not Path(str(f)).exists():
            raise FileNotFoundError(
                f"Required file/dir missing: {f}\n"
                "Run step_extract + step_simulate first."
            )

    result_dir.mkdir(parents=True, exist_ok=True)

    # ── Load Vm and select frames ──────────────────────────────────────
    print("\nLoading Vm(t) for frame selection…", flush=True)
    Vm_all  = torch.load(str(vm_path), map_location="cpu")
    T_total = Vm_all.shape[0]

    if n_frames is None or n_frames >= T_total:
        frame_idx = np.arange(T_total, dtype=int)
        print(f"  Using all {T_total} frames (1 ms/frame)", flush=True)
    else:
        qrs_end   = min(50, T_total)
        n_qrs     = min(30, qrs_end)
        n_tail    = n_frames - n_qrs
        t_qrs     = np.linspace(0, qrs_end - 1, n_qrs,  dtype=int)
        t_tail    = np.linspace(qrs_end, T_total - 1, n_tail, dtype=int)
        frame_idx = np.unique(np.concatenate([t_qrs, t_tail]))[:n_frames]
        print(f"  Subsampled to {len(frame_idx)} frames (of {T_total})", flush=True)
    frame_ms  = frame_idx.astype(np.float32)
    N_frames  = len(frame_idx)

    # ── Build ECGSolver  (same setup as step_ecg) ─────────────────────
    heart_to_torso = np.load(str(h2t_file))
    print(f"\n[BSPM Forward]  heart nodes: {len(heart_to_torso)}", flush=True)

    # Use float64 for CG accuracy
    solver_dtype = torch.float64

    solver = ECGSolver(
        torso_mesh_dir=str(torso_dir),
        heart_mesh_dir=str(vent_dir),
        heart_to_torso_node=heart_to_torso,
        device=device,
        dtype=solver_dtype,
    )

    for tag, g in CONDUCTIVITY.items():
        solver.set_torso_conductivity([tag], g=float(g))
    solver.set_heart_conductivity(
        [24, 25],
        il=SIGMA_BLOOD, it=SIGMA_BLOOD,
        el=SIGMA_BLOOD, et=SIGMA_BLOOD,
    )
    solver.set_heart_conductivity(
        [34, 35],
        il=MILLI_VENTRICULAR_ENDO_IL, it=MILLI_VENTRICULAR_ENDO_IT,
        el=MILLI_VENTRICULAR_EPI_EL,  et=MILLI_VENTRICULAR_EPI_ET,
    )

    print("Assembling FEM stiffness matrices…", flush=True)
    t0 = time.time()
    solver.build()
    print(f"  Assembly: {time.time()-t0:.1f} s", flush=True)

    _load_electrodes_kcl(solver, elec_vtx)
    ground_idx = solver.electrodes[solver.ground]

    # ── Extract torso surface node indices via pyvista ─────────────────
    print("Extracting torso surface…", flush=True)
    import pandas as pd
    pts_file  = sorted(torso_dir.glob("*.pts"))[0]
    elem_file = sorted(torso_dir.glob("*.elem"))[0]

    nodes_mm = pd.read_csv(
        pts_file, sep=r"\s+", header=None, skiprows=1,
        usecols=[0, 1, 2], engine="c", dtype=np.float32,
    ).values / 1000.0   # µm → mm

    raw_elems = pd.read_csv(
        elem_file, sep=r"\s+", header=None, skiprows=1,
        usecols=[1, 2, 3, 4], engine="c", dtype=np.int64,
    ).values

    cells    = np.hstack([np.full((len(raw_elems), 1), 4, dtype=np.int64), raw_elems])
    celltypes= np.full(len(raw_elems), 10, dtype=np.uint8)   # VTK_TETRA = 10
    grid     = pv.UnstructuredGrid(cells.ravel(), celltypes, nodes_mm.astype(np.float64))
    surface  = grid.extract_surface().triangulate()
    orig_ids = surface.point_data["vtkOriginalPointIds"].copy()   # [N_surface]
    N_surface = len(orig_ids)
    print(f"  Surface nodes: {N_surface}", flush=True)

    # ── Forward solve per frame ────────────────────────────────────────
    print(f"\nRunning {N_frames} forward solves…", flush=True)
    bspm = np.zeros((N_frames, N_surface), dtype=np.float32)

    for i, t_idx in enumerate(frame_idx):
        t0 = time.time()
        Vm_t = Vm_all[int(t_idx)].to(device=device, dtype=solver_dtype)
        Vm_torso = solver._embed_Vm_in_torso(Vm_t)
        rhs      = -(solver.K_i_torso @ Vm_torso)
        phi_e    = solve_singular_system(
            solver.K_torso, rhs,
            ground_node=ground_idx,
            device=device, dtype=solver_dtype,
            a_tol=1e-8, r_tol=1e-8, max_iter=5000,
        )
        bspm[i]  = phi_e[orig_ids].cpu().float().numpy()
        elapsed  = time.time() - t0
        print(f"  [{i+1:3d}/{N_frames}]  t={frame_ms[i]:.0f} ms  "
              f"phi range [{bspm[i].min():.3f}, {bspm[i].max():.3f}] mV  "
              f"({elapsed:.1f} s)", flush=True)

    # ── Save ───────────────────────────────────────────────────────────
    torch.save(torch.from_numpy(bspm), str(bspm_cache))
    np.save(str(result_dir / "bspm_frame_ms.npy"),      frame_ms)
    np.save(str(result_dir / "torso_surface_nodes.npy"), orig_ids)

    abs_max = float(np.percentile(np.abs(bspm), 99.5))
    print(f"\nBSPM saved → {bspm_cache}")
    print(f"  Shape: {bspm.shape}  |  99.5th-percentile |phi| = {abs_max:.3f} mV")


# ==============================================================================
#  Plotting
# ==============================================================================

def _plot_ecg_12lead(
    ecg: dict[str, torch.Tensor],
    dt_ms: float,
    filepath: Path,
) -> None:
    """Plot 12-lead ECG in standard 3×4 clinical layout."""
    import matplotlib.pyplot as plt

    lead_order = [
        "I",   "aVR", "V1", "V4",
        "II",  "aVL", "V2", "V5",
        "III", "aVF", "V3", "V6",
    ]
    fig, axes = plt.subplots(3, 4, figsize=(16, 8), sharex=True)
    fig.patch.set_facecolor("#f8f8f0")

    for ax, lead in zip(axes.flatten(), lead_order):
        if lead not in ecg:
            ax.set_visible(False)
            continue
        sig = ecg[lead].detach().cpu().float().numpy()
        t   = np.arange(len(sig)) * dt_ms
        ax.plot(t, sig, "k-", lw=0.9)
        ax.axhline(0, color="#aaaaaa", lw=0.4, ls="--")
        ax.set_title(lead, fontsize=11, fontweight="bold")
        ax.set_ylabel("mV", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_facecolor("#f8f8f0")

    for ax in axes[-1]:
        ax.set_xlabel("ms", fontsize=8)

    fig.suptitle("12-Lead ECG  —  KCL torso model  (TenTusscher-Panfilov 2006)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    filepath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(filepath), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved ECG plot → {filepath}")


def _save_ecg_csv(
    ecg: dict[str, torch.Tensor],
    dt_ms: float,
    filepath: Path,
) -> None:
    """Save all ECG leads to a CSV file."""
    import pandas as pd

    n_time = next(iter(ecg.values())).shape[0]
    t = np.arange(n_time) * dt_ms
    df = pd.DataFrame({"time_ms": t})
    for lead, sig in sorted(ecg.items()):
        df[lead] = sig.detach().cpu().float().numpy()
    df.to_csv(str(filepath), index=False)
    print(f"  Saved ECG CSV → {filepath}")


# ==============================================================================
#  Master runner
# ==============================================================================

def run(
    sim_dir: Path = DEFAULT_SIM_DIR,
    result_dir: Path = DEFAULT_RESULT,
    device_str: str = "auto",
    T: float = 600.0,
    dt: float = 0.05,
    snapshot_interval: int = 1,
    dtype_str: str = "float32",
    skip_extract: bool = False,
    skip_sim: bool = False,
    skip_ecg: bool = False,
    run_bspm: bool = False,
    bspm_frames: int | None = None,
    force_extract: bool = False,
    force_sim: bool = False,
    force_ecg: bool = False,
    force_bspm: bool = False,
) -> None:
    device = _device(device_str)
    dtype  = torch.float32 if dtype_str == "float32" else torch.float64

    print(f"\nKCL Pipeline — device={device}, dtype={dtype}")
    print(f"  sim_dir    : {sim_dir}")
    print(f"  result_dir : {result_dir}")

    # ── Step 1: Extract ─────────────────────────────────────────────────
    if not skip_extract:
        step_extract(sim_dir, force=force_extract)

    # ── Step 2: Monodomain simulation ───────────────────────────────────
    if not skip_sim:
        Vm = step_simulate(
            sim_dir=sim_dir,
            result_dir=result_dir,
            device=device,
            dtype=dtype,
            T=T,
            dt=dt,
            snapshot_interval=snapshot_interval,
            force=force_sim,
        )
    else:
        vm_cache = result_dir / "Vm.pt"
        if not vm_cache.exists():
            raise FileNotFoundError(
                f"--skip-sim specified but no cached Vm found at {vm_cache}"
            )
        print(f"Loading Vm from cache: {vm_cache}")
        Vm = torch.load(str(vm_cache), map_location=device)

    print(f"\nVm shape: {Vm.shape}  (snapshots × heart nodes)")

    # ── Step 3: ECG forward solve ────────────────────────────────────────
    if not skip_ecg:
        # snapshot_interval * dt gives time step in ms between snapshots
        dt_snapshot_ms = snapshot_interval * dt
        ecg = step_ecg(
            sim_dir=sim_dir,
            result_dir=result_dir,
            Vm=Vm,
            device=device,
            dtype=dtype,
            dt_ms=dt_snapshot_ms,
            force=force_ecg,
        )
        print(f"\nECG leads: {sorted(ecg.keys())}")
        for lead, sig in sorted(ecg.items()):
            v = sig.cpu().float()
            print(f"  {lead:>4s}: min={v.min():.3f} mV  max={v.max():.3f} mV")

    # ── Step 4: BSPM (optional, compute-heavy) ───────────────────────────
    if run_bspm:
        step_bspm(
            sim_dir=sim_dir,
            result_dir=result_dir,
            device=device,
            dtype=dtype,
            n_frames=bspm_frames,
            force=force_bspm,
        )

    print("\nPipeline complete.")


# ==============================================================================
#  CLI
# ==============================================================================

def main() -> None:
    # ── Launch GUI when called with no arguments ───────────────────────────
    import sys as _sys
    if len(_sys.argv) == 1:
        try:
            from torchcor.ecg.kcl_gui import launch
            launch()
            return
        except Exception as _e:
            print(f"[warn] GUI unavailable ({_e}), running CLI mode.\n")

    p = argparse.ArgumentParser(
        description="KCL whole-torso cardiac simulation pipeline"
    )
    p.add_argument("--sim-dir",    default=str(DEFAULT_SIM_DIR))
    p.add_argument("--result-dir", default=str(DEFAULT_RESULT))
    p.add_argument("--device",     default="auto",
                   help="e.g. cuda:0, cpu, auto")
    p.add_argument("--dtype",      default="float32",
                   choices=["float32", "float64"])
    p.add_argument("--T",          type=float, default=600.0,
                   help="Simulation duration (ms)")
    p.add_argument("--dt",         type=float, default=0.05,
                   help="Time step (ms)")
    p.add_argument("--snapshot-interval", type=int, default=1,
                   help="Save Vm every N time steps")
    p.add_argument("--skip-extract", action="store_true")
    p.add_argument("--skip-sim",     action="store_true")
    p.add_argument("--skip-ecg",     action="store_true")
    p.add_argument("--bspm",         action="store_true",
                   help="Also run full-forward BSPM solve (slow: ~1 CG solve / frame)")
    p.add_argument("--bspm-frames",  type=int, default=None,
                   help="Number of time frames for BSPM (default: all)")
    p.add_argument("--force-extract", action="store_true")
    p.add_argument("--force-sim",     action="store_true")
    p.add_argument("--force-ecg",     action="store_true")
    p.add_argument("--force-bspm",    action="store_true")
    args = p.parse_args()

    run(
        sim_dir=Path(args.sim_dir),
        result_dir=Path(args.result_dir),
        device_str=args.device,
        T=args.T,
        dt=args.dt,
        snapshot_interval=args.snapshot_interval,
        dtype_str=args.dtype,
        skip_extract=args.skip_extract,
        skip_sim=args.skip_sim,
        skip_ecg=args.skip_ecg,
        run_bspm=args.bspm,
        bspm_frames=args.bspm_frames,
        force_extract=args.force_extract,
        force_sim=args.force_sim,
        force_ecg=args.force_ecg,
        force_bspm=args.force_bspm,
    )


if __name__ == "__main__":
    main()
