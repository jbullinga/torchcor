"""
torchcor/ecg/kcl_layers.py
==========================
Transmural EPI / MID / ENDO / SEPTUM layer assignment for KCL ventricular
wall nodes.

Free-wall transmural depth (ENDO → EPI)
----------------------------------------
Uses a hybrid of two geometrically complementary methods:

  1. COM-ray method
     ---------------
     The centre-of-mass (COM) of the ventricular blood pool defines a natural
     "inside" reference point.  For each wall node v the radial distance from
     the COM is compared with the expected endocardial and epicardial radii in
     the same angular direction (estimated by k-NN on the unit sphere):

         depth_ray = (r_v − r_endo_est) / (r_epi_est − r_endo_est)

     Because rays from the COM pass sequentially through ENDO → MID → EPI this
     naturally handles the curved, non-convex geometry of both ventricles.

  2. Euclidean distance method
     --------------------------
     For each wall node v:

         depth_dist = d_endo / (d_endo + d_epi)

     where d_endo / d_epi are the min distances to the endocardial and
     epicardial boundary surfaces respectively.

  Combined depth = 0.5 * (depth_ray + depth_dist), clipped to [0, 1].

Interventricular septum detection
------------------------------------
The septum is identified using the LV COM-ray method:

  A wall node is classified as SEPTAL when a ray from the LV blood-pool COM
  through that node continues into RV blood-pool territory — i.e. RV blood-pool
  nodes exist in approximately the same angular direction at a *larger* radial
  distance from the LV COM than the wall node itself.

  Once detected, septal transmural depth is computed *across the septum*:

      sep_depth = d_LV_face / (d_LV_face + d_RV_face)

  where d_LV_face / d_RV_face are Euclidean distances to the LV-facing and
  RV-facing surfaces of the septal wall.

  Septal cell-type assignment (TenTusscher-Panfilov convention):
    sep_depth < 1/3        → ENDO  (LV endocardial face of septum)
    1/3 ≤ sep_depth < 2/3  → MCELL (mid-septum)
    sep_depth ≥ 2/3        → ENDO  (RV endocardial face — also contacts blood)

  This means both faces of the septum are ENDO, which is physiologically
  correct; the outer EPI layer is reserved for the free wall only.

Cell-type thresholds for free-wall nodes (TenTusscher-Panfilov 2006):
  ENDO  : depth  < 1/3
  MCELL : 1/3 ≤ depth < 2/3
  EPI   : depth ≥ 2/3

The depth arrays and layer assignments are cached to disk so the expensive
KDTree queries only run once per mesh.
"""

from __future__ import annotations

import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# CARP region tags
# ---------------------------------------------------------------------------
LV_BLOOD_TAG = 24
RV_BLOOD_TAG = 25
BLOOD_TAGS   = {LV_BLOOD_TAG, RV_BLOOD_TAG}
WALL_TAGS    = {34, 35}

# Cache filenames (written alongside the ventricles mesh)
_DEPTH_FILE   = "transmural_depth.npy"    # float32 (N_nodes,)  NaN for non-wall
_LAYERS_FILE  = "layer_assignment.npz"    # endo/mid/epi/sep node-index arrays
_SEPTUM_FILE  = "septum_depth.npy"        # float32 (N_nodes,)  NaN for non-septal


# =============================================================================
#  Septal detection
# =============================================================================

def detect_septum(
    nodes_mm:    np.ndarray,   # (N, 3)  all mesh nodes in mm
    elem_conn:   np.ndarray,   # (M, 4)  tetrahedral connectivity
    elem_regs:   np.ndarray,   # (M,)    region tags
    k_angular:   int   = 8,    # angular k-NN neighbours on unit sphere
    cos_thresh:  float = 0.70, # min mean cosine similarity (≈ within 45°)
) -> tuple[np.ndarray, np.ndarray]:
    """
    Identify interventricular septal wall nodes using the LV COM-ray method.

    A wall node is SEPTAL when:
      (a) RV blood-pool nodes exist in approximately the same angular direction
          from the LV COM (mean cosine similarity > cos_thresh), AND
      (b) Those RV blood-pool nodes are at a *larger* radial distance from the
          LV COM than the wall node itself (the ray continues into RV territory).

    Septal transmural depth is 0 at the LV-facing surface and 1 at the
    RV-facing surface of the septal wall.

    Parameters
    ----------
    cos_thresh : angular locality gate.  0.70 ≈ within 45°; 0.85 ≈ within 32°.
                 The septum spans ~60° of the LV circumference, so 0.70 is the
                 recommended default.

    Returns
    -------
    sep_node_idx : (N_sep,) int   global wall-node indices of septal nodes
    sep_depth    : (N_sep,) float32  transmural depth 0=LV-face, 1=RV-face
    """
    from scipy.spatial import cKDTree

    # ── LV blood-pool COM ─────────────────────────────────────────────────
    lv_mask     = elem_regs == LV_BLOOD_TAG
    lv_node_idx = np.unique(elem_conn[lv_mask].reshape(-1))
    lv_com      = nodes_mm[lv_node_idx].mean(axis=0)
    print(
        f"  LV COM (mm): [{lv_com[0]:.1f}, {lv_com[1]:.1f}, {lv_com[2]:.1f}]",
        flush=True,
    )

    # ── RV blood-pool nodes ───────────────────────────────────────────────
    rv_mask     = elem_regs == RV_BLOOD_TAG
    rv_node_idx = np.unique(elem_conn[rv_mask].reshape(-1))
    rv_coords   = nodes_mm[rv_node_idx]

    # ── Wall nodes ────────────────────────────────────────────────────────
    wall_mask_el  = np.isin(elem_regs, list(WALL_TAGS))
    wall_node_idx = np.unique(elem_conn[wall_mask_el].reshape(-1))
    wall_coords   = nodes_mm[wall_node_idx]
    N_wall        = len(wall_node_idx)

    # ── Directions and radii from LV COM ─────────────────────────────────
    def _dirs_radii(pts: np.ndarray):
        v = pts - lv_com
        r = np.linalg.norm(v, axis=1)
        return v / np.maximum(r, 1e-10)[:, None], r   # (N,3), (N,)

    wall_dir, wall_r = _dirs_radii(wall_coords)
    rv_dir,   rv_r   = _dirs_radii(rv_coords)

    # ── Angular k-NN of RV blood-pool nodes for every wall node ──────────
    k_eff = min(k_angular, len(rv_node_idx))
    tree_rv_ang  = cKDTree(rv_dir)
    _, rv_nn     = tree_rv_ang.query(wall_dir, k=k_eff, workers=-1)

    # Handle k=1 case (rv_nn could be 1-D)
    if rv_nn.ndim == 1:
        rv_nn = rv_nn[:, None]

    # Mean cosine similarity: dot product of unit vectors
    rv_nn_dirs = rv_dir[rv_nn]                           # (N_wall, k, 3)
    cos_sim    = (wall_dir[:, None, :] * rv_nn_dirs).sum(axis=2)  # (N_wall, k)
    mean_cos   = cos_sim.mean(axis=1)                   # (N_wall,)

    # Mean radius of k-NN RV nodes in the same angular direction
    r_rv_est   = rv_r[rv_nn].mean(axis=1)               # (N_wall,)

    # ── Septal condition ──────────────────────────────────────────────────
    # Ray from LV COM through wall node continues into RV territory:
    #   (a) RV blood is in approximately the same direction (mean_cos > thresh)
    #   (b) RV blood is further from LV COM than this wall node (r_rv_est > wall_r)
    is_septal    = (mean_cos > cos_thresh) & (r_rv_est > wall_r)
    sep_local    = np.where(is_septal)[0]               # indices into wall_node_idx
    sep_node_idx = wall_node_idx[sep_local]             # global node indices

    print(
        f"  Septal nodes detected: {len(sep_node_idx):,} / {N_wall:,} wall nodes "
        f"({100 * len(sep_node_idx) / max(N_wall, 1):.1f}%)  "
        f"[cos_thresh={cos_thresh:.2f}]",
        flush=True,
    )

    # ── Septal depth: LV-face → RV-face ──────────────────────────────────
    sep_coords = nodes_mm[sep_node_idx]

    lv_set = set(lv_node_idx.tolist())
    rv_set = set(rv_node_idx.tolist())

    # LV-facing septal surface: septal wall nodes shared with LV blood pool
    lv_sep_idx = np.array([i for i in sep_node_idx if i in lv_set], dtype=int)
    # RV-facing septal surface: septal wall nodes shared with RV blood pool
    rv_sep_idx = np.array([i for i in sep_node_idx if i in rv_set], dtype=int)

    # Fallback: if no shared nodes found, use the nearest blood-pool nodes
    if len(lv_sep_idx) == 0:
        print(
            "  [Septum] No wall nodes shared with LV blood pool — "
            "falling back to LV blood-pool nodes for LV-face reference.",
            flush=True,
        )
        lv_sep_idx = lv_node_idx
    if len(rv_sep_idx) == 0:
        print(
            "  [Septum] No wall nodes shared with RV blood pool — "
            "falling back to RV blood-pool nodes for RV-face reference.",
            flush=True,
        )
        rv_sep_idx = rv_node_idx

    tree_lv_sep = cKDTree(nodes_mm[lv_sep_idx])
    tree_rv_sep = cKDTree(nodes_mm[rv_sep_idx])
    d_lv_face, _ = tree_lv_sep.query(sep_coords, workers=-1)
    d_rv_face, _ = tree_rv_sep.query(sep_coords, workers=-1)

    sep_depth = (
        d_lv_face / np.maximum(d_lv_face + d_rv_face, 1e-6)
    ).astype(np.float32)

    print(
        f"  Septal depth — mean: {sep_depth.mean():.3f}  "
        f"std: {sep_depth.std():.3f}  "
        f"LV-face nodes: {len(lv_sep_idx):,}  "
        f"RV-face nodes: {len(rv_sep_idx):,}",
        flush=True,
    )

    return sep_node_idx, sep_depth


# =============================================================================
#  Free-wall transmural depth
# =============================================================================

def compute_transmural_depth(
    nodes_mm:  np.ndarray,   # (N, 3)  all ventricular mesh nodes in mm
    elem_conn: np.ndarray,   # (M, 4)  tetrahedral connectivity
    elem_regs: np.ndarray,   # (M,)    region tags
    k_angular: int = 8,      # angular neighbours on unit sphere
) -> np.ndarray:
    """
    Compute normalised transmural depth for every node in the ventricular mesh.

    Returns
    -------
    depth : (N,) float32
        0.0 = endocardium, 1.0 = epicardium, NaN = non-wall nodes.

    Note: septal nodes are included here with depth relative to the blood-pool
    COM; they will be re-classified separately by assign_wall_layers().
    """
    from scipy.spatial import cKDTree
    import pyvista as pv

    # ── 1. Blood-pool nodes and combined centre of mass ───────────────────
    bp_mask     = np.isin(elem_regs, list(BLOOD_TAGS))
    bp_node_idx = np.unique(elem_conn[bp_mask].reshape(-1))
    bp_coords   = nodes_mm[bp_node_idx]
    com         = bp_coords.mean(axis=0)                   # (3,)
    print(
        f"  Blood-pool COM (mm): [{com[0]:.1f}, {com[1]:.1f}, {com[2]:.1f}]",
        flush=True,
    )

    # ── 2. Wall nodes ─────────────────────────────────────────────────────
    wall_mask     = np.isin(elem_regs, list(WALL_TAGS))
    wall_node_idx = np.unique(elem_conn[wall_mask].reshape(-1))
    wall_coords   = nodes_mm[wall_node_idx]                # (N_wall, 3)

    # ── 3. Endo boundary: nodes shared between blood-pool AND wall ────────
    bp_set        = set(bp_node_idx.tolist())
    wall_set      = set(wall_node_idx.tolist())
    endo_glob_idx = np.array(sorted(bp_set & wall_set), dtype=int)
    endo_coords   = nodes_mm[endo_glob_idx]                # (N_endo, 3)

    # ── 4. Epi boundary: outer surface of wall-only mesh ─────────────────
    n_wall    = wall_mask.sum()
    wall_conn = elem_conn[wall_mask]
    cells_vtk = np.column_stack(
        [np.full(n_wall, 4, dtype=np.int64), wall_conn.astype(np.int64)]
    ).ravel()
    cell_types = np.full(n_wall, 10, dtype=np.uint8)       # VTK_TETRA = 10
    grid_wall  = pv.UnstructuredGrid(cells_vtk, cell_types, nodes_mm)
    surf       = grid_wall.extract_surface()
    surf_ids   = surf.point_data.get("vtkOriginalPointIds", None)
    if surf_ids is None:
        surf_ids = np.arange(len(nodes_mm))
    # Epi = surface nodes that are NOT blood-pool nodes
    epi_glob_idx = np.array([i for i in surf_ids if i not in bp_set], dtype=int)
    epi_coords   = nodes_mm[epi_glob_idx]                  # (N_epi, 3)

    print(
        f"  Endo boundary: {len(endo_glob_idx):,} nodes  |  "
        f"Epi boundary: {len(epi_glob_idx):,} nodes  |  "
        f"Wall total: {len(wall_node_idx):,} nodes",
        flush=True,
    )

    # ── 5. Helper: direction unit vectors + radii from COM ────────────────
    def _from_com(pts):
        v = pts - com
        r = np.linalg.norm(v, axis=1)
        return v / np.maximum(r, 1e-10)[:, None], r        # dirs (N,3), radii (N,)

    wall_dir, wall_r = _from_com(wall_coords)
    endo_dir, endo_r = _from_com(endo_coords)
    epi_dir,  epi_r  = _from_com(epi_coords)

    # ── 6. COM-ray depth ──────────────────────────────────────────────────
    print(f"  Building angular KDTrees (k={k_angular})...", flush=True)
    tree_endo_ang = cKDTree(endo_dir)
    tree_epi_ang  = cKDTree(epi_dir)

    _, endo_nn = tree_endo_ang.query(wall_dir, k=k_angular, workers=-1)
    _, epi_nn  = tree_epi_ang.query(wall_dir,  k=k_angular, workers=-1)

    r_endo_est = endo_r[endo_nn].mean(axis=1)              # (N_wall,)
    r_epi_est  = epi_r[epi_nn].mean(axis=1)                # (N_wall,)
    denom_ray  = np.maximum(r_epi_est - r_endo_est, 1e-6)
    depth_ray  = np.clip((wall_r - r_endo_est) / denom_ray, 0.0, 1.0)

    # ── 7. Euclidean distance depth ───────────────────────────────────────
    print("  Computing Euclidean distances to endo/epi surfaces...", flush=True)
    tree_endo_3d = cKDTree(endo_coords)
    tree_epi_3d  = cKDTree(epi_coords)
    d_endo, _    = tree_endo_3d.query(wall_coords, workers=-1)
    d_epi,  _    = tree_epi_3d.query(wall_coords,  workers=-1)
    depth_dist   = np.clip(d_endo / np.maximum(d_endo + d_epi, 1e-6), 0.0, 1.0)

    # ── 8. Combine: equal weight on both methods ──────────────────────────
    depth_wall = (0.5 * (depth_ray + depth_dist)).astype(np.float32)

    print(
        f"  Depth stats — mean: {depth_wall.mean():.3f}  "
        f"std: {depth_wall.std():.3f}  "
        f"min: {depth_wall.min():.3f}  max: {depth_wall.max():.3f}",
        flush=True,
    )

    # ── 9. Expand to full-mesh array (NaN for non-wall nodes) ─────────────
    depth_full = np.full(len(nodes_mm), np.nan, dtype=np.float32)
    depth_full[wall_node_idx] = depth_wall
    return depth_full


# =============================================================================
#  Layer assignment  (main entry point)
# =============================================================================

def assign_wall_layers(
    nodes_mm:   np.ndarray,
    elem_conn:  np.ndarray,
    elem_regs:  np.ndarray,
    thresholds: tuple[float, float] = (1 / 3, 2 / 3),
    k_angular:  int   = 8,
    cos_thresh: float = 0.70,
    cache_dir:  Path | None = None,
    force:      bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Partition wall node indices into ENDO / MCELL / EPI layers, with the
    interventricular septum detected and classified separately.

    Parameters
    ----------
    thresholds  : (endo_max, mid_max) depth thresholds.
                  Default (1/3, 2/3) gives equal-thickness thirds.
    cos_thresh  : angular locality gate for septal detection (0.70 recommended).
    cache_dir   : if provided, depth + layer indices are cached here.
    force       : recompute even if cache exists.

    Returns
    -------
    endo_idx : int array
        Global node indices assigned ENDO cell type.
        Includes LV-face septal nodes (depth < 1/3 across septum) AND
        RV-face septal nodes (depth ≥ 2/3 across septum).
    mid_idx  : int array
        Global node indices assigned MCELL cell type.
        Includes mid-septal nodes (1/3 ≤ septal depth < 2/3).
    epi_idx  : int array
        Global node indices assigned EPI cell type (free wall only).
    sep_node_idx : int array
        All septal wall nodes (for visualisation / analysis).
        These are already correctly merged into endo_idx / mid_idx above.

    Cell-type assignment summary
    ----------------------------
    Free wall:
      depth < 1/3      → ENDO
      1/3 ≤ depth < 2/3 → MCELL
      depth ≥ 2/3       → EPI

    Septum (depth measured LV-face → RV-face):
      sep_depth < 1/3      → ENDO  (LV endocardial face)
      1/3 ≤ sep_depth < 2/3 → MCELL (mid-septum)
      sep_depth ≥ 2/3       → ENDO  (RV endocardial face — also touches blood)
    """
    # ── Try to load from cache ─────────────────────────────────────────────
    if cache_dir is not None and not force:
        layers_path = Path(cache_dir) / _LAYERS_FILE
        if layers_path.exists():
            data     = np.load(layers_path)
            endo_idx = data["endo_idx"]
            mid_idx  = data["mid_idx"]
            epi_idx  = data["epi_idx"]
            if "sep_node_idx" in data:
                sep_node_idx = data["sep_node_idx"]
            else:
                # Pre-septal cache: invalidate and recompute
                print(
                    "  [Layer cache] Cache predates septal detection — "
                    "recomputing (use --force-layers to suppress this check).",
                    flush=True,
                )
                data.close()
                # Fall through to recompute below
                sep_node_idx = None  # sentinel
            if sep_node_idx is not None:
                n_total = len(endo_idx) + len(mid_idx) + len(epi_idx)
                print(
                    f"  [Layer cache] ENDO {len(endo_idx):,}  "
                    f"MID {len(mid_idx):,}  EPI {len(epi_idx):,}  "
                    f"SEPTUM {len(sep_node_idx):,}  "
                    f"(total wall nodes {n_total:,})",
                    flush=True,
                )
                return endo_idx, mid_idx, epi_idx, sep_node_idx

    # ── Detect septal nodes ────────────────────────────────────────────────
    print("  Detecting interventricular septum...", flush=True)
    sep_node_idx, sep_depth = detect_septum(
        nodes_mm, elem_conn, elem_regs,
        k_angular=k_angular, cos_thresh=cos_thresh,
    )
    sep_set = set(sep_node_idx.tolist())

    # ── Compute free-wall transmural depth ────────────────────────────────
    print("  Computing free-wall transmural depth...", flush=True)
    depth = compute_transmural_depth(
        nodes_mm, elem_conn, elem_regs, k_angular=k_angular
    )

    # ── Save depth to cache ────────────────────────────────────────────────
    if cache_dir is not None:
        np.save(Path(cache_dir) / _DEPTH_FILE, depth)
        print(f"  Depth array saved → {cache_dir}/{_DEPTH_FILE}", flush=True)
        # Also save per-node septal depth
        sep_depth_full = np.full(len(nodes_mm), np.nan, dtype=np.float32)
        sep_depth_full[sep_node_idx] = sep_depth
        np.save(Path(cache_dir) / _SEPTUM_FILE, sep_depth_full)
        print(f"  Septal depth saved → {cache_dir}/{_SEPTUM_FILE}", flush=True)

    # ── Partition wall nodes into free-wall layers ────────────────────────
    wall_mask_el  = np.isin(elem_regs, list(WALL_TAGS))
    wall_node_idx = np.unique(elem_conn[wall_mask_el].reshape(-1))
    # Exclude septal nodes from free-wall classification
    fw_node_idx   = np.array([i for i in wall_node_idx if i not in sep_set], dtype=int)
    depth_fw      = depth[fw_node_idx]

    t1, t2 = thresholds
    endo_fw = fw_node_idx[depth_fw < t1]
    mid_fw  = fw_node_idx[(depth_fw >= t1) & (depth_fw < t2)]
    epi_fw  = fw_node_idx[depth_fw >= t2]

    # ── Partition septal nodes using septal depth ─────────────────────────
    sep_endo_lv  = sep_node_idx[sep_depth < t1]           # LV-face → ENDO
    sep_mid      = sep_node_idx[(sep_depth >= t1) & (sep_depth < t2)]  # MCELL
    sep_endo_rv  = sep_node_idx[sep_depth >= t2]           # RV-face → ENDO

    # ── Merge ─────────────────────────────────────────────────────────────
    endo_idx = np.unique(np.concatenate([endo_fw, sep_endo_lv, sep_endo_rv]))
    mid_idx  = np.unique(np.concatenate([mid_fw,  sep_mid]))
    epi_idx  = np.sort(epi_fw)

    n_fw    = len(fw_node_idx)
    n_sep   = len(sep_node_idx)
    n_total = len(wall_node_idx)

    print(
        f"\n  ── Layer assignment summary ───────────────────────────────────\n"
        f"  Free wall ({n_fw:,} nodes, t1={t1:.2f}, t2={t2:.2f}):\n"
        f"    ENDO: {len(endo_fw):,} ({100*len(endo_fw)/max(n_fw,1):.1f}%)  "
        f"MID: {len(mid_fw):,} ({100*len(mid_fw)/max(n_fw,1):.1f}%)  "
        f"EPI: {len(epi_fw):,} ({100*len(epi_fw)/max(n_fw,1):.1f}%)\n"
        f"  Septum ({n_sep:,} nodes, {100*n_sep/max(n_total,1):.1f}% of wall):\n"
        f"    LV-ENDO: {len(sep_endo_lv):,}  MID: {len(sep_mid):,}  "
        f"RV-ENDO: {len(sep_endo_rv):,}\n"
        f"  Merged totals — ENDO: {len(endo_idx):,}  MID: {len(mid_idx):,}  "
        f"EPI: {len(epi_idx):,}\n"
        f"  ──────────────────────────────────────────────────────────────",
        flush=True,
    )

    # ── Save layer indices to cache ────────────────────────────────────────
    if cache_dir is not None:
        np.savez(
            Path(cache_dir) / _LAYERS_FILE,
            endo_idx=endo_idx,
            mid_idx=mid_idx,
            epi_idx=epi_idx,
            sep_node_idx=sep_node_idx,
        )
        print(f"  Layer indices saved → {cache_dir}/{_LAYERS_FILE}", flush=True)

    return endo_idx, mid_idx, epi_idx, sep_node_idx
