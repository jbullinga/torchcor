"""
synthetic_torso.py
==================
Generate a parameterised elliptical-cylinder chest around the standalone
ventricle model (Case_1) for ECG forward solving and 3-D visualisation.

Geometry
--------
The torso is an *elliptical cylinder* with:
  * elliptical cross-section in X-Y  ( semi-axes  a = width/2,  b = depth/2 )
  * flat top  cap at  Z = +height_mm   (superior)
  * flat bottom cap at Z = -floor_mm    (inferior)

All coordinates follow the **cardiac convention** used throughout this
pipeline  (Z up = superior,  X = left,  Y = anterior).
The HTML viewer applies the same  (x,y,z) → (−x, z, y)  Three.js transform
as the activation viewer so the axes gizmo is consistent.

Outputs
-------
  <out_dir>/synthetic_torso.html   — interactive Three.js viewer
  <out_dir>/torso/                 — CARP-format surface mesh (.pts / .elem / .lon)
                                     ready for use with kcl_pipeline

Usage
-----
  python -m torchcor.ecg.synthetic_torso            # default parameters
  python -m torchcor.ecg.synthetic_torso --width 340 --depth 220 --height 160 --floor 80
"""
from __future__ import annotations

import argparse
import base64
import struct
import textwrap
from pathlib import Path

import numpy as np

# ── Default paths ─────────────────────────────────────────────────────────────
VENTRICLE_DIR = Path(r"C:\Users\bulli\cardiac_data\ventricle\Case_1")
DEFAULT_OUT   = Path(r"C:\Users\bulli\cardiac_data\synthetic_torso")

# ── Default torso geometry (mm) ───────────────────────────────────────────────
DEFAULT_WIDTH  = 320.0   # full chest width         (X, left–right)
DEFAULT_DEPTH  = 200.0   # full chest depth         (Y, anterior–posterior)
DEFAULT_HEIGHT = 160.0   # height above Z = 0       (superior)
DEFAULT_FLOOR  =  80.0   # depth  below Z = 0       (inferior / diaphragm)


# ==============================================================================
#  Mesh helpers
# ==============================================================================

def load_ventricle_surface(vtk_path: Path = VENTRICLE_DIR / "Case_1.vtk",
                            target_tris: int = 60_000) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Load Case_1.vtk, extract the outer surface, decimate, and return
    (verts_mm, tris, meta) where meta carries bounds and region info.
    """
    import pyvista as pv

    print(f"Loading ventricle from {vtk_path} …", flush=True)
    mesh = pv.read(str(vtk_path))
    print(f"  Loaded: {mesh.n_points} nodes, {mesh.n_cells} elements", flush=True)

    # Extract outer surface
    surface = mesh.extract_surface()

    # Collect scalar array names for coloring (region labels, etc.)
    scalar_names = list(surface.point_data.keys()) + list(surface.cell_data.keys())
    print(f"  Scalar arrays: {scalar_names}", flush=True)

    # Decimate if needed
    n_tris = surface.n_cells
    if n_tris > target_tris:
        ratio = 1.0 - target_tris / n_tris
        ratio = min(ratio, 0.95)
        surface = surface.decimate(ratio)
        surface = surface.triangulate()
        print(f"  Decimated → {surface.n_cells} triangles", flush=True)
    else:
        surface = surface.triangulate()
        print(f"  Triangulated → {surface.n_cells} triangles", flush=True)

    pts  = surface.points.astype(np.float32)

    # ── Auto-detect coordinate units ─────────────────────────────────────────
    # Typical cardiac mesh in mm: max extent ~200 mm
    # If coordinates are in µm the extent will be ~200,000 — divide by 1000
    extent = pts.max() - pts.min()
    if extent > 5_000:
        scale = 1.0 / 1000.0   # µm → mm
        print(f"  Coordinate extent {extent:.0f} — assuming micrometres, scaling ÷1000 → mm")
    elif extent > 500:
        scale = 1.0 / 10.0     # 0.1 mm → mm
        print(f"  Coordinate extent {extent:.0f} — scaling ÷10 → mm")
    else:
        scale = 1.0
        print(f"  Coordinate extent {extent:.1f} mm — no unit conversion needed")
    pts = (pts * scale).astype(np.float32)

    # Extract triangle connectivity
    faces = surface.faces.reshape(-1, 4)   # (n_tri, 4) — first col is always 3
    tris  = faces[:, 1:].astype(np.int32)

    # Bounding box
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    meta = dict(
        bounds_lo=lo, bounds_hi=hi,
        centre=(lo + hi) / 2,
        scalar_names=scalar_names,
        unit_scale=scale,
    )

    # Scalar data for coloring (first point-data array if available)
    scalars = None
    region_tag = None
    for name in surface.point_data.keys():
        data = surface.point_data[name]
        if np.issubdtype(data.dtype, np.integer):
            region_tag = data.astype(np.int32)
            print(f"  Region tag array: '{name}', unique values: {np.unique(region_tag)}")
            break
        elif scalars is None:
            scalars = data.astype(np.float32)

    meta["region_tag"] = region_tag
    meta["scalars"]    = scalars

    return pts, tris, meta


def auto_position_ventricle(pts: np.ndarray, meta: dict,
                              width: float, depth: float,
                              height: float, floor: float
                              ) -> np.ndarray:
    """
    Translate the ventricle so that:
      * centroid is at  (0, -depth*0.05, 0)   (slightly anterior)
      * inferior bound  sits at  Z ≈ -floor*0.2
    Returns translated points.
    """
    lo, hi = meta["bounds_lo"], meta["bounds_hi"]
    ctr = (lo + hi) / 2.0

    # Target centroid in the torso
    target_xy = np.array([0.0, -depth * 0.05, 0.0], dtype=np.float32)

    # Shift so inferior bound of heart is at about -floor*0.2
    z_shift = -floor * 0.2 - lo[2]
    offset  = target_xy - ctr
    offset[2] = z_shift

    return (pts + offset).astype(np.float32)


def make_torso_carp_mesh(width: float = DEFAULT_WIDTH,
                          depth: float = DEFAULT_DEPTH,
                          height: float = DEFAULT_HEIGHT,
                          floor: float  = DEFAULT_FLOOR,
                          n_theta: int  = 48,
                          n_z: int      = 20,
                          out_dir: Path = DEFAULT_OUT / "torso") -> Path:
    """
    Write a simple surface-only CARP mesh of the elliptical cylinder to out_dir.
    Files produced:
      torso.pts  — node coordinates
      torso.elem — surface triangles (tag = 1)
      torso.lon  — dummy fibre direction (required by pipeline)

    For a volumetric mesh suitable for FEM forward solving, install gmsh and
    use make_torso_volume_mesh() instead.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    a = width  / 2.0
    b = depth  / 2.0
    z_bot = -floor
    z_top =  height

    theta  = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    z_vals = np.linspace(z_bot, z_top, n_z)

    # ── Wall nodes (n_z × n_theta) ────────────────────────────────────────────
    TH, ZZ  = np.meshgrid(theta, z_vals)       # (n_z, n_theta)
    x_wall  = (a * np.cos(TH)).ravel()
    y_wall  = (b * np.sin(TH)).ravel()
    z_wall  = ZZ.ravel()
    wall_pts = np.column_stack([x_wall, y_wall, z_wall])

    # Cap centres
    top_ctr_idx = len(wall_pts)
    bot_ctr_idx = top_ctr_idx + 1
    all_pts = np.vstack([
        wall_pts,
        [[0.0, 0.0,  z_top]],   # top centre
        [[0.0, 0.0,  z_bot]],   # bottom centre
    ])

    # ── Triangles ─────────────────────────────────────────────────────────────
    tris = []
    for iz in range(n_z - 1):
        for it in range(n_theta):
            i00 = iz * n_theta + it
            i01 = iz * n_theta + (it + 1) % n_theta
            i10 = (iz + 1) * n_theta + it
            i11 = (iz + 1) * n_theta + (it + 1) % n_theta
            tris.append((i00, i01, i11))
            tris.append((i00, i11, i10))

    # Top cap  (outward normal = +Z  →  wind CCW from above)
    top_ring = (n_z - 1) * n_theta
    for it in range(n_theta):
        i0 = top_ring + it
        i1 = top_ring + (it + 1) % n_theta
        tris.append((top_ctr_idx, i0, i1))

    # Bottom cap (outward normal = −Z)
    for it in range(n_theta):
        i0 = it
        i1 = (it + 1) % n_theta
        tris.append((bot_ctr_idx, i1, i0))

    tris = np.array(tris, dtype=np.int32)

    # ── Write CARP files ──────────────────────────────────────────────────────
    pts_file  = out_dir / "torso.pts"
    elem_file = out_dir / "torso.elem"
    lon_file  = out_dir / "torso.lon"

    np.savetxt(str(pts_file),  all_pts, fmt="%.4f",
               header=str(len(all_pts)), comments="")
    with open(str(elem_file), "w") as f:
        f.write(f"{len(tris)}\n")
        for t in tris:
            f.write(f"Tr {t[0]} {t[1]} {t[2]} 1\n")
    with open(str(lon_file), "w") as f:
        f.write("1\n")
        for _ in range(len(tris)):
            f.write("1.0 0.0 0.0\n")

    print(f"  CARP mesh → {out_dir}  ({len(all_pts)} nodes, {len(tris)} triangles)")
    return out_dir


def make_torso_volume_mesh(width: float = DEFAULT_WIDTH,
                            depth: float = DEFAULT_DEPTH,
                            height: float = DEFAULT_HEIGHT,
                            floor: float  = DEFAULT_FLOOR,
                            lc: float     = 15.0,
                            out_dir: Path = DEFAULT_OUT / "torso") -> Path:
    """
    Generate a volumetric tetrahedral mesh of the elliptical cylinder using gmsh.
    Requires:  pip install gmsh
    lc : characteristic mesh length (mm). Smaller = finer mesh.
    """
    try:
        import gmsh
    except ImportError:
        raise ImportError(
            "gmsh is required for volume mesh generation.\n"
            "Install with:  pip install gmsh"
        )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.model.add("synthetic_torso")

    a = width  / 2.0
    b = depth  / 2.0
    z_top =  height
    z_bot = -floor

    # Build elliptical cylinder via boolean:
    # 1. Create a cylinder (radius=1), scale X/Y to a/b
    tag = gmsh.model.occ.addCylinder(0, 0, z_bot, 0, 0, z_top - z_bot, 1.0)
    gmsh.model.occ.dilate([(3, tag)], 0, 0, 0, a, b, 1.0)
    gmsh.model.occ.synchronize()

    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", lc)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", lc / 3.0)
    gmsh.model.mesh.generate(3)

    msh_path = str(out_dir / "torso.msh")
    gmsh.write(msh_path)
    gmsh.finalize()

    # Convert to CARP via meshio
    try:
        import meshio
        m = meshio.read(msh_path)
        pts  = m.points.astype(np.float32)
        tets = m.cells_dict.get("tetra", np.zeros((0, 4), dtype=int))
        np.savetxt(str(out_dir / "torso.pts"), pts, fmt="%.4f",
                   header=str(len(pts)), comments="")
        with open(str(out_dir / "torso.elem"), "w") as f:
            f.write(f"{len(tets)}\n")
            for tet in tets:
                f.write(f"Tt {tet[0]} {tet[1]} {tet[2]} {tet[3]} 1\n")
        with open(str(out_dir / "torso.lon"), "w") as f:
            f.write("1\n")
            for _ in range(len(tets)):
                f.write("1.0 0.0 0.0 0.0 1.0 0.0\n")
        print(f"  Volume mesh → {out_dir}  ({len(pts)} nodes, {len(tets)} tets)")
    except ImportError:
        print(f"  meshio not installed — raw .msh at {msh_path}")

    return out_dir


# ==============================================================================
#  HTML viewer
# ==============================================================================

def _f32_b64(arr: np.ndarray) -> str:
    """Pack a float32 array as base64."""
    return base64.b64encode(arr.astype(np.float32).tobytes()).decode()


def _i32_b64(arr: np.ndarray) -> str:
    """Pack an int32 array as base64."""
    return base64.b64encode(arr.astype(np.int32).tobytes()).decode()


def save_synthetic_torso_html(
    ventricle_vtk: Path  = VENTRICLE_DIR / "Case_1.vtk",
    out_dir:        Path  = DEFAULT_OUT,
    width:          float = DEFAULT_WIDTH,
    depth:          float = DEFAULT_DEPTH,
    height:         float = DEFAULT_HEIGHT,
    floor:          float = DEFAULT_FLOOR,
    target_tris:    int   = 60_000,
) -> Path:
    """
    Generate a self-contained Three.js HTML file showing:
      • the ventricle (solid, region-coloured if tags are present)
      • a synthetic elliptical-cylinder chest (semi-transparent, adjustable)
      • interactive sliders to resize the torso in real-time
      • axes gizmo and orbit controls
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load ventricle ────────────────────────────────────────────────────────
    pts_raw, tris, meta = load_ventricle_surface(ventricle_vtk, target_tris)
    pts = auto_position_ventricle(pts_raw, meta, width, depth, height, floor)

    # Update meta bounds to reflect translated position (used for camera)
    lo_t = pts.min(axis=0)
    hi_t = pts.max(axis=0)
    meta["bounds_lo"] = lo_t
    meta["bounds_hi"] = hi_t
    meta["centre"]    = (lo_t + hi_t) / 2.0

    # Apply coordinate transform: cardiac (x,y,z) → Three.js (−x, z, y)
    verts_3js = np.empty_like(pts)
    verts_3js[:, 0] = -pts[:, 0]
    verts_3js[:, 1] =  pts[:, 2]
    verts_3js[:, 2] =  pts[:, 1]

    # Per-vertex region colours
    region_tag = meta.get("region_tag")
    if region_tag is not None:
        unique_tags = np.unique(region_tag)
        # Map each tag to a distinct colour
        rng = np.random.default_rng(42)
        tag_colours: dict[int, tuple[float, float, float]] = {}
        # Pre-defined palette for common cardiac tags
        palette = {
            1:  (0.85, 0.25, 0.25),   # LV wall — red
            2:  (0.95, 0.45, 0.20),   # RV wall — orange
            3:  (0.60, 0.15, 0.15),   # septum  — dark red
            4:  (0.80, 0.30, 0.30),   # other
            24: (0.50, 0.70, 0.95),   # LV blood — blue
            25: (0.55, 0.75, 0.98),   # RV blood — blue
            34: (0.85, 0.25, 0.25),   # LV myocardium
            35: (0.95, 0.45, 0.20),   # RV myocardium
        }
        for t in unique_tags:
            tag_colours[int(t)] = palette.get(int(t),
                                              tuple(rng.uniform(0.3, 0.9, 3).tolist()))
        # Build vertex colour array
        vcols = np.zeros((len(pts), 3), dtype=np.float32)
        for t, col in tag_colours.items():
            mask = region_tag == t
            vcols[mask] = col
        has_regions = True
    else:
        # Uniform heart colour
        vcols = np.full((len(pts), 3), [0.80, 0.25, 0.25], dtype=np.float32)
        has_regions = False

    # Encode geometry for embedding
    verts_b64  = _f32_b64(verts_3js.ravel())
    tris_b64   = _i32_b64(tris.ravel())
    vcols_b64  = _f32_b64(vcols.ravel())
    n_verts    = len(pts)
    n_tris     = len(tris)

    # Centre of the ventricle in Three.js coords (for camera)
    ctr_cardiac = meta["centre"]
    cam_target  = [-ctr_cardiac[0], ctr_cardiac[2], ctr_cardiac[1]]

    # Camera pull-back: ~3× the larger of torso width or ventricle extent
    vent_extent = float(np.max(meta["bounds_hi"] - meta["bounds_lo"]))
    cam_dist    = max(width, depth, height + floor, vent_extent) * 2.5

    html = textwrap.dedent(f"""\
    <!DOCTYPE html>
    <html lang="en">
    <head>
    <meta charset="utf-8">
    <title>Synthetic Torso — Ventricle Viewer</title>
    <style>
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ background: #1a1a2e; color: #eee; font-family: 'Segoe UI', sans-serif;
           display: flex; flex-direction: column; height: 100vh; overflow: hidden; }}

    #header {{ padding: 8px 16px; background: #16213e; display: flex;
               align-items: center; gap: 16px; border-bottom: 1px solid #0f3460; }}
    #header h1 {{ font-size: 1rem; font-weight: 600; color: #89b4fa; }}

    #main {{ display: flex; flex: 1; min-height: 0; overflow: hidden; }}

    #canvas-wrap {{ flex: 1; position: relative; min-width: 0; overflow: hidden; }}
    canvas#c {{ position: absolute; top: 0; left: 0; display: block; }}


    #panel {{ width: 280px; background: #16213e; padding: 16px; overflow-y: auto;
              border-left: 1px solid #0f3460; display: flex; flex-direction: column; gap: 16px; }}

    .section {{ background: #0f3460; border-radius: 8px; padding: 12px; }}
    .section h3 {{ font-size: 0.78rem; text-transform: uppercase; letter-spacing: .08em;
                   color: #89b4fa; margin-bottom: 10px; }}

    .row {{ display: flex; align-items: center; justify-content: space-between;
            margin-bottom: 8px; }}
    .row:last-child {{ margin-bottom: 0; }}
    .row label {{ font-size: 0.82rem; color: #cdd6f4; width: 80px; flex-shrink: 0; }}
    .row input[type=range] {{ flex: 1; margin: 0 8px; accent-color: #89b4fa; }}
    .row .val {{ font-size: 0.82rem; color: #a6e3a1; width: 48px; text-align: right; }}

    .toggle-row {{ display: flex; align-items: center; gap: 8px; margin-bottom: 6px; }}
    .toggle-row label {{ font-size: 0.82rem; color: #cdd6f4; }}
    input[type=checkbox] {{ accent-color: #89b4fa; width: 15px; height: 15px; }}

    #info {{ font-size: 0.75rem; color: #6c7086; margin-top: auto; padding-top: 8px; }}
    </style>
    </head>
    <body>

    <div id="header">
      <h1>🫀 Synthetic Torso — Ventricle Viewer</h1>
      <span style="font-size:0.78rem;color:#6c7086;">
        Ventricle: {n_verts:,} nodes · {n_tris:,} triangles
        {'· Region-coloured' if has_regions else ''}
      </span>
    </div>

    <div id="main">
      <div id="canvas-wrap">
        <canvas id="c"></canvas>
      </div>

      <div id="panel">

        <!-- Torso geometry controls -->
        <div class="section">
          <h3>Torso Geometry (mm)</h3>
          <div class="row">
            <label>Width (X)</label>
            <input type="range" id="sl-width"  min="200" max="500" step="5" value="{width:.0f}">
            <span class="val" id="lbl-width">{width:.0f}</span>
          </div>
          <div class="row">
            <label>Depth (Y)</label>
            <input type="range" id="sl-depth"  min="100" max="400" step="5" value="{depth:.0f}">
            <span class="val" id="lbl-depth">{depth:.0f}</span>
          </div>
          <div class="row">
            <label>Height (+Z)</label>
            <input type="range" id="sl-height" min="50"  max="400" step="5" value="{height:.0f}">
            <span class="val" id="lbl-height">{height:.0f}</span>
          </div>
          <div class="row">
            <label>Floor (−Z)</label>
            <input type="range" id="sl-floor"  min="20"  max="200" step="5" value="{floor:.0f}">
            <span class="val" id="lbl-floor">{floor:.0f}</span>
          </div>
        </div>

        <!-- Display options -->
        <div class="section">
          <h3>Display</h3>
          <div class="toggle-row">
            <input type="checkbox" id="chk-torso" checked>
            <label for="chk-torso">Show torso</label>
          </div>
          <div class="toggle-row">
            <input type="checkbox" id="chk-wire" checked>
            <label for="chk-wire">Wireframe overlay</label>
          </div>
          <div class="toggle-row">
            <input type="checkbox" id="chk-heart" checked>
            <label for="chk-heart">Show ventricle</label>
          </div>
          <div class="row">
            <label>Opacity</label>
            <input type="range" id="sl-opacity" min="0.05" max="0.60" step="0.05" value="0.18">
            <span class="val" id="lbl-opacity">0.18</span>
          </div>
          <div class="row">
            <label>Resolution</label>
            <input type="range" id="sl-res" min="16" max="80" step="4" value="48">
            <span class="val" id="lbl-res">48</span>
          </div>
        </div>

        <!-- Dimensions readout -->
        <div class="section">
          <h3>Ventricle Bounds (mm)</h3>
          <div style="font-size:0.78rem; color:#cdd6f4; line-height:1.7;">
            X: {meta['bounds_lo'][0]:.1f} → {meta['bounds_hi'][0]:.1f}<br>
            Y: {meta['bounds_lo'][1]:.1f} → {meta['bounds_hi'][1]:.1f}<br>
            Z: {meta['bounds_lo'][2]:.1f} → {meta['bounds_hi'][2]:.1f}
          </div>
        </div>

        <div id="info">
          Drag to rotate · Scroll to zoom · Right-drag to pan
        </div>
      </div>
    </div>

    <!-- Three.js r158 CDN -->
    <script src="https://cdn.jsdelivr.net/npm/three@0.158.0/build/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.158.0/examples/js/controls/OrbitControls.js"></script>

    <script>
    // ── Embedded ventricle geometry ───────────────────────────────────────────
    const N_VERTS  = {n_verts};
    const N_TRIS   = {n_tris};
    const VERTS_B64  = "{verts_b64}";
    const TRIS_B64   = "{tris_b64}";
    const VCOLS_B64  = "{vcols_b64}";

    function b64ToF32(b64) {{
      const bin = atob(b64);
      const buf = new ArrayBuffer(bin.length);
      const u8  = new Uint8Array(buf);
      for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
      return new Float32Array(buf);
    }}
    function b64ToI32(b64) {{
      const bin = atob(b64);
      const buf = new ArrayBuffer(bin.length);
      const u8  = new Uint8Array(buf);
      for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
      return new Int32Array(buf);
    }}

    const heartVerts = b64ToF32(VERTS_B64);
    const heartIdx   = b64ToI32(TRIS_B64);
    const heartCols  = b64ToF32(VCOLS_B64);

    // ── Scene setup ───────────────────────────────────────────────────────────
    const canvas   = document.getElementById('c');
    const renderer = new THREE.WebGLRenderer({{ canvas, antialias: true }});
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.autoClear = false;

    const scene    = new THREE.Scene();
    scene.background = new THREE.Color(0x1a1a2e);

    const camera   = new THREE.PerspectiveCamera(40, 1, 1, {cam_dist * 10:.0f});
    const target   = new THREE.Vector3({cam_target[0]:.1f}, {cam_target[1]:.1f}, {cam_target[2]:.1f});
    camera.position.set(target.x, target.y + 50, target.z + {cam_dist:.1f});
    camera.lookAt(target);

    const controls = new THREE.OrbitControls(camera, canvas);
    controls.target.copy(target);
    controls.enableDamping  = true;
    controls.dampingFactor  = 0.08;
    controls.minDistance    = {max(10, vent_extent * 0.1):.0f};
    controls.maxDistance    = {cam_dist * 8:.0f};
    controls.update();

    // Lights
    scene.add(new THREE.AmbientLight(0xffffff, 0.5));
    const dLight = new THREE.DirectionalLight(0xffffff, 0.8);
    dLight.position.set(1, 2, 1);
    scene.add(dLight);
    const dLight2 = new THREE.DirectionalLight(0x8888ff, 0.3);
    dLight2.position.set(-1, -1, -1);
    scene.add(dLight2);

    // ── Ventricle mesh ────────────────────────────────────────────────────────
    const heartGeo = new THREE.BufferGeometry();
    heartGeo.setAttribute('position', new THREE.BufferAttribute(heartVerts, 3));
    heartGeo.setAttribute('color',    new THREE.BufferAttribute(heartCols,  3));
    heartGeo.setIndex(new THREE.BufferAttribute(new Uint32Array(heartIdx.buffer,
                                                                 heartIdx.byteOffset,
                                                                 heartIdx.length), 1));
    heartGeo.computeVertexNormals();

    const heartMat = new THREE.MeshPhongMaterial({{
      vertexColors: true,
      shininess:    40,
      side: THREE.DoubleSide,
    }});
    const heartMesh = new THREE.Mesh(heartGeo, heartMat);
    scene.add(heartMesh);

    // ── Torso mesh (built/rebuilt from sliders) ───────────────────────────────
    let torsoMesh  = null;
    let torsoWire  = null;

    function buildTorso(widthMm, depthMm, heightMm, floorMm, nTheta) {{
      const a    = widthMm / 2;
      const b    = depthMm / 2;
      const zBot = -floorMm;
      const zTop =  heightMm;
      const nZ   = Math.max(8, Math.round(nTheta / 2));

      const pos = [];
      const idx = [];

      // ── Side wall ────────────────────────────────────────────────────────
      for (let iz = 0; iz <= nZ; iz++) {{
        const cz = zBot + (zTop - zBot) * iz / nZ;  // cardiac Z
        for (let it = 0; it < nTheta; it++) {{
          const angle = 2 * Math.PI * it / nTheta;
          const cx = a * Math.cos(angle);   // cardiac X
          const cy = b * Math.sin(angle);   // cardiac Y
          // Transform: cardiac(x,y,z) → Three.js(−x, z, y)
          pos.push(-cx, cz, cy);
        }}
      }}

      // Side triangles
      for (let iz = 0; iz < nZ; iz++) {{
        for (let it = 0; it < nTheta; it++) {{
          const a0 = iz       * nTheta + it;
          const a1 = iz       * nTheta + (it + 1) % nTheta;
          const b0 = (iz + 1) * nTheta + it;
          const b1 = (iz + 1) * nTheta + (it + 1) % nTheta;
          idx.push(a0, a1, b1);
          idx.push(a0, b1, b0);
        }}
      }}

      // ── Top cap  (cardiac Z = zTop → Three.js Y = zTop) ──────────────────
      const topRing   = nZ * nTheta;
      const topCtrIdx = (nZ + 1) * nTheta;
      pos.push(0, zTop, 0);            // centre of top cap
      for (let it = 0; it < nTheta; it++) {{
        const i0 = topRing + it;
        const i1 = topRing + (it + 1) % nTheta;
        idx.push(topCtrIdx, i0, i1);
      }}

      // ── Bottom cap (cardiac Z = zBot → Three.js Y = zBot) ────────────────
      const botRing   = 0;
      const botCtrIdx = topCtrIdx + 1;
      pos.push(0, zBot, 0);            // centre of bottom cap
      for (let it = 0; it < nTheta; it++) {{
        const i0 = botRing + it;
        const i1 = botRing + (it + 1) % nTheta;
        idx.push(botCtrIdx, i1, i0);
      }}

      const geom = new THREE.BufferGeometry();
      geom.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
      geom.setIndex(idx);
      geom.computeVertexNormals();
      return geom;
    }}

    function rebuildTorso() {{
      const w  = parseFloat(document.getElementById('sl-width').value);
      const d  = parseFloat(document.getElementById('sl-depth').value);
      const h  = parseFloat(document.getElementById('sl-height').value);
      const f  = parseFloat(document.getElementById('sl-floor').value);
      const nt = parseInt(document.getElementById('sl-res').value);
      const op = parseFloat(document.getElementById('sl-opacity').value);

      if (torsoMesh) {{ scene.remove(torsoMesh); torsoMesh.geometry.dispose(); }}
      if (torsoWire) {{ scene.remove(torsoWire);  torsoWire.geometry.dispose(); }}

      const geom = buildTorso(w, d, h, f, nt);

      const mat = new THREE.MeshPhongMaterial({{
        color:       0xe8c4a0,
        transparent: true,
        opacity:     op,
        side:        THREE.DoubleSide,
        depthWrite:  false,
        shininess:   20,
      }});
      torsoMesh = new THREE.Mesh(geom, mat);
      scene.add(torsoMesh);

      if (document.getElementById('chk-wire').checked) {{
        const wireMat = new THREE.MeshBasicMaterial({{
          color:     0x89b4fa,
          wireframe: true,
          transparent: true,
          opacity:   0.12,
        }});
        torsoWire = new THREE.Mesh(geom.clone(), wireMat);
        scene.add(torsoWire);
      }}
    }}

    // ── Axes gizmo (scissored corner of main canvas) ─────────────────────────
    const axesScene  = new THREE.Scene();
    const axesCamera = new THREE.PerspectiveCamera(50, 1, 0.1, 20);

    function makeAxLine(x, y, z, color) {{
      const g = new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(0, 0, 0), new THREE.Vector3(x, y, z)
      ]);
      return new THREE.Line(g, new THREE.LineBasicMaterial({{ color, linewidth: 2 }}));
    }}
    function axisSprite(text, color) {{
      const c = document.createElement('canvas'); c.width = 64; c.height = 64;
      const ctx = c.getContext('2d');
      ctx.fillStyle = color;
      ctx.font = 'bold 44px sans-serif';
      ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      ctx.fillText(text, 32, 32);
      const tex = new THREE.CanvasTexture(c);
      const sp  = new THREE.Sprite(new THREE.SpriteMaterial({{ map: tex, transparent: true }}));
      sp.scale.set(0.4, 0.4, 1);
      return sp;
    }}

    axesScene.add(makeAxLine(-1, 0, 0, '#ff4444'));  // cardiac +X → red
    axesScene.add(makeAxLine(0,  1, 0, '#4488ff'));  // cardiac +Z → blue (up)
    axesScene.add(makeAxLine(0,  0, 1, '#44ff44'));  // cardiac +Y → green

    const spX = axisSprite('X', '#ff4444'); spX.position.set(-1.35, 0,    0   );
    const spZ = axisSprite('Z', '#4488ff'); spZ.position.set(0,     1.35, 0   );
    const spY = axisSprite('Y', '#44ff44'); spY.position.set(0,     0,    1.35);
    axesScene.add(spX); axesScene.add(spZ); axesScene.add(spY);
    axesScene.add(new THREE.AmbientLight(0xffffff, 1.0));

    // ── Slider wiring ─────────────────────────────────────────────────────────
    function wireSlider(id, lblId, rebuild) {{
      const sl  = document.getElementById(id);
      const lbl = document.getElementById(lblId);
      sl.addEventListener('input', () => {{ lbl.textContent = sl.value; if (rebuild) rebuildTorso(); }});
    }}
    wireSlider('sl-width',   'lbl-width',   true);
    wireSlider('sl-depth',   'lbl-depth',   true);
    wireSlider('sl-height',  'lbl-height',  true);
    wireSlider('sl-floor',   'lbl-floor',   true);
    wireSlider('sl-opacity', 'lbl-opacity', true);
    wireSlider('sl-res',     'lbl-res',     true);

    document.getElementById('chk-torso').addEventListener('change', e => {{
      if (torsoMesh) torsoMesh.visible = e.target.checked;
      if (torsoWire) torsoWire.visible = e.target.checked;
    }});
    document.getElementById('chk-wire').addEventListener('change', () => rebuildTorso());
    document.getElementById('chk-heart').addEventListener('change', e => {{
      heartMesh.visible = e.target.checked;
    }});

    // ── Resize handler ────────────────────────────────────────────────────────
    const PANEL_W = 280;
    function canvasSize() {{
      const hh = document.getElementById('header').offsetHeight || 48;
      return {{ w: Math.max(1, window.innerWidth - PANEL_W),
                h: Math.max(1, window.innerHeight - hh) }};
    }}
    function onResize() {{
      const {{ w, h }} = canvasSize();
      renderer.setSize(w, h);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
    }}
    window.addEventListener('resize', onResize);
    onResize();   // size immediately — window dimensions are always available

    // ── Render loop ───────────────────────────────────────────────────────────
    const GS = 120;  // gizmo size in CSS pixels
    function animate() {{
      requestAnimationFrame(animate);
      controls.update();

      const {{ w, h }} = canvasSize();
      if (renderer.domElement.width  !== Math.round(w * window.devicePixelRatio) ||
          renderer.domElement.height !== Math.round(h * window.devicePixelRatio)) {{
        renderer.setSize(w, h);
        camera.aspect = w / h;
        camera.updateProjectionMatrix();
      }}

      // ── Main scene (full viewport) ────────────────────────────────────────
      renderer.setViewport(0, 0, w, h);
      renderer.setScissor(0, 0, w, h);
      renderer.setScissorTest(true);
      renderer.clear();
      renderer.render(scene, camera);

      // ── Axes gizmo (lower-left scissored corner) ──────────────────────────
      renderer.setViewport(10, 10, GS, GS);
      renderer.setScissor(10, 10, GS, GS);
      renderer.clearDepth();
      const camDir = camera.position.clone().sub(controls.target).normalize();
      axesCamera.position.copy(camDir.multiplyScalar(2.5));
      axesCamera.lookAt(0, 0, 0);
      axesCamera.up.copy(camera.up);
      renderer.render(axesScene, axesCamera);

      renderer.setScissorTest(false);
    }}

    // ── Init ──────────────────────────────────────────────────────────────────
    rebuildTorso();
    animate();
    </script>
    </body>
    </html>
    """)

    out_path = out_dir / "synthetic_torso.html"
    out_path.write_text(html, encoding="utf-8")
    print(f"  Saved → {out_path}  ({out_path.stat().st_size // 1024} KB)")
    return out_path


# ==============================================================================
#  CLI
# ==============================================================================

def main() -> None:
    p = argparse.ArgumentParser(
        description="Generate synthetic elliptical-cylinder chest + HTML viewer"
    )
    p.add_argument("--ventricle", default=str(VENTRICLE_DIR / "Case_1.vtk"),
                   help="Path to Case_1.vtk (or any .vtk / .vtu ventricle mesh)")
    p.add_argument("--out",    default=str(DEFAULT_OUT),
                   help="Output directory")
    p.add_argument("--width",  type=float, default=DEFAULT_WIDTH,
                   help="Torso width  mm  (X, left-right)    default=%(default)s")
    p.add_argument("--depth",  type=float, default=DEFAULT_DEPTH,
                   help="Torso depth  mm  (Y, ant-post)       default=%(default)s")
    p.add_argument("--height", type=float, default=DEFAULT_HEIGHT,
                   help="Torso height mm  (Z above 0)         default=%(default)s")
    p.add_argument("--floor",  type=float, default=DEFAULT_FLOOR,
                   help="Torso floor  mm  (Z below 0)         default=%(default)s")
    p.add_argument("--tris",   type=int,   default=60_000,
                   help="Target triangles for ventricle surface (default=%(default)s)")
    p.add_argument("--mesh",   action="store_true",
                   help="Also write CARP surface mesh to <out>/torso/")
    p.add_argument("--volume", action="store_true",
                   help="Also write volumetric tet mesh via gmsh (requires: pip install gmsh meshio)")
    p.add_argument("--no-html", action="store_true",
                   help="Skip HTML generation")
    args = p.parse_args()

    out_dir = Path(args.out)

    if not args.no_html:
        print("\n── HTML viewer ──────────────────────────────────────────────")
        save_synthetic_torso_html(
            ventricle_vtk = Path(args.ventricle),
            out_dir       = out_dir,
            width         = args.width,
            depth         = args.depth,
            height        = args.height,
            floor         = args.floor,
            target_tris   = args.tris,
        )

    if args.mesh:
        print("\n── CARP surface mesh ────────────────────────────────────────")
        make_torso_carp_mesh(
            width=args.width, depth=args.depth,
            height=args.height, floor=args.floor,
            out_dir=out_dir / "torso",
        )

    if args.volume:
        print("\n── Volumetric mesh (gmsh) ───────────────────────────────────")
        make_torso_volume_mesh(
            width=args.width, depth=args.depth,
            height=args.height, floor=args.floor,
            out_dir=out_dir / "torso",
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
