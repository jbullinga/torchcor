"""
torchcor/ecg/kcl_visualize.py
==============================
Post-hoc (and pipeline-debug) visualisation for the KCL simulation.

Interactive viewers
-------------------
  --anatomy     Interactive 3-D torso with tissues + ECG electrodes
  --ecg         12-lead ECG  →  saved as Plotly HTML (opens in browser)
  --activation  Static AT/APD map on biventricular mesh  (pyvista)
  --animate     Animated Vm(t) wavefront with time slider + play/pause  (pyvista)
  --all         All of the above in sequence

Portable file export  (no Python needed to open)
-------------------------------------------------
  --save-ecg        Save ecg_12lead.html  (Plotly, opens in browser)
  --save-anatomy    Save anatomy.html     (vtk.js WebGL, opens in browser)
  --save-activation Save activation.html  (vtk.js WebGL, opens in browser)
  --save-mp4        Save activation_movie.mp4  (H.264 video)
  --save-bspm       Save bspm_3d.html        (Three.js, opens in browser)
  --save-all        All of the above (except --save-bspm, which needs --bspm first)

Usage
-----
    python -m torchcor.ecg.kcl_visualize --ecg
    python -m torchcor.ecg.kcl_visualize --animate
    python -m torchcor.ecg.kcl_visualize --animate --save-mp4
    python -m torchcor.ecg.kcl_visualize --save-all
    python -m torchcor.ecg.kcl_visualize --all
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# ── Default paths (match kcl_pipeline.py) ─────────────────────────────────────
_KCL_DIR        = Path(r"C:\Users\bulli\cardiac_data\kcl_torso\KCL_torso3")
DEFAULT_VTU     = _KCL_DIR / "KCL_torso3_anatomy.vtu"
DEFAULT_SIM_DIR = _KCL_DIR / "sim"
DEFAULT_RESULT  = _KCL_DIR / "sim" / "results"

# Vm colormap limits (mV)  — resting ~-85 mV, plateau ~40 mV
VM_CLIM = (-90.0, 40.0)


# ==============================================================================
#  Shared mesh loader  (used by activation and animation)
# ==============================================================================

def _load_ventricular_mesh(sim_dir: Path):
    """
    Load the biventricular CARP mesh.

    Returns
    -------
    nodes_mm  : (N, 3) float64
    elem_conn : (M, 4) int32   — all elements
    elem_regs : (M,)   int32
    wall_mask : (M,)   bool    — True for wall elements (tags 34, 35)
    """
    import pandas as pd

    vent_dir  = sim_dir / "ventricles"
    pts_file  = vent_dir / "ventricles.pts"
    elem_file = vent_dir / "ventricles.elem"

    for f in (pts_file, elem_file):
        if not f.exists():
            raise FileNotFoundError(
                f"Missing: {f}\n"
                "Run the pipeline first:  python -m torchcor.ecg.kcl_pipeline"
            )

    print("Loading ventricular CARP mesh...", flush=True)
    nodes_um  = np.loadtxt(pts_file, skiprows=1, dtype=np.float64)
    nodes_mm  = nodes_um / 1000.0

    df_elem   = pd.read_csv(elem_file, sep=r"\s+", header=None, skiprows=1,
                            dtype={0: str}, engine="c")
    elem_conn = df_elem.iloc[:, 1:5].values.astype(np.int32)
    elem_regs = df_elem.iloc[:, 5].values.astype(np.int32)

    wall_mask = np.isin(elem_regs, [34, 35])
    print(f"  Nodes: {len(nodes_mm):,}   Wall elements (34,35): {wall_mask.sum():,}")
    return nodes_mm, elem_conn, elem_regs, wall_mask


def _build_pv_grid(nodes_mm, elem_conn, mask):
    """Build a pyvista UnstructuredGrid from a tetrahedral subset."""
    import pyvista as pv
    sub_conn  = elem_conn[mask]
    n         = len(sub_conn)
    cells_vtk = np.column_stack(
        [np.full(n, 4, dtype=np.int64), sub_conn]
    ).ravel()
    cell_types = np.full(n, pv.CellType.TETRA, dtype=np.uint8)
    return pv.UnstructuredGrid(cells_vtk, cell_types, nodes_mm)


# ==============================================================================
#  1. Anatomy viewer  (viewer_3d.launch)
# ==============================================================================

def show_anatomy(vtu_path: Path = DEFAULT_VTU,
                 save_html: bool = False,
                 out_dir: Path = DEFAULT_RESULT) -> None:
    """
    Open the interactive 3-D tissue viewer with ECG electrode discs.
    If save_html=True also exports anatomy.html via pyvista vtk.js backend.
    """
    sys.path.insert(0, str(Path(__file__).parent))
    from viewer_3d import launch
    print(f"Opening anatomy viewer: {vtu_path.name}")
    launch(vtu_path)  # blocks until window closed

    if save_html:
        _export_anatomy_html(vtu_path, out_dir)


def _export_anatomy_html(vtu_path: Path, out_dir: Path) -> None:
    """Quick surface export of the VTU file as a standalone HTML."""
    import pyvista as pv
    print("Exporting anatomy HTML...", flush=True)
    mesh = pv.read(str(vtu_path))
    pl = pv.Plotter(off_screen=True, window_size=(1200, 900))
    pl.add_mesh(mesh, scalars="elemTag" if "elemTag" in mesh.array_names else None,
                cmap="tab20", show_scalar_bar=True)
    pl.set_background("white")
    pl.reset_camera()
    html_path = out_dir / "anatomy.html"
    out_dir.mkdir(parents=True, exist_ok=True)
    pl.export_html(str(html_path))
    print(f"  Saved -> {html_path}")


# ==============================================================================
#  2. 12-lead ECG  (Plotly HTML — self-contained, browser-based)
# ==============================================================================

def show_ecg(result_dir: Path = DEFAULT_RESULT,
             save_html: bool = True) -> None:
    """
    Render 12-lead ECG as a Plotly HTML file and open it in the default browser.
    The HTML file is saved to result_dir/ecg_12lead.html regardless of save_html
    (it's cheap and always useful).
    """
    csv_path = result_dir / "ecg_12lead.csv"
    pt_path  = result_dir / "ecg_12lead.pt"

    # ── Load data ─────────────────────────────────────────────────────────
    if csv_path.exists():
        import pandas as pd
        df   = pd.read_csv(csv_path)
        t_ms = df["time_ms"].values
        ecg  = {col: df[col].values for col in df.columns if col != "time_ms"}
        print(f"Loaded ECG from {csv_path.name}  ({len(t_ms)} time points)")
    elif pt_path.exists():
        import torch
        data  = torch.load(str(pt_path), map_location="cpu")
        ecg   = {k: v.float().numpy() for k, v in data.items()}
        n_t   = next(iter(ecg.values())).shape[0]
        t_ms  = np.arange(n_t).astype(np.float32)
        print(f"Loaded ECG from {pt_path.name}  ({n_t} time points)")
    else:
        raise FileNotFoundError(
            f"No ECG results found in {result_dir}\n"
            "Run the pipeline first:  python -m torchcor.ecg.kcl_pipeline"
        )

    html_path = _save_ecg_plotly(ecg, t_ms, result_dir)

    # Open in default browser
    import webbrowser
    webbrowser.open(html_path.as_uri())


def _save_ecg_plotly(ecg: dict, t_ms: np.ndarray, out_dir: Path) -> Path:
    """
    Save 12-lead ECG as a self-contained Plotly HTML file.

    Returns path to the saved file.
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        raise ImportError(
            "plotly is required for HTML ECG export.\n"
            "Install with:  pip install plotly"
        )

    LEAD_ORDER = [
        ["I",   "aVR", "V1", "V4"],
        ["II",  "aVL", "V2", "V5"],
        ["III", "aVF", "V3", "V6"],
    ]
    flat_leads    = [l for row in LEAD_ORDER for l in row]
    subplot_titles = flat_leads

    fig = make_subplots(
        rows=3, cols=4,
        subplot_titles=subplot_titles,
        shared_xaxes=False,
        horizontal_spacing=0.07,
        vertical_spacing=0.12,
    )

    for row_idx, row_leads in enumerate(LEAD_ORDER, start=1):
        for col_idx, lead in enumerate(row_leads, start=1):
            sig = np.asarray(ecg.get(lead, []), dtype=np.float32)
            if len(sig) == 0:
                continue
            fig.add_trace(
                go.Scatter(
                    x=t_ms, y=sig,
                    mode="lines",
                    line=dict(color="black", width=1.2),
                    name=lead,
                    showlegend=False,
                    hovertemplate=(
                        f"<b>{lead}</b><br>"
                        "t = %{x:.1f} ms<br>"
                        "Vm = %{y:.3f} mV"
                        "<extra></extra>"
                    ),
                ),
                row=row_idx, col=col_idx,
            )
            # Zero line
            fig.add_hline(
                y=0, line_color="#cccccc", line_width=0.8,
                row=row_idx, col=col_idx,
            )
            # Shade the QRS complex (±2% of duration around peak |amplitude|)
            if len(sig) > 20:
                pk = int(np.argmax(np.abs(sig)))
                hw = max(5, int(len(sig) * 0.02))
                t0 = float(t_ms[max(0, pk - hw)])
                t1 = float(t_ms[min(len(t_ms) - 1, pk + hw)])
                fig.add_vrect(
                    x0=t0, x1=t1,
                    fillcolor="steelblue", opacity=0.08, line_width=0,
                    row=row_idx, col=col_idx,
                )

    fig.update_layout(
        title=dict(
            text="12-Lead ECG — KCL torso model  (Ten Tusscher-Panfilov 2006)",
            font=dict(size=15, family="Arial"),
        ),
        height=750,
        width=1300,
        plot_bgcolor="#f5f5ee",
        paper_bgcolor="#f5f5ee",
        margin=dict(l=40, r=20, t=80, b=40),
    )
    fig.update_annotations(font_size=12, font_family="Arial Black")
    fig.update_xaxes(title_text="ms", title_font_size=9, tickfont_size=8)
    fig.update_yaxes(title_text="mV", title_font_size=9, tickfont_size=8)

    out_dir.mkdir(parents=True, exist_ok=True)
    html_path = out_dir / "ecg_12lead.html"
    fig.write_html(str(html_path), include_plotlyjs="cdn")
    print(f"  Saved ECG HTML -> {html_path}")
    return html_path


# ==============================================================================
#  3. Static activation-time / APD map  (pyvista)
# ==============================================================================

def show_activation(
    sim_dir: Path = DEFAULT_SIM_DIR,
    result_dir: Path = DEFAULT_RESULT,
    save_html: bool = False,
) -> None:
    """
    Show (and optionally export) the static AT / APD map on wall elements.
    For the time-animated wavefront use show_activation_animated().
    """
    import pyvista as pv

    at_file = result_dir / "activation_times.npy"
    rt_file = result_dir / "repolarization_times.npy"

    if not at_file.exists():
        raise FileNotFoundError(
            f"Missing: {at_file}\n"
            "Run the pipeline first:  python -m torchcor.ecg.kcl_pipeline"
        )

    nodes_mm, elem_conn, elem_regs, wall_mask = _load_ventricular_mesh(sim_dir)
    grid = _build_pv_grid(nodes_mm, elem_conn, wall_mask)

    at_ms = np.load(str(at_file)).astype(np.float32)
    has_rt = rt_file.exists()
    rt_ms  = np.load(str(rt_file)).astype(np.float32) if has_rt else None

    grid.point_data["AT (ms)"] = at_ms
    if has_rt:
        grid.point_data["RT (ms)"]  = rt_ms
        grid.point_data["APD (ms)"] = np.clip(rt_ms - at_ms, 0, None)

    at_act  = at_ms[at_ms < at_ms.max() * 0.98]
    at_clim = (0.0, float(np.percentile(at_act, 99)) if len(at_act) else 40.0)
    print(f"  AT range: {at_ms.min():.1f} - {at_ms.max():.1f} ms"
          f"  (colour clim 0 - {at_clim[1]:.0f} ms)")

    shape = (1, 2) if has_rt else (1, 1)
    win_w = 1400 if has_rt else 800
    off   = save_html  # off_screen when exporting HTML

    pl = pv.Plotter(
        title="Cardiac Activation Map",
        shape=shape,
        window_size=(win_w, 700),
        off_screen=off,
    )

    cam_pos = (200, 600, 200)
    cam_fp  = (0, 0, 100)
    cam_up  = (0, 0, 1)

    # AT panel
    pl.subplot(0, 0)
    pl.add_mesh(
        grid.threshold(list(at_clim), scalars="AT (ms)"),
        scalars="AT (ms)", cmap="jet", clim=list(at_clim),
        smooth_shading=True, show_scalar_bar=True,
        scalar_bar_args=dict(title="AT (ms)", width=0.35, position_x=0.62),
    )
    pl.add_text("Activation Time", position="upper_edge",
                font_size=10, color="black")
    pl.set_background("white")
    pl.add_axes(line_width=2)
    pl.camera.position    = cam_pos
    pl.camera.focal_point = cam_fp
    pl.camera.up          = cam_up
    pl.reset_camera()

    if has_rt:
        pl.subplot(0, 1)
        apd_arr = grid.point_data["APD (ms)"]
        apd_max = float(np.percentile(apd_arr[apd_arr > 0], 99)) if (apd_arr > 0).any() else 400
        pl.add_mesh(
            grid, scalars="APD (ms)", cmap="RdYlGn_r",
            clim=[0, apd_max], smooth_shading=True, show_scalar_bar=True,
            scalar_bar_args=dict(title="APD (ms)", width=0.35, position_x=0.62),
        )
        pl.add_text("Action Potential Duration", position="upper_edge",
                    font_size=10, color="black")
        pl.set_background("white")
        pl.add_axes(line_width=2)
        pl.camera.position    = cam_pos
        pl.camera.focal_point = cam_fp
        pl.camera.up          = cam_up
        pl.reset_camera()

    if save_html:
        html_path = result_dir / "activation.html"
        pl.export_html(str(html_path))
        print(f"  Saved activation HTML -> {html_path}")
    else:
        pl.show()


# ==============================================================================
#  4. Animated Vm(t) wavefront  —  slider + play/pause + optional MP4
# ==============================================================================

def show_activation_animated(
    sim_dir: Path = DEFAULT_SIM_DIR,
    result_dir: Path = DEFAULT_RESULT,
    save_mp4: bool = False,
    fps: int = 25,
    mp4_path: Path | None = None,
) -> None:
    """
    Animated transmembrane potential Vm(t) on the biventricular wall mesh.

    Interactive mode (default)
    --------------------------
    * A horizontal slider at the bottom of the window controls time.
      Drag to scrub; the 3-D mesh updates instantly.
    * A green/red PLAY button toggles automatic playback at ~fps frames/sec.
    * The current time is shown in the upper-right corner.
    * Rotate / zoom the 3-D view freely at any time.

    MP4 export  (--save-mp4)
    -------------------------
    Renders off-screen and saves an H.264 MP4.  No interactive window opens.
    Requires imageio[ffmpeg]:  pip install imageio[ffmpeg]
    """
    import pyvista as pv
    import torch

    vm_path = result_dir / "Vm.pt"
    if not vm_path.exists():
        raise FileNotFoundError(
            f"Missing: {vm_path}\n"
            "Run the pipeline first:  python -m torchcor.ecg.kcl_pipeline"
        )

    nodes_mm, elem_conn, elem_regs, wall_mask = _load_ventricular_mesh(sim_dir)
    grid = _build_pv_grid(nodes_mm, elem_conn, wall_mask)

    print(f"Loading Vm(t) from {vm_path} ...", flush=True)
    Vm_all = torch.load(str(vm_path), map_location="cpu").numpy().astype(np.float32)
    T_frames, N_nodes = Vm_all.shape
    dt_ms = 1.0   # 1 ms per snapshot (601 frames over 600 ms)
    print(f"  Vm shape: {Vm_all.shape}  ({(T_frames-1)*dt_ms:.0f} ms total)")

    # ── Colour bar range: fixed so the wavefront pops clearly ────────────
    vm_clim = list(VM_CLIM)   # [-90, 40] mV

    # ── Set initial frame ─────────────────────────────────────────────────
    grid.point_data["Vm (mV)"] = Vm_all[0]

    if save_mp4:
        _save_activation_mp4(
            grid=grid, Vm_all=Vm_all, T_frames=T_frames, dt_ms=dt_ms,
            vm_clim=vm_clim, nodes_mm=nodes_mm,
            out_path=mp4_path or result_dir / "activation_movie.mp4",
            fps=fps,
        )
        return

    # ── Interactive window ────────────────────────────────────────────────
    pl = pv.Plotter(
        title="Cardiac Activation Spread  |  KCL biventricular mesh",
        window_size=(1200, 850),
    )

    actor = pl.add_mesh(
        grid, scalars="Vm (mV)", cmap="RdBu_r",
        clim=vm_clim, smooth_shading=True,
        scalar_bar_args=dict(
            title="Vm (mV)", width=0.08, height=0.55,
            position_x=0.91, position_y=0.25,
            label_font_size=11, title_font_size=12,
        ),
    )

    # Use coordinate tuple → creates vtkTextActor (has SetInput);
    # string positions create vtkCornerAnnotation (different API)
    time_label = pl.add_text(
        "t = 0.0 ms", position=(0.74, 0.93),
        font_size=12, color="black",
        viewport=True,
    )

    # Shared mutable state
    state = {"t": 0, "playing": False}

    # ── Slider callback ────────────────────────────────────────────────────
    def set_frame(value: float) -> None:
        t = int(np.clip(round(value), 0, T_frames - 1))
        state["t"] = t
        grid.point_data["Vm (mV)"] = Vm_all[t]
        time_label.SetInput(f"t = {t * dt_ms:.0f} ms")

    slider = pl.add_slider_widget(
        callback=set_frame,
        rng=[0.0, float(T_frames - 1)],
        value=0.0,
        title="Time (ms)",
        pointa=(0.04, 0.06),   # left anchor
        pointb=(0.88, 0.06),   # right anchor
        style="modern",
        fmt="%.0f",
        slider_width=0.025,
        tube_width=0.008,
        color="steelblue",
        pass_widget=False,
    )

    # ── Play / Pause checkbox button ──────────────────────────────────────
    def toggle_play(value: bool) -> None:
        state["playing"] = value

    pl.add_checkbox_button_widget(
        callback=toggle_play,
        value=False,
        position=(12, 85),
        size=38,
        color_on="green",
        color_off="#cc4444",
    )
    pl.add_text(
        "PLAY", position=(55, 96),
        font_size=9, color="black",
    )

    # ── Timer drives automatic playback ──────────────────────────────────
    # duration = ms between timer events; at 40 ms = 25 fps
    timer_interval_ms = max(16, int(1000 / fps))

    def _advance(*_args) -> None:
        if not state["playing"]:
            return
        t_next = (state["t"] + 1) % T_frames
        state["t"] = t_next
        grid.point_data["Vm (mV)"] = Vm_all[t_next]
        time_label.SetInput(f"t = {t_next * dt_ms:.0f} ms")
        # Sync slider display without re-triggering callback
        slider.GetRepresentation().SetValue(float(t_next))
        pl.render()

    pl.add_timer_event(
        max_steps=9_999_999,
        duration=timer_interval_ms,
        callback=_advance,
    )

    # ── Camera / scene settings ───────────────────────────────────────────
    pl.set_background("white")
    pl.add_axes(line_width=2)
    # Anterior-oblique view, looking from patient's right
    pl.camera.position    = (200.0, 600.0, 200.0)
    pl.camera.focal_point = (0.0,   0.0,   100.0)
    pl.camera.up          = (0.0,   0.0,   1.0)
    pl.reset_camera()

    # Instructions overlay
    pl.add_text(
        "Drag slider to scrub time  |  PLAY button for auto-playback  "
        "|  Left-drag to rotate  |  Scroll to zoom",
        position=(0.02, 0.01),
        font_size=8,
        color="gray",
        viewport=True,
    )

    pl.show()


def _save_activation_mp4(
    grid,
    Vm_all: np.ndarray,
    T_frames: int,
    dt_ms: float,
    vm_clim: list,
    nodes_mm: np.ndarray,
    out_path: Path,
    fps: int,
) -> None:
    """Render all Vm(t) frames off-screen and write to MP4."""
    import pyvista as pv

    print(f"Rendering {T_frames} frames at {fps} fps -> {out_path}", flush=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pl = pv.Plotter(off_screen=True, window_size=(1280, 720))

    grid.point_data["Vm (mV)"] = Vm_all[0]
    pl.add_mesh(
        grid, scalars="Vm (mV)", cmap="RdBu_r",
        clim=vm_clim, smooth_shading=True,
        scalar_bar_args=dict(
            title="Vm (mV)", width=0.08, height=0.55,
            position_x=0.91, position_y=0.25,
            label_font_size=11, title_font_size=12,
        ),
    )
    time_label = pl.add_text("t = 0 ms", position=(0.74, 0.93),
                             font_size=14, color="black", viewport=True)
    pl.add_text(
        "KCL Torso Model  |  Ten Tusscher-Panfilov 2006",
        position=(0.02, 0.93), font_size=9, color="gray", viewport=True,
    )
    pl.set_background("white")
    pl.add_axes(line_width=2)
    pl.camera.position    = (200.0, 600.0, 200.0)
    pl.camera.focal_point = (0.0,   0.0,   100.0)
    pl.camera.up          = (0.0,   0.0,   1.0)
    pl.reset_camera()

    pl.open_movie(str(out_path), framerate=fps, quality=5)

    for t in range(T_frames):
        grid.point_data["Vm (mV)"] = Vm_all[t]
        time_label.SetInput(f"t = {t * dt_ms:.0f} ms")
        pl.render()
        pl.write_frame()
        if t % 50 == 0:
            print(f"  Frame {t}/{T_frames}  ({t/(T_frames-1)*100:.0f}%)",
                  end="\r", flush=True)

    pl.close()
    size_mb = out_path.stat().st_size / 1e6
    print(f"\n  Saved MP4 ({size_mb:.1f} MB) -> {out_path}")


# ==============================================================================
#  CLI
# ==============================================================================

# ==============================================================================
#  5. Self-contained 3-D HTML viewer  (Three.js, rotatable + animated)
# ==============================================================================

def save_activation_3d_html(
    sim_dir: Path = DEFAULT_SIM_DIR,
    result_dir: Path = DEFAULT_RESULT,
    n_frames: int = None,
    target_triangles: int = 80_000,
    fps: int = 20,
) -> Path:
    """
    Build a single self-contained HTML file where:
      - The biventricular wall surface is rendered as a 3-D mesh in WebGL
      - The transmembrane potential Vm(t) is animated as vertex colours
      - The user can freely rotate / zoom / pan the heart while it plays
      - A time slider lets the user scrub to any frame
      - A Play/Pause button drives auto-playback

    Requires only a web browser to open — no Python, no server.

    Output
    ------
    result_dir/activation_3d.html

    Parameters
    ----------
    n_frames         : frames to embed; None (default) = every ms in Vm.pt
    target_triangles : surface mesh target after decimation (default 80 000)
    fps              : playback speed in browser (default 20 fps)
    """
    import base64
    import pyvista as pv
    import torch
    from scipy.spatial import cKDTree

    vm_path = result_dir / "Vm.pt"
    if not vm_path.exists():
        raise FileNotFoundError(
            f"Missing: {vm_path}\n"
            "Run the pipeline first:  python -m torchcor.ecg.kcl_pipeline"
        )

    # ── 1. Build wall mesh and extract surface ────────────────────────────
    nodes_mm, elem_conn, elem_regs, wall_mask = _load_ventricular_mesh(sim_dir)
    grid = _build_pv_grid(nodes_mm, elem_conn, wall_mask)

    print("Extracting surface mesh...", flush=True)
    surface = grid.extract_surface().triangulate()
    print(f"  Raw surface: {surface.n_points:,} nodes, {surface.n_cells:,} triangles")

    # Get original-point-id mapping (surface → 480K global node array)
    orig_ids = None
    if "vtkOriginalPointIds" in surface.point_data.keys():
        orig_ids = surface.point_data["vtkOriginalPointIds"].copy()

    # ── 2. Decimate if needed ─────────────────────────────────────────────
    if surface.n_cells > target_triangles:
        reduction = 1.0 - target_triangles / surface.n_cells
        print(f"  Decimating to ~{target_triangles:,} triangles...", flush=True)
        decimated = surface.decimate(float(np.clip(reduction, 0.01, 0.99)))
        decimated = decimated.triangulate()
        print(f"  After decimation: {decimated.n_points:,} nodes, "
              f"{decimated.n_cells:,} triangles")
    else:
        decimated = surface

    # ── 3. Map decimated nodes back to Vm indices via KDTree ──────────────
    # After decimation, vtkOriginalPointIds is invalid; use nearest-neighbour
    # in 3-D space to find the closest pre-decimation surface point.
    print("  Mapping decimated nodes to Vm indices...", flush=True)
    tree = cKDTree(surface.points)
    _, dec_to_surf = tree.query(decimated.points, workers=-1)  # (N_dec,)
    if orig_ids is not None:
        dec_to_global = orig_ids[dec_to_surf]   # index into 480 K node array
    else:
        # Fallback: KDTree against all 480K nodes
        tree2 = cKDTree(nodes_mm)
        _, dec_to_global = tree2.query(decimated.points, workers=-1)

    # ── 4. Pack surface geometry ──────────────────────────────────────────
    # Two transforms applied together:
    #   (a) 180° rotation around Z  →  (x,y,z) → (−x, −y, z)
    #       makes +X / +Y anatomically correct (matches anatomy viewer)
    #   (b) Z-up → Y-up remap       →  (x,y,z) → (x, z, −y)
    #       keeps Three.js / OrbitControls in their native Y-up world
    # Combined: cardiac (x, y, z) → Three.js (−x, z, +y)
    #   Three.js X = −cardiac X  (left-right, anatomically corrected)
    #   Three.js Y =  cardiac Z  (inferior-superior, vertical)
    #   Three.js Z =  cardiac Y  (anterior-posterior, depth)
    _p = decimated.points.astype(np.float32)
    verts_mm = np.empty_like(_p)
    verts_mm[:, 0] = -_p[:, 0]   # Three.js X = −cardiac X (180° Z flip)
    verts_mm[:, 1] =  _p[:, 2]   # Three.js Y =  cardiac Z (up)
    verts_mm[:, 2] =  _p[:, 1]   # Three.js Z =  cardiac Y (depth, sign flipped by 180°)
    faces_raw = decimated.faces.reshape(-1, 4)[:, 1:]       # (M, 3)
    tris      = faces_raw.astype(np.uint32)                 # (M, 3)
    N_verts   = len(verts_mm)
    N_tris    = len(tris)

    # ── 5. Load Vm and select frames ──────────────────────────────────────
    print("Loading Vm(t)...", flush=True)
    Vm_all  = torch.load(str(vm_path), map_location="cpu").numpy().astype(np.float32)
    T_total = Vm_all.shape[0]

    if n_frames is None or n_frames >= T_total:
        # Full 1 ms resolution — every available snapshot
        frame_idx = np.arange(T_total, dtype=int)
        print(f"  Using all {T_total} frames (1 ms/frame)", flush=True)
    else:
        # Legacy subsampling: dense in QRS, sparse in plateau/repol
        qrs_end   = min(50, T_total)
        n_qrs     = min(30, qrs_end)
        n_tail    = n_frames - n_qrs
        t_qrs     = np.linspace(0, qrs_end - 1, n_qrs,  dtype=int)
        t_tail    = np.linspace(qrs_end, T_total - 1, n_tail, dtype=int)
        frame_idx = np.unique(np.concatenate([t_qrs, t_tail]))[:n_frames]
        print(f"  Subsampled to {len(frame_idx)} frames (of {T_total})", flush=True)

    frame_ms = frame_idx.astype(np.float32)
    N_frames = len(frame_idx)

    # Extract Vm at surface nodes for every sampled frame: (N_frames, N_verts)
    Vm_surf = Vm_all[frame_idx][:, dec_to_global].astype(np.float32)
    del Vm_all
    print(f"  Frames: {N_frames}  |  Surface nodes: {N_verts:,}")

    # ── 6. Base64-encode all binary blobs ─────────────────────────────────
    def b64(arr: np.ndarray) -> str:
        return base64.b64encode(arr.tobytes()).decode()

    verts_b64  = b64(verts_mm)
    tris_b64   = b64(tris)
    vm_b64     = b64(Vm_surf)       # row-major: [frame0_node0, frame0_node1, ...]
    times_b64  = b64(frame_ms)

    size_mb = (len(verts_b64) + len(tris_b64) + len(vm_b64) + len(times_b64)) / 1e6
    print(f"  Embedded data size: {size_mb:.1f} MB (base64)", flush=True)

    # ── 7. Generate and save HTML ─────────────────────────────────────────
    t_max_ms = int(frame_ms[-1])   # last sampled frame time in ms
    html = _build_3d_html(
        verts_b64=verts_b64, tris_b64=tris_b64,
        vm_b64=vm_b64, times_b64=times_b64,
        n_verts=N_verts, n_tris=N_tris, n_frames=N_frames,
        t_max_ms=t_max_ms,
        fps=fps, vm_min=float(VM_CLIM[0]), vm_max=float(VM_CLIM[1]),
    )

    out_path = result_dir / "activation_3d.html"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    total_mb = out_path.stat().st_size / 1e6
    print(f"  Saved 3-D HTML ({total_mb:.1f} MB) -> {out_path}")

    import webbrowser
    webbrowser.open(out_path.as_uri())
    return out_path


# ==============================================================================
#  6. Body Surface Potential Map  —  3-D rotatable animated HTML
# ==============================================================================

def save_bspm_3d_html(
    sim_dir:          Path = DEFAULT_SIM_DIR,
    result_dir:       Path = DEFAULT_RESULT,
    n_frames:         int  = None,
    target_triangles: int  = 150_000,
    fps:              int  = 20,
) -> Path:
    """
    Build a self-contained Three.js HTML viewer for the body surface potential
    map (BSPM).  Requires bspm.pt (from kcl_pipeline --bspm) to be present
    in result_dir.

    The viewer is identical to the activation_3d.html viewer:
      - Rotate / zoom / pan with mouse
      - Time slider (0 → T ms), arrow-key stepping (1 ms), Space = play/pause
      - Axes gizmo in the lower-left corner (Z up)
      - RdBu_r colormap, symmetric about 0 mV

    Saved as  result_dir / bspm_3d.html
    """
    import torch
    from scipy.spatial import cKDTree
    import pandas as pd
    import base64
    import pyvista as pv
    import webbrowser

    bspm_path    = result_dir / "bspm.pt"
    frame_ms_path = result_dir / "bspm_frame_ms.npy"
    torso_dir    = sim_dir / "torso"

    for f in (bspm_path, frame_ms_path, torso_dir):
        if not Path(str(f)).exists():
            raise FileNotFoundError(
                f"Required file missing: {f}\n"
                "Run:  python -m torchcor.ecg.kcl_pipeline "
                "--skip-extract --skip-sim --skip-ecg --bspm"
            )

    # ── 1. Load torso CARP mesh ────────────────────────────────────────
    print("Loading torso CARP mesh…", flush=True)
    pts_file  = sorted(torso_dir.glob("*.pts"))[0]
    elem_file = sorted(torso_dir.glob("*.elem"))[0]

    nodes_raw = pd.read_csv(
        pts_file, sep=r"\s+", header=None, skiprows=1,
        usecols=[0, 1, 2], engine="c", dtype=np.float32,
    ).values
    nodes_mm = nodes_raw / 1000.0     # µm → mm

    raw_elems = pd.read_csv(
        elem_file, sep=r"\s+", header=None, skiprows=1,
        usecols=[1, 2, 3, 4], engine="c", dtype=np.int64,
    ).values
    print(f"  Torso: {len(nodes_mm):,} nodes  {len(raw_elems):,} elements", flush=True)

    # ── 2. Extract and decimate torso surface ──────────────────────────
    print("Extracting torso surface…", flush=True)
    cells     = np.hstack([np.full((len(raw_elems), 1), 4, dtype=np.int64), raw_elems])
    celltypes = np.full(len(raw_elems), 10, dtype=np.uint8)   # VTK_TETRA=10
    grid      = pv.UnstructuredGrid(cells.ravel(), celltypes, nodes_mm.astype(np.float64))
    surface   = grid.extract_surface().triangulate()
    orig_ids  = surface.point_data["vtkOriginalPointIds"].copy()
    print(f"  Raw surface: {surface.n_points:,} nodes, {surface.n_cells:,} triangles",
          flush=True)

    if surface.n_cells > target_triangles:
        reduction = 1.0 - target_triangles / surface.n_cells
        print(f"  Decimating to ~{target_triangles:,} triangles…", flush=True)
        decimated = surface.decimate(reduction).triangulate()
    else:
        decimated = surface
    print(f"  After decimation: {decimated.n_points:,} nodes, "
          f"{decimated.n_cells:,} triangles", flush=True)

    # ── 3. Map decimated surface nodes → bspm column indices ──────────
    print("Mapping decimated nodes to BSPM indices…", flush=True)
    tree = cKDTree(surface.points)
    _, dec_to_surf = tree.query(decimated.points, workers=-1)
    # dec_to_surf[i] = index into orig_ids (= surface node index)
    # bspm[:, j] holds phi for surface node j → bspm[:, dec_to_surf[i]]

    # ── 4. Pack surface geometry ───────────────────────────────────────
    # Same coord transform as activation viewer:
    # cardiac (x,y,z) → Three.js (−x, z, y)  [180° Z-flip + Z-up→Y-up]
    _p = decimated.points.astype(np.float32)
    verts_mm = np.empty_like(_p)
    verts_mm[:, 0] = -_p[:, 0]
    verts_mm[:, 1] =  _p[:, 2]
    verts_mm[:, 2] =  _p[:, 1]

    faces_raw = decimated.faces.reshape(-1, 4)[:, 1:]
    N_verts   = len(verts_mm)
    N_tris    = len(faces_raw)

    verts_b64 = base64.b64encode(verts_mm.astype(np.float32).tobytes()).decode()
    tris_b64  = base64.b64encode(faces_raw.astype(np.uint32).tobytes()).decode()

    # ── 5. Load BSPM and select frames ────────────────────────────────
    print("Loading BSPM…", flush=True)
    bspm_all  = torch.load(str(bspm_path), map_location="cpu").numpy()  # [N_stored × N_surf]
    frame_ms  = np.load(str(frame_ms_path))                              # [N_stored]
    N_stored  = bspm_all.shape[0]

    # Use all stored frames by default (1 ms resolution when step_bspm ran with --bspm-frames 600+)
    n_use     = N_stored if (n_frames is None or n_frames >= N_stored) else n_frames
    sel_idx   = np.arange(n_use)
    bspm_sel  = bspm_all[sel_idx][:, dec_to_surf]  # [n_use × N_dec]
    times_sel = frame_ms[sel_idx].astype(np.float32)
    N_frames_out = len(sel_idx)
    print(f"  Using {N_frames_out} of {N_stored} stored frames", flush=True)

    print(f"  Frames: {N_frames_out}  |  Surface nodes (decimated): {N_verts}", flush=True)

    # ── 6. Symmetric colormap range ───────────────────────────────────
    abs_max = float(np.percentile(np.abs(bspm_sel), 99.5))
    abs_max = max(abs_max, 0.1)   # avoid zero range
    phi_min, phi_max = -abs_max, abs_max
    print(f"  Colormap range: [{phi_min:.3f}, {phi_max:.3f}] mV", flush=True)

    vm_b64    = base64.b64encode(bspm_sel.astype(np.float32).tobytes()).decode()
    times_b64 = base64.b64encode(times_sel.tobytes()).decode()

    size_mb = (len(verts_b64) + len(tris_b64) + len(vm_b64) + len(times_b64)) / 1e6
    print(f"  Embedded data size: {size_mb:.1f} MB (base64)", flush=True)

    # ── 7. Generate HTML ───────────────────────────────────────────────
    t_max_ms = int(times_sel[-1])
    html = _build_3d_html(
        verts_b64=verts_b64, tris_b64=tris_b64,
        vm_b64=vm_b64, times_b64=times_b64,
        n_verts=N_verts, n_tris=N_tris, n_frames=N_frames_out,
        t_max_ms=t_max_ms,
        fps=fps, vm_min=phi_min, vm_max=phi_max,
        cb_label="\u03c6 (mV)",
        page_title="Body Surface Potential Map \u2014 KCL Torso Model",
    )

    out_path = result_dir / "bspm_3d.html"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    total_mb = out_path.stat().st_size / 1e6
    print(f"  Saved BSPM 3-D HTML ({total_mb:.1f} MB) -> {out_path}")

    webbrowser.open(out_path.as_uri())
    return out_path


def _build_3d_html(
    verts_b64: str, tris_b64: str, vm_b64: str, times_b64: str,
    n_verts: int, n_tris: int, n_frames: int, t_max_ms: int,
    fps: int, vm_min: float, vm_max: float,
    cb_label: str = "Vm (mV)",
    page_title: str = "Cardiac Activation \u2014 KCL Torso Model",
) -> str:
    """Return a self-contained Three.js HTML string."""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{page_title}</title>
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ background:#111827; color:#e5e7eb; font-family:Arial,sans-serif; overflow:hidden; }}
#viewport {{ width:100vw; height:calc(100vh - 72px); display:block; }}
#controls {{
  position:fixed; bottom:0; left:0; right:0; height:72px;
  background:rgba(10,10,20,0.88); display:flex; align-items:center;
  padding:0 18px; gap:12px; border-top:1px solid #374151;
}}
#play-btn {{
  background:#10b981; color:#fff; border:none; border-radius:6px;
  padding:7px 16px; font-size:13px; cursor:pointer; min-width:68px;
  transition:background 0.15s;
}}
#play-btn.paused {{ background:#ef4444; }}
#time-slider {{ flex:1; accent-color:#3b82f6; cursor:pointer; height:5px; }}
#time-label {{ font-size:12px; color:#60a5fa; min-width:80px; font-variant-numeric:tabular-nums; }}
#credits {{ font-size:10px; color:#6b7280; white-space:nowrap; }}
#colorbar {{
  position:fixed; right:16px; top:16px;
  background:rgba(10,10,20,0.75); border:1px solid #374151;
  padding:8px 10px; border-radius:8px; font-size:11px; text-align:center; color:#9ca3af;
}}
#cb-canvas {{ display:block; margin:6px auto 2px; border:1px solid #374151; }}
</style>
</head>
<body>
<canvas id="viewport"></canvas>
<div id="controls">
  <button id="play-btn">&#9654; Play</button>
  <span id="time-label">t = 0 ms</span>
  <input type="range" id="time-slider" min="0" max="{t_max_ms}" value="0" step="1">
  <span id="credits">KCL Torso Model &nbsp;&bull;&nbsp; Ten Tusscher-Panfilov 2006 &nbsp;&bull;&nbsp; Drag slider or use &larr;&rarr; keys (1 ms) &nbsp;&bull;&nbsp; Space = play/pause</span>
</div>
<div id="colorbar">
  <div>{cb_label}</div>
  <canvas id="cb-canvas" width="18" height="130"></canvas>
  <div id="cb-max">{vm_max:.0f}</div>
  <div id="cb-min" style="margin-top:2px">{vm_min:.0f}</div>
</div>

<!-- Three.js via importmap (modern ES-module approach) -->
<script type="importmap">
{{
  "imports": {{
    "three": "https://cdn.jsdelivr.net/npm/three@0.162.0/build/three.module.min.js",
    "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.162.0/examples/jsm/"
  }}
}}
</script>

<script type="module">
import * as THREE from 'three';
import {{ OrbitControls }} from 'three/addons/controls/OrbitControls.js';

// ── Decode base64 → typed arrays ────────────────────────────────────────────
function b64decode(b64, TypedArray) {{
  const bin = atob(b64);
  const buf = new ArrayBuffer(bin.length);
  const u8  = new Uint8Array(buf);
  for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
  return new TypedArray(buf);
}}

const N_VERTS  = {n_verts};
const N_TRIS   = {n_tris};
const N_FRAMES = {n_frames};
const FPS      = {fps};
const VM_MIN   = {vm_min};
const VM_MAX   = {vm_max};

const positions = b64decode("{verts_b64}",  Float32Array);  // N_VERTS * 3
const indices   = b64decode("{tris_b64}",   Uint32Array );  // N_TRIS  * 3
const vmData    = b64decode("{vm_b64}",     Float32Array);  // N_FRAMES * N_VERTS
const frameTimes= b64decode("{times_b64}",  Float32Array);  // N_FRAMES

// ── RdBu_r colormap (matches matplotlib) ────────────────────────────────────
// 0→dark-red (resting -90 mV), 0.5→white (mid), 1→dark-blue (plateau +40 mV)
const CMAP_STOPS = [
  [0.000, 0.647, 0.000, 0.149],
  [0.125, 0.839, 0.188, 0.153],
  [0.250, 0.957, 0.647, 0.510],
  [0.375, 0.992, 0.859, 0.780],
  [0.500, 0.969, 0.969, 0.969],
  [0.625, 0.820, 0.898, 0.941],
  [0.750, 0.573, 0.773, 0.871],
  [0.875, 0.263, 0.576, 0.765],
  [1.000, 0.020, 0.188, 0.380],
];

function vmToRGB(vm, out, off) {{
  const t = Math.max(0, Math.min(1, (vm - VM_MIN) / (VM_MAX - VM_MIN)));
  let i = 0;
  while (i < CMAP_STOPS.length - 2 && CMAP_STOPS[i + 1][0] <= t) i++;
  const lo = CMAP_STOPS[i], hi = CMAP_STOPS[i + 1];
  const f  = (t - lo[0]) / (hi[0] - lo[0]);
  out[off    ] = lo[1] + f * (hi[1] - lo[1]);
  out[off + 1] = lo[2] + f * (hi[2] - lo[2]);
  out[off + 2] = lo[3] + f * (hi[3] - lo[3]);
}}

// ── Draw colorbar ────────────────────────────────────────────────────────────
(function () {{
  const cv  = document.getElementById('cb-canvas');
  const ctx = cv.getContext('2d');
  const tmp = new Float32Array(3);
  for (let y = 0; y < cv.height; y++) {{
    const t = 1 - y / (cv.height - 1);
    vmToRGB(VM_MIN + t * (VM_MAX - VM_MIN), tmp, 0);
    ctx.fillStyle = `rgb(${{(tmp[0]*255)|0}},${{(tmp[1]*255)|0}},${{(tmp[2]*255)|0}})`;
    ctx.fillRect(0, y, cv.width, 1);
  }}
}})();

// ── Three.js renderer ────────────────────────────────────────────────────────
const canvas   = document.getElementById('viewport');
const renderer = new THREE.WebGLRenderer({{ canvas, antialias: true }});
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setSize(canvas.clientWidth, canvas.clientHeight);

const scene  = new THREE.Scene();
scene.background = new THREE.Color(0x111827);

const camera = new THREE.PerspectiveCamera(
  45,
  canvas.clientWidth / canvas.clientHeight,
  0.1, 10000
);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.07;

// Lighting: ambient + two rim lights for depth
scene.add(new THREE.AmbientLight(0xffffff, 0.45));
const key = new THREE.DirectionalLight(0xfff5e0, 0.9);
key.position.set(200, 300, 300);
scene.add(key);
const fill = new THREE.DirectionalLight(0xe0f0ff, 0.4);
fill.position.set(-200, -100, 150);
scene.add(fill);

// ── Build geometry ───────────────────────────────────────────────────────────
const geometry = new THREE.BufferGeometry();
geometry.setAttribute('position', new THREE.BufferAttribute(positions.slice(), 3));
geometry.setIndex(new THREE.BufferAttribute(indices.slice(), 1));

const colors = new Float32Array(N_VERTS * 3);
geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
geometry.computeVertexNormals();

const material = new THREE.MeshPhongMaterial({{
  vertexColors: true,
  side: THREE.DoubleSide,
  shininess: 20,
  specular: new THREE.Color(0.15, 0.15, 0.15),
}});
const heartMesh = new THREE.Mesh(geometry, material);
scene.add(heartMesh);

// Auto-fit camera to mesh bounding sphere
geometry.computeBoundingSphere();
const bs = geometry.boundingSphere;
controls.target.copy(bs.center);
// Y-up world (cardiac Z is now Three.js Y after the Python vertex remap).
// View from the front (−Z direction = cardiac anterior) with slight elevation.
camera.position.set(
  bs.center.x,
  bs.center.y + bs.radius * 0.3,
  bs.center.z - bs.radius * 2.5,
);
camera.lookAt(bs.center);
controls.update();

// ── Frame / time update ───────────────────────────────────────────────────────
let currentFrame  = 0;
let currentTimeMs = 0;   // tracks requested time in ms (may differ from snapped frame)

function timeToFrame(tms) {{
  // Binary search — frameTimes is monotonically increasing
  let lo = 0, hi = N_FRAMES - 1;
  while (lo < hi) {{
    const mid = (lo + hi) >> 1;
    if (frameTimes[mid] < tms) lo = mid + 1; else hi = mid;
  }}
  // Prefer whichever neighbour is closer
  if (lo > 0 && Math.abs(frameTimes[lo-1] - tms) < Math.abs(frameTimes[lo] - tms)) lo--;
  return lo;
}}

function _applyFrame(fi) {{
  const base = fi * N_VERTS;
  for (let i = 0; i < N_VERTS; i++) vmToRGB(vmData[base + i], colors, i * 3);
  geometry.attributes.color.needsUpdate = true;
  document.getElementById('time-label').textContent = 't = ' + currentTimeMs.toFixed(0) + ' ms';
  document.getElementById('time-slider').value = currentTimeMs;
}}

function setTime(tms) {{
  currentTimeMs = Math.max(0, Math.min(frameTimes[N_FRAMES - 1], tms));
  currentFrame  = timeToFrame(currentTimeMs);
  _applyFrame(currentFrame);
}}

setTime(0);

// ── Playback controls ─────────────────────────────────────────────────────────
let playing   = false;
let lastStamp = 0;
const frameDur = 1000 / FPS;

const playBtn   = document.getElementById('play-btn');
const slider    = document.getElementById('time-slider');

playBtn.addEventListener('click', () => {{
  playing = !playing;
  playBtn.textContent = playing ? '\u23F8 Pause' : '\u25B6 Play';
  playBtn.classList.toggle('paused', playing);
}});

slider.addEventListener('input', e => {{
  playing = false;
  playBtn.textContent = '\u25B6 Play';
  playBtn.classList.remove('paused');
  setTime(parseFloat(e.target.value));
}});

// ── Keyboard controls: ←/→ = ±1 ms step, Space = play/pause ─────────────────
document.addEventListener('keydown', e => {{
  if (e.key === 'ArrowRight' || e.key === 'ArrowLeft') {{
    playing = false;
    playBtn.textContent = '\u25B6 Play';
    playBtn.classList.remove('paused');
    setTime(currentTimeMs + (e.key === 'ArrowRight' ? 1 : -1));
    e.preventDefault();
  }} else if (e.key === ' ') {{
    playBtn.click();
    e.preventDefault();
  }}
}});

// ── Axes inset scene (lower-left corner, rotates with main camera) ────────────
const axesScene  = new THREE.Scene();
const AXES_PX    = 100;   // inset size in CSS pixels
const AXES_PAD   =  10;   // inset padding from canvas edges

const axesCamera = new THREE.PerspectiveCamera(50, 1, 0.1, 100);
axesCamera.position.set(0, 0, 2.5);
const _axesTmp  = new THREE.Vector3();   // reused each frame — no GC pressure

// Custom axis lines — colours follow cardiac convention (X=red, Y=green, Z=blue).
// Vertex remap (180° Z-flip + Z-up→Y-up):
//   Three.js X = −cardiac X,  Three.js Y = cardiac Z (up),  Three.js Z = cardiac Y
function makeAxLine(dx, dy, dz, hex) {{
  const mat = new THREE.LineBasicMaterial({{ color: hex }});
  const pts = [new THREE.Vector3(0,0,0), new THREE.Vector3(dx,dy,dz)];
  return new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts), mat);
}}
axesScene.add(makeAxLine(-1, 0, 0, '#ff4444'));  // cardiac +X → red   (Three.js −X)
axesScene.add(makeAxLine(0,  1, 0, '#4488ff'));  // cardiac  Z → blue  (Three.js  Y, vertical)
axesScene.add(makeAxLine(0,  0, 1, '#44ff44'));  // cardiac +Y → green (Three.js  Z)

function axisSprite(text, hex) {{
  const cv  = document.createElement('canvas');
  cv.width  = 64; cv.height = 64;
  const ctx = cv.getContext('2d');
  ctx.font         = 'bold 46px Arial';
  ctx.fillStyle    = hex;
  ctx.textAlign    = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText(text, 32, 34);
  const sp = new THREE.Sprite(new THREE.SpriteMaterial({{
    map: new THREE.CanvasTexture(cv), depthTest: false, transparent: true
  }}));
  sp.scale.set(0.45, 0.45, 1);
  sp.renderOrder = 1;
  return sp;
}}

const spX = axisSprite('X', '#ff4444'); spX.position.set(-1.35, 0,    0   );  // cardiac +X tip
const spZ = axisSprite('Z', '#4488ff'); spZ.position.set(0,     1.35, 0   );  // vertical
const spY = axisSprite('Y', '#44ff44'); spY.position.set(0,     0,    1.35);
axesScene.add(spX, spZ, spY);

// ── Render loop ───────────────────────────────────────────────────────────────
function animate(ts) {{
  requestAnimationFrame(animate);
  if (playing && ts - lastStamp >= frameDur) {{
    const next = (currentFrame + 1) % N_FRAMES;
    currentTimeMs = frameTimes[next];   // sync time label to the actual frame time
    currentFrame  = next;
    _applyFrame(currentFrame);
    lastStamp = ts;
  }}
  controls.update();

  const w = canvas.clientWidth, h = canvas.clientHeight;

  // 1. Main scene — full viewport
  renderer.setViewport(0, 0, w, h);
  renderer.setScissorTest(false);
  renderer.render(scene, camera);

  // 2. Axes inset — lower-left AXES_PX × AXES_PX, synchronized to camera rotation
  renderer.setViewport(AXES_PAD, AXES_PAD, AXES_PX, AXES_PX);
  renderer.setScissor(AXES_PAD, AXES_PAD, AXES_PX, AXES_PX);
  renderer.setScissorTest(true);
  renderer.clearDepth();           // keep main-scene colour, reset depth only
  // Position axes camera in the same direction as main camera is from its target,
  // then look at the axesScene origin → gizmo stays perfectly centred regardless
  // of panning or zoom.
  _axesTmp.subVectors(camera.position, controls.target).normalize().multiplyScalar(2.5);
  axesCamera.position.copy(_axesTmp);
  axesCamera.up.copy(camera.up);
  axesCamera.lookAt(0, 0, 0);
  renderer.render(axesScene, axesCamera);
  renderer.setScissorTest(false);
}}
requestAnimationFrame(animate);

// ── Resize ────────────────────────────────────────────────────────────────────
window.addEventListener('resize', () => {{
  const w = canvas.clientWidth, h = canvas.clientHeight;
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  renderer.setSize(w, h);
}});
</script>
</body>
</html>"""


# ==============================================================================
#  CLI
# ==============================================================================

def main() -> None:
    p = argparse.ArgumentParser(
        description="KCL simulation visualisation tool",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # ── View modes ──────────────────────────────────────────────────────────
    p.add_argument("--anatomy",    action="store_true",
                   help="Interactive 3-D anatomy viewer")
    p.add_argument("--ecg",        action="store_true",
                   help="12-lead ECG as Plotly HTML (opens browser)")
    p.add_argument("--activation", action="store_true",
                   help="Static AT/APD map on biventricular mesh")
    p.add_argument("--animate",    action="store_true",
                   help="Animated Vm(t) with time slider + play/pause")
    p.add_argument("--all",        action="store_true",
                   help="All views: anatomy, ecg, activation, animate")

    # ── Export modes ────────────────────────────────────────────────────────
    p.add_argument("--save-ecg",        action="store_true",
                   help="Save ecg_12lead.html (Plotly)")
    p.add_argument("--save-anatomy",    action="store_true",
                   help="Save anatomy.html (vtk.js WebGL)")
    p.add_argument("--save-activation", action="store_true",
                   help="Save activation.html (vtk.js WebGL)")
    p.add_argument("--save-mp4",        action="store_true",
                   help="Save activation_movie.mp4 (H.264)")
    p.add_argument("--save-3d",         action="store_true",
                   help="Save activation_3d.html — rotatable 3-D heart with animated Vm(t)")
    p.add_argument("--save-bspm",       action="store_true",
                   help="Save bspm_3d.html — rotatable torso with animated body surface potential")
    p.add_argument("--save-all",        action="store_true",
                   help="All export formats (ecg, anatomy, activation, mp4, 3d)")
    p.add_argument("--fps",        type=int, default=25,
                   help="Frames per second for MP4 / playback (default: 25)")
    p.add_argument("--3d-frames",  type=int, default=None, dest="frames_3d",
                   help="Number of animation frames to embed in 3-D HTML (default: all)")
    p.add_argument("--3d-tris",    type=int, default=80_000, dest="tris_3d",
                   help="Target triangle count after decimation (default: 80000)")

    # ── Paths ───────────────────────────────────────────────────────────────
    p.add_argument("--vtu",        default=str(DEFAULT_VTU))
    p.add_argument("--sim-dir",    default=str(DEFAULT_SIM_DIR))
    p.add_argument("--result-dir", default=str(DEFAULT_RESULT))
    p.add_argument("--mp4-path",   default=None,
                   help="Custom output path for MP4 (default: results/activation_movie.mp4)")

    args = p.parse_args()

    vtu        = Path(args.vtu)
    sim_dir    = Path(args.sim_dir)
    result_dir = Path(args.result_dir)
    fps        = args.fps
    mp4_path   = Path(args.mp4_path) if args.mp4_path else None

    # Expand --all / --save-all
    do_anatomy    = args.anatomy    or args.all
    do_ecg        = args.ecg        or args.all or args.save_ecg    or args.save_all
    do_activation = args.activation or args.all or args.save_activation or args.save_all
    do_animate    = args.animate    or args.all or args.save_mp4    or args.save_all
    save_ecg      = args.save_ecg        or args.save_all
    save_anatomy  = args.save_anatomy    or args.save_all
    save_act_html = args.save_activation or args.save_all
    save_mp4      = args.save_mp4        or args.save_all
    save_3d       = args.save_3d         or args.save_all
    save_bspm     = args.save_bspm

    if not any([do_anatomy, do_ecg, do_activation, do_animate,
                save_ecg, save_anatomy, save_act_html, save_mp4, save_3d, save_bspm]):
        p.print_help()
        print("\nExamples:")
        print("  --ecg                  Show ECG in browser (HTML)")
        print("  --animate              Interactive wavefront with slider")
        print("  --animate --save-mp4   Render and save MP4 movie")
        print("  --save-3d              Rotatable 3-D heart + animated Vm(t) in browser")
        print("  --save-all             Save all portable formats")
        return

    if do_anatomy:
        show_anatomy(vtu_path=vtu, save_html=save_anatomy, out_dir=result_dir)
    elif save_anatomy:
        _export_anatomy_html(vtu, result_dir)

    if do_ecg or save_ecg:
        show_ecg(result_dir=result_dir, save_html=True)

    if do_activation:
        show_activation(sim_dir=sim_dir, result_dir=result_dir,
                        save_html=save_act_html)
    elif save_act_html:
        show_activation(sim_dir=sim_dir, result_dir=result_dir,
                        save_html=True)

    if do_animate:
        show_activation_animated(
            sim_dir=sim_dir, result_dir=result_dir,
            save_mp4=save_mp4, fps=fps, mp4_path=mp4_path,
        )
    elif save_mp4:
        show_activation_animated(
            sim_dir=sim_dir, result_dir=result_dir,
            save_mp4=True, fps=fps, mp4_path=mp4_path,
        )

    if save_3d:
        save_activation_3d_html(
            sim_dir=sim_dir, result_dir=result_dir,
            n_frames=args.frames_3d,
            target_triangles=args.tris_3d,
            fps=fps,
        )

    if save_bspm:
        save_bspm_3d_html(
            sim_dir=sim_dir, result_dir=result_dir,
            n_frames=args.frames_3d,
            target_triangles=150_000,
            fps=fps,
        )


if __name__ == "__main__":
    main()
