"""kcl_gui.py  —  Pipeline control GUI for the KCL cardiac ECG pipeline.

Launch with:
    python -m torchcor.ecg.kcl_gui

Also launched automatically when kcl_pipeline is run with no CLI arguments.
"""
from __future__ import annotations

import io
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk


# ── Default paths (imported from pipeline so they stay in sync) ───────────────
try:
    from torchcor.ecg.kcl_pipeline import DEFAULT_SIM_DIR, DEFAULT_RESULT
except Exception:
    DEFAULT_SIM_DIR = Path(r"C:\Users\bulli\cardiac_data\kcl_torso\KCL_torso3\sim")
    DEFAULT_RESULT  = DEFAULT_SIM_DIR / "results"


# ── Conductivity presets  (σiL, σiT, σeL, σeT)  S/m ─────────────────────────
CONDUCTIVITY_PRESETS: dict[str, tuple[float, float, float, float]] = {
    "Roth / Potse 1997    (0.300 / 0.030 / 0.300 / 0.120)": (0.300, 0.030, 0.300, 0.120),
    "Patel & Roth 2015    (0.240 / 0.035 / 0.240 / 0.200)": (0.240, 0.035, 0.240, 0.200),
    "Hooks 2007           (0.260 / 0.026 / 0.260 / 0.250)": (0.260, 0.026, 0.260, 0.250),
    "Clerc 1976           (0.174 / 0.019 / 0.625 / 0.236)": (0.174, 0.019, 0.625, 0.236),
    "Stinstra 2005        (0.160 / 0.005 / 0.210 / 0.060)": (0.160, 0.005, 0.210, 0.060),
}

# ── Pipeline step definitions ─────────────────────────────────────────────────
#   primary  : path relative to sim_dir used for mtime staleness check
#   outputs  : all paths relative to sim_dir shown in tooltip / detail view
#   depends_on: keys of upstream steps whose primary output must be ≤ this one
STEP_DEFS: list[dict] = [
    dict(
        key        = "extract",
        label      = "Extract mesh",
        primary    = "ventricles",
        outputs    = ["ventricles", "torso", "heart_to_torso.npy", "electrodes.vtx"],
        depends_on = [],
    ),
    dict(
        key        = "simulate",
        label      = "Simulate",
        primary    = "results/Vm.pt",
        outputs    = ["results/Vm.pt",
                      "results/activation_times.npy",
                      "results/repolarization_times.npy"],
        depends_on = ["extract"],
    ),
    dict(
        key        = "ecg",
        label      = "Compute ECG",
        primary    = "results/ecg_12lead.pt",
        outputs    = ["results/ecg_12lead.pt", "results/ecg_12lead.html",
                      "results/ecg_12lead.png", "results/ecg_12lead.csv"],
        depends_on = ["simulate"],
    ),
    dict(
        key        = "bspm",
        label      = "Compute BSPM",
        primary    = "results/bspm.pt",
        outputs    = ["results/bspm.pt", "results/bspm_frame_ms.npy",
                      "results/torso_surface_nodes.npy"],
        depends_on = ["simulate"],
    ),
    dict(
        key        = "view3d",
        label      = "3D Activation viewer",
        primary    = "results/activation_3d.html",
        outputs    = ["results/activation_3d.html"],
        depends_on = ["simulate"],
    ),
    dict(
        key        = "bspm3d",
        label      = "3D BSPM viewer",
        primary    = "results/bspm_3d.html",
        outputs    = ["results/bspm_3d.html"],
        depends_on = ["bspm"],
    ),
    dict(
        key        = "movie",
        label      = "Activation movie",
        primary    = "results/activation_movie.mp4",
        outputs    = ["results/activation_movie.mp4"],
        depends_on = ["simulate"],
    ),
]

# ── Colours ───────────────────────────────────────────────────────────────────
C_OK    = "#1e7e34"   # dark green
C_STALE = "#c07000"   # amber
C_MISS  = "#b02020"   # red
C_INFO  = "#1a5276"   # blue (log)

PAD = 8


# ── Stdout / stderr redirector ────────────────────────────────────────────────
class _TextRedirector(io.TextIOBase):
    """Write to a tkinter Text widget from any thread."""

    def __init__(self, widget: tk.Text, tag: str = "stdout"):
        self._w   = widget
        self._tag = tag

    def write(self, s: str) -> int:
        self._w.after(0, self._insert, s)
        return len(s)

    def _insert(self, s: str) -> None:
        self._w.configure(state="normal")
        self._w.insert(tk.END, s, self._tag)
        self._w.see(tk.END)
        self._w.configure(state="disabled")

    def flush(self) -> None:
        pass


# ── Main GUI class ────────────────────────────────────────────────────────────
class PipelineGUI:

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        root.title("KCL Cardiac ECG Pipeline")
        root.minsize(980, 760)

        # Threading state
        self._stop_event  = threading.Event()
        self._run_thread: threading.Thread | None = None
        self._orig_stdout = sys.stdout
        self._orig_stderr = sys.stderr

        # ── Tk variables ──────────────────────────────────────────────────────
        self.var_sim_dir    = tk.StringVar(value=str(DEFAULT_SIM_DIR))
        self.var_result_dir = tk.StringVar(value=str(DEFAULT_RESULT))
        self.var_device     = tk.StringVar(value="cuda:0")
        self.var_dtype      = tk.StringVar(value="float32")
        self.var_T          = tk.DoubleVar(value=600.0)
        self.var_dt         = tk.DoubleVar(value=0.05)
        self.var_preset     = tk.StringVar(value=list(CONDUCTIVITY_PRESETS)[0])

        # Per-step variables filled in _build_steps_frame
        self.step_vars: dict[str, dict[str, tk.Variable | tk.Widget]] = {}
        for sd in STEP_DEFS:
            self.step_vars[sd["key"]] = {
                "enabled": tk.BooleanVar(value=True),
                "force":   tk.BooleanVar(value=False),
            }

        self._build_ui()
        root.after(300, self.refresh_status)

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        outer = ttk.Frame(self.root, padding=PAD)
        outer.pack(fill=tk.BOTH, expand=True)

        # Row 0 — directories + source files side by side
        top = ttk.Frame(outer)
        top.pack(fill=tk.X, pady=(0, PAD))
        top.columnconfigure(0, weight=2)
        top.columnconfigure(1, weight=3)
        self._build_dirs_frame(top)
        self._build_sources_frame(top)

        # Row 1 — pipeline steps
        self._build_steps_frame(outer)

        # Row 2 — configuration
        self._build_config_frame(outer)

        # Row 3 — run / cancel
        self._build_run_frame(outer)

        # Row 4 — log (expands)
        self._build_log_frame(outer)

    # ── Directories ───────────────────────────────────────────────────────────

    def _build_dirs_frame(self, parent: tk.Widget) -> None:
        frm = ttk.LabelFrame(parent, text="Directories", padding=PAD)
        frm.grid(row=0, column=0, sticky="nsew", padx=(0, PAD // 2))
        frm.columnconfigure(1, weight=1)

        for row, (lbl, var) in enumerate([
            ("Sim dir:",    self.var_sim_dir),
            ("Result dir:", self.var_result_dir),
        ]):
            ttk.Label(frm, text=lbl).grid(row=row, column=0, sticky="w",
                                           pady=(0 if row == 0 else PAD // 2, 0))
            ttk.Entry(frm, textvariable=var, width=42).grid(
                row=row, column=1, sticky="ew",
                padx=(PAD // 2, PAD // 2),
                pady=(0 if row == 0 else PAD // 2, 0))
            ttk.Button(frm, text="Browse…", width=9,
                       command=lambda v=var: self._browse(v)).grid(
                row=row, column=2,
                pady=(0 if row == 0 else PAD // 2, 0))

        # Refresh status when directories change
        for var in (self.var_sim_dir, self.var_result_dir):
            var.trace_add("write", lambda *_: self.root.after(300, self.refresh_status))

    # ── Source files ──────────────────────────────────────────────────────────

    def _build_sources_frame(self, parent: tk.Widget) -> None:
        frm = ttk.LabelFrame(parent, text="Source / Input Files", padding=PAD)
        frm.grid(row=0, column=1, sticky="nsew", padx=(PAD // 2, 0))
        frm.columnconfigure(0, weight=1)
        frm.rowconfigure(0, weight=1)

        cols = ("name", "modified", "status")
        self.src_tree = ttk.Treeview(frm, columns=cols, show="headings", height=5)
        self.src_tree.heading("name",     text="File")
        self.src_tree.heading("modified", text="Modified")
        self.src_tree.heading("status",   text="")
        self.src_tree.column("name",     width=200, stretch=True)
        self.src_tree.column("modified", width=130, anchor="center")
        self.src_tree.column("status",   width=30,  anchor="center")
        self.src_tree.grid(row=0, column=0, sticky="nsew")

        sb = ttk.Scrollbar(frm, orient="vertical", command=self.src_tree.yview)
        sb.grid(row=0, column=1, sticky="ns")
        self.src_tree.configure(yscrollcommand=sb.set)

    # ── Pipeline steps ────────────────────────────────────────────────────────

    def _build_steps_frame(self, parent: tk.Widget) -> None:
        outer = ttk.LabelFrame(parent, text="Pipeline Steps", padding=PAD)
        outer.pack(fill=tk.X, pady=(0, PAD))

        # Column header
        hdr = ttk.Frame(outer)
        hdr.pack(fill=tk.X)
        for text, width, anchor in [
            ("Run",      4,  "center"),
            ("Step",     24, "w"),
            ("Force",    6,  "center"),
            ("Primary output",   28, "w"),
            ("Modified", 18, "w"),
            ("Status",   10, "w"),
        ]:
            ttk.Label(hdr, text=text, width=width, anchor=anchor,
                      font=("", 9, "bold")).pack(side=tk.LEFT)
        ttk.Separator(outer, orient="horizontal").pack(fill=tk.X, pady=(2, 4))

        for sd in STEP_DEFS:
            key = sd["key"]
            sv  = self.step_vars[key]
            row = ttk.Frame(outer)
            row.pack(fill=tk.X, pady=1)

            ttk.Checkbutton(row, variable=sv["enabled"], width=3).pack(side=tk.LEFT)
            ttk.Label(row, text=sd["label"], width=24, anchor="w").pack(side=tk.LEFT)
            ttk.Checkbutton(row, variable=sv["force"],   width=5).pack(side=tk.LEFT)

            prim_name = Path(sd["primary"]).name
            ttk.Label(row, text=prim_name, width=28, anchor="w",
                      foreground="#555").pack(side=tk.LEFT)

            mod_lbl  = ttk.Label(row, text="—", width=18, anchor="w")
            mod_lbl.pack(side=tk.LEFT)
            sv["mod_lbl"] = mod_lbl

            stat_lbl = ttk.Label(row, text="", width=10, anchor="w",
                                 font=("", 9, "bold"))
            stat_lbl.pack(side=tk.LEFT)
            sv["stat_lbl"] = stat_lbl

        # Footer: legend + refresh button
        foot = ttk.Frame(outer)
        foot.pack(fill=tk.X, pady=(6, 0))
        ttk.Label(foot,
                  text="✅ Current   ⚠ Stale (dependency newer)   ✗ Missing",
                  foreground="#666", font=("", 8)).pack(side=tk.LEFT)
        ttk.Button(foot, text="↻  Refresh", width=11,
                   command=self.refresh_status).pack(side=tk.RIGHT)

    # ── Configuration ─────────────────────────────────────────────────────────

    def _build_config_frame(self, parent: tk.Widget) -> None:
        frm = ttk.LabelFrame(parent, text="Configuration", padding=PAD)
        frm.pack(fill=tk.X, pady=(0, PAD))

        # Row 1 — device / dtype / T / dt
        r1 = ttk.Frame(frm)
        r1.pack(fill=tk.X)

        ttk.Label(r1, text="Device:").pack(side=tk.LEFT)
        ttk.Combobox(r1, textvariable=self.var_device,
                     values=self._detect_devices(),
                     width=10).pack(side=tk.LEFT, padx=(4, 16))

        ttk.Label(r1, text="dtype:").pack(side=tk.LEFT)
        ttk.Combobox(r1, textvariable=self.var_dtype,
                     values=["float32", "float64"],
                     width=10).pack(side=tk.LEFT, padx=(4, 16))

        ttk.Label(r1, text="T (ms):").pack(side=tk.LEFT)
        ttk.Entry(r1, textvariable=self.var_T, width=7).pack(side=tk.LEFT, padx=(4, 16))

        ttk.Label(r1, text="dt (ms):").pack(side=tk.LEFT)
        ttk.Entry(r1, textvariable=self.var_dt, width=7).pack(side=tk.LEFT, padx=(4, 0))

        # Row 2 — conductivity preset
        r2 = ttk.Frame(frm)
        r2.pack(fill=tk.X, pady=(PAD // 2, 0))

        ttk.Label(r2, text="Conductivity preset:").pack(side=tk.LEFT)
        ttk.Combobox(r2, textvariable=self.var_preset,
                     values=list(CONDUCTIVITY_PRESETS),
                     width=52, state="readonly").pack(side=tk.LEFT, padx=(4, 0))

        self._sigma_lbl = ttk.Label(r2, text="", foreground="#555", font=("", 8))
        self._sigma_lbl.pack(side=tk.LEFT, padx=(10, 0))
        self.var_preset.trace_add("write", lambda *_: self._refresh_sigma_lbl())
        self._refresh_sigma_lbl()

    # ── Run / Cancel ──────────────────────────────────────────────────────────

    def _build_run_frame(self, parent: tk.Widget) -> None:
        frm = ttk.Frame(parent)
        frm.pack(fill=tk.X, pady=(0, PAD))

        self._run_btn = ttk.Button(frm, text="▶   Run Selected Steps",
                                   command=self._on_run, width=24)
        self._run_btn.pack(side=tk.LEFT)

        self._cancel_btn = ttk.Button(frm, text="✕  Cancel",
                                      command=self._on_cancel,
                                      width=12, state="disabled")
        self._cancel_btn.pack(side=tk.LEFT, padx=(PAD, 0))

        self._progress = ttk.Progressbar(frm, mode="indeterminate", length=180)
        self._progress.pack(side=tk.LEFT, padx=(PAD * 2, 0))

        self._status_lbl = ttk.Label(frm, text="")
        self._status_lbl.pack(side=tk.LEFT, padx=(PAD, 0))

    # ── Log ───────────────────────────────────────────────────────────────────

    def _build_log_frame(self, parent: tk.Widget) -> None:
        frm = ttk.LabelFrame(parent, text="Log", padding=PAD)
        frm.pack(fill=tk.BOTH, expand=True)

        tb = ttk.Frame(frm)
        tb.pack(fill=tk.X)
        ttk.Button(tb, text="Clear", command=self._clear_log,
                   width=7).pack(side=tk.RIGHT)
        ttk.Button(tb, text="Save…", command=self._save_log,
                   width=7).pack(side=tk.RIGHT, padx=(0, 4))

        self._log_txt = tk.Text(frm, height=14, state="disabled",
                                wrap="word", font=("Consolas", 9),
                                bg="#f7f7f7", relief="flat")
        self._log_txt.pack(fill=tk.BOTH, expand=True, pady=(4, 0))

        sb = ttk.Scrollbar(frm, orient="vertical", command=self._log_txt.yview)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self._log_txt.configure(yscrollcommand=sb.set)

        self._log_txt.tag_configure("stdout", foreground="#111")
        self._log_txt.tag_configure("stderr", foreground="#a00")
        self._log_txt.tag_configure("info",
                                    foreground=C_INFO,
                                    font=("Consolas", 9, "bold"))

    # ── Status refresh ────────────────────────────────────────────────────────

    def refresh_status(self) -> None:
        """Recompute file mtimes and update all status widgets."""
        sim_dir = Path(self.var_sim_dir.get())

        # Source files
        self._refresh_sources(sim_dir)

        # Step mtimes
        mtime: dict[str, float | None] = {}
        for sd in STEP_DEFS:
            p  = sim_dir / sd["primary"]
            mt = None
            if p.exists():
                try:
                    mt = p.stat().st_mtime
                except OSError:
                    pass
            mtime[sd["key"]] = mt

        # Update each step row
        for sd in STEP_DEFS:
            key = sd["key"]
            sv  = self.step_vars[key]
            mt  = mtime[key]

            if mt is None:
                icon  = "✗ Missing"
                color = C_MISS
                mod   = "—"
            else:
                stale = any(
                    mtime.get(dep) is not None and mtime[dep] > mt
                    for dep in sd["depends_on"]
                )
                mod   = datetime.fromtimestamp(mt).strftime("%Y-%m-%d  %H:%M")
                if stale:
                    icon  = "⚠ Stale"
                    color = C_STALE
                else:
                    icon  = "✅ Current"
                    color = C_OK

            sv["stat_lbl"].config(text=icon, foreground=color)
            sv["mod_lbl"].config(text=mod)

        # Auto-select: enable stale / missing steps, disable current ones
        self._auto_select(mtime)

    def _refresh_sources(self, sim_dir: Path) -> None:
        tree = self.src_tree
        tree.delete(*tree.get_children())

        if not sim_dir.exists():
            tree.insert("", tk.END, values=("(directory not found)", "", ""))
            return

        # Collect candidate source files
        found: list[Path] = []
        for pat in ("*.vtu", "*.vtk", "*.elem", "*.pts", "*.lon"):
            found.extend(sim_dir.glob(pat))
        found = sorted(set(found), key=lambda p: p.name)

        if not found:
            tree.insert("", tk.END, values=("(no mesh files found)", "", ""))
            return

        for p in found:
            try:
                mt  = p.stat().st_mtime
                mod = datetime.fromtimestamp(mt).strftime("%Y-%m-%d  %H:%M")
            except OSError:
                mod = "—"
            tree.insert("", tk.END, values=(p.name, mod, "✅"))

    def _auto_select(self, mtime: dict[str, float | None]) -> None:
        """Check steps that are stale or missing; uncheck those that are current."""
        for sd in STEP_DEFS:
            key = sd["key"]
            mt  = mtime[key]
            if mt is None:
                self.step_vars[key]["enabled"].set(True)
            else:
                stale = any(
                    mtime.get(dep) is not None and mtime[dep] > mt
                    for dep in sd["depends_on"]
                )
                self.step_vars[key]["enabled"].set(stale)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _browse(self, var: tk.StringVar) -> None:
        d = filedialog.askdirectory(initialdir=var.get() or ".", title="Select directory")
        if d:
            var.set(d)

    @staticmethod
    def _detect_devices() -> list[str]:
        devs = ["cpu"]
        try:
            import torch
            for i in range(torch.cuda.device_count()):
                devs.append(f"cuda:{i}")
        except Exception:
            pass
        return devs

    def _refresh_sigma_lbl(self) -> None:
        vals = CONDUCTIVITY_PRESETS.get(self.var_preset.get())
        if vals:
            il, it, el, et = vals
            self._sigma_lbl.config(
                text=f"σiL={il:.3f}  σiT={it:.3f}  σeL={el:.3f}  σeT={et:.3f} S/m"
            )

    # ── Run pipeline ──────────────────────────────────────────────────────────

    def _on_run(self) -> None:
        steps = [sd["key"] for sd in STEP_DEFS
                 if self.step_vars[sd["key"]]["enabled"].get()]
        if not steps:
            messagebox.showinfo("Nothing selected",
                                "Tick at least one pipeline step and try again.")
            return

        self._stop_event.clear()
        self._run_btn.config(state="disabled")
        self._cancel_btn.config(state="normal")
        self._progress.start(12)
        self._status_lbl.config(text="Running…", foreground="#333")

        # Redirect prints to the log widget
        sys.stdout = _TextRedirector(self._log_txt, "stdout")
        sys.stderr = _TextRedirector(self._log_txt, "stderr")

        config = dict(
            sim_dir    = self.var_sim_dir.get(),
            result_dir = self.var_result_dir.get(),
            device     = self.var_device.get(),
            dtype      = self.var_dtype.get(),
            T          = float(self.var_T.get()),
            dt         = float(self.var_dt.get()),
            preset     = self.var_preset.get(),
        )
        force = {sd["key"]: bool(self.step_vars[sd["key"]]["force"].get())
                 for sd in STEP_DEFS}

        self._run_thread = threading.Thread(
            target=self._thread_run_pipeline,
            args=(steps, force, config),
            daemon=True,
        )
        self._run_thread.start()

    def _on_cancel(self) -> None:
        self._stop_event.set()
        self._log("\n⚠  Cancel requested — will stop after current step.\n", "stderr")

    def _thread_run_pipeline(self, steps: list[str],
                              force: dict[str, bool],
                              config: dict) -> None:
        """Background thread: execute selected pipeline steps in order."""
        import torch

        sim_dir    = Path(config["sim_dir"])
        result_dir = Path(config["result_dir"])
        device     = torch.device(config["device"])
        dtype      = getattr(torch, config["dtype"])
        T_ms       = config["T"]
        dt_ms      = config["dt"]

        # Apply conductivity preset by patching kcl_pipeline module constants
        preset_vals = CONDUCTIVITY_PRESETS.get(config["preset"])
        if preset_vals:
            try:
                import torchcor.ecg.kcl_pipeline as _pipe
                il, it, el, et = preset_vals
                _pipe.SIGMA_I_LONG                  = il
                _pipe.SIGMA_I_TRANS                 = it
                _pipe.SIGMA_E_LONG                  = el
                _pipe.SIGMA_E_TRANS                 = et
                _pipe.MILLI_VENTRICULAR_ENDO_IL     = il
                _pipe.MILLI_VENTRICULAR_ENDO_IT     = it
                _pipe.MILLI_VENTRICULAR_EPI_EL      = el
                _pipe.MILLI_VENTRICULAR_EPI_ET      = et
            except Exception as exc:
                self._log(f"[warn] Could not apply conductivity preset: {exc}\n", "stderr")

        # Import pipeline functions
        try:
            from torchcor.ecg.kcl_pipeline import (
                step_extract, step_simulate, step_ecg, step_bspm,
            )
            from torchcor.ecg.kcl_visualize import (
                save_activation_3d_html, save_bspm_3d_html,
                show_activation_animated,
            )
        except ImportError as exc:
            self._log(f"Import error: {exc}\n", "stderr")
            self._finish_run(success=False)
            return

        def stopped() -> bool:
            if self._stop_event.is_set():
                self._log("  Pipeline cancelled by user.\n", "stderr")
                return True
            return False

        success = True
        try:
            # ── Extract ───────────────────────────────────────────────────
            if "extract" in steps and not stopped():
                self._log("\n══  Extract mesh  ══\n", "info")
                step_extract(sim_dir, force=force["extract"])

            # ── Simulate ──────────────────────────────────────────────────
            if "simulate" in steps and not stopped():
                self._log("\n══  Simulate  ══\n", "info")
                step_simulate(
                    sim_dir=sim_dir,
                    result_dir=result_dir,
                    device=device,
                    dtype=dtype,
                    T=T_ms,
                    dt=dt_ms,
                    force=force["simulate"],
                )

            # Load Vm once for downstream steps
            Vm = None
            Vm_path = result_dir / "Vm.pt"
            if Vm_path.exists():
                self._log(f"\nLoading Vm from {Vm_path.name}…\n", "info")
                Vm = torch.load(str(Vm_path), map_location=device)

            # ── ECG ───────────────────────────────────────────────────────
            if "ecg" in steps and not stopped():
                self._log("\n══  Compute ECG  ══\n", "info")
                if Vm is None:
                    self._log("  ✗  Vm.pt not found — skipping ECG.\n", "stderr")
                else:
                    step_ecg(sim_dir, result_dir, Vm, device, dtype,
                             force=force["ecg"])

            # ── BSPM ──────────────────────────────────────────────────────
            if "bspm" in steps and not stopped():
                self._log("\n══  Compute BSPM  ══\n", "info")
                step_bspm(sim_dir, result_dir,
                          device=device, dtype=dtype,
                          force=force["bspm"])

            # ── 3D Activation viewer ──────────────────────────────────────
            if "view3d" in steps and not stopped():
                self._log("\n══  3D Activation viewer  ══\n", "info")
                save_activation_3d_html(sim_dir=sim_dir, result_dir=result_dir)

            # ── 3D BSPM viewer ────────────────────────────────────────────
            if "bspm3d" in steps and not stopped():
                self._log("\n══  3D BSPM viewer  ══\n", "info")
                save_bspm_3d_html(sim_dir=sim_dir, result_dir=result_dir)

            # ── Activation movie ──────────────────────────────────────────
            if "movie" in steps and not stopped():
                self._log("\n══  Activation movie  ══\n", "info")
                show_activation_animated(
                    sim_dir=sim_dir, result_dir=result_dir,
                    save_mp4=True,
                )

        except Exception as exc:
            import traceback
            self._log(f"\n✗  Error: {exc}\n{traceback.format_exc()}\n", "stderr")
            success = False

        self._finish_run(success=success and not self._stop_event.is_set())

    def _finish_run(self, success: bool) -> None:
        """Called from background thread — schedules UI cleanup on main thread."""
        self.root.after(0, self._on_run_done, success)

    def _on_run_done(self, success: bool) -> None:
        """UI cleanup — runs on main thread."""
        sys.stdout = self._orig_stdout
        sys.stderr = self._orig_stderr
        self._progress.stop()
        self._run_btn.config(state="normal")
        self._cancel_btn.config(state="disabled")
        if self._stop_event.is_set():
            self._status_lbl.config(text="⚠  Cancelled", foreground=C_STALE)
        elif success:
            self._status_lbl.config(text="✅  Done", foreground=C_OK)
        else:
            self._status_lbl.config(text="✗  Error — see log", foreground=C_MISS)
        self.refresh_status()

    # ── Log helpers ───────────────────────────────────────────────────────────

    def _log(self, msg: str, tag: str = "stdout") -> None:
        """Thread-safe: write a timestamped message to the log widget."""
        ts   = datetime.now().strftime("%H:%M:%S")
        text = f"[{ts}]  {msg}"
        self._log_txt.after(0, self._log_insert, text, tag)

    def _log_insert(self, msg: str, tag: str) -> None:
        self._log_txt.configure(state="normal")
        self._log_txt.insert(tk.END, msg, tag)
        self._log_txt.see(tk.END)
        self._log_txt.configure(state="disabled")

    def _clear_log(self) -> None:
        self._log_txt.configure(state="normal")
        self._log_txt.delete("1.0", tk.END)
        self._log_txt.configure(state="disabled")

    def _save_log(self) -> None:
        path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
            title="Save log as…",
        )
        if path:
            Path(path).write_text(
                self._log_txt.get("1.0", tk.END), encoding="utf-8"
            )


# ── Entry points ──────────────────────────────────────────────────────────────

def launch() -> None:
    """Open the pipeline GUI and block until the window is closed."""
    root = tk.Tk()
    # Use system theme on Windows for native look
    try:
        root.tk.call("source", "azure.tcl")
        root.tk.call("set_theme", "light")
    except Exception:
        pass  # No custom theme available — fine
    PipelineGUI(root)
    root.mainloop()


if __name__ == "__main__":
    launch()
