#!/usr/bin/env python3
"""H-SPAR particle simulator: a step-by-step UI.

    1 Velocity field     load a field from velocity_fields/<name>/raw
    2 Spawn edge         choose the boundary edge(s) particles enter from
    3 Release & settings particles per release, release interval, time step, diffusion
    4 Run & view         run the simulation and play it back

Writes lagrangian_sim/outputs/<field>/<name>.csv (same columns as lagrangian_sim,
so particle_sampling reads it) and <name>.json with every setting used.

    python3 tools/particle_simulator/app.py
"""
from __future__ import annotations

import queue
import subprocess
import sys
import threading
import time
from dataclasses import fields as dc_fields
from pathlib import Path

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import matplotlib
import numpy as np

matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import engine as en  # noqa: E402

STEPS = ["Velocity field", "Spawn edge", "Release & settings", "Run & view"]
EDGE_COLORS = {"shore": "#4d4d4d", "inflow": "#1a9850", "outflow": "#f46d43"}
PICK_PX = 10  # click tolerance in screen pixels
PLAY_SPEEDS = {"1×": 1, "5×": 5, "20×": 20, "60×": 60}  # simulated seconds per second


class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("H-SPAR particle simulator")
        root.geometry("1400x880")
        root.minsize(1100, 700)

        self.step = 0
        self.field: en.Field | None = None
        self.series = None  # snapshot reader for drawing (main thread only)
        self.selected: set[int] = set()
        self.settings = en.Settings()
        self.sim: en.Simulation | None = None
        self.sim_thread: threading.Thread | None = None
        self.frame = 0
        self.playing = False
        self.shown_field_t = None
        self.msgs: queue.Queue = queue.Queue()

        self._build_layout()
        self.panels = [self._panel_field(), self._panel_edges(), self._panel_settings(), self._panel_run()]
        self.show_step(0)
        self.root.after(100, self._poll)

    # ================================================================ layout
    def _build_layout(self):
        style = ttk.Style()
        style.configure("Step.TLabel", padding=(10, 6), foreground="#888")
        style.configure("StepOn.TLabel", padding=(10, 6), foreground="#000", font=("TkDefaultFont", 10, "bold"))
        style.configure("Head.TLabel", font=("TkDefaultFont", 12, "bold"))

        bar = ttk.Frame(self.root, padding=(8, 4))
        bar.pack(side=tk.TOP, fill=tk.X)
        self.step_labels = []
        for i, name in enumerate(STEPS):
            if i:
                ttk.Label(bar, text="→", foreground="#aaa").pack(side=tk.LEFT)
            lab = ttk.Label(bar, text=f"{i + 1}  {name}", style="Step.TLabel")
            lab.pack(side=tk.LEFT)
            self.step_labels.append(lab)
        ttk.Separator(self.root).pack(fill=tk.X)

        foot = ttk.Frame(self.root, padding=8)
        foot.pack(side=tk.BOTTOM, fill=tk.X)
        self.status = tk.StringVar(value="")
        ttk.Label(foot, textvariable=self.status, foreground="#444").pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.next_btn = ttk.Button(foot, text="Next ▶", command=self.go_next)
        self.next_btn.pack(side=tk.RIGHT)
        self.back_btn = ttk.Button(foot, text="◀ Back", command=self.go_back)
        self.back_btn.pack(side=tk.RIGHT, padx=6)
        ttk.Separator(self.root).pack(side=tk.BOTTOM, fill=tk.X)

        body = ttk.Frame(self.root)
        body.pack(fill=tk.BOTH, expand=True)
        self.side = ttk.Frame(body, width=400, padding=10)
        self.side.pack(side=tk.RIGHT, fill=tk.Y)
        self.side.pack_propagate(False)

        plot = ttk.Frame(body)
        plot.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.fig = Figure(figsize=(9, 7))
        self.ax = self.fig.add_axes((0.07, 0.07, 0.82, 0.88))
        self.cax = self.fig.add_axes((0.91, 0.25, 0.015, 0.5))
        self.cax.set_visible(False)
        self._empty_axes("Choose a velocity field and press Load")
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot)
        self.toolbar = NavigationToolbar2Tk(self.canvas, plot, pack_toolbar=False)
        self.toolbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.canvas.mpl_connect("button_press_event", self.on_click)

    def _empty_axes(self, text):
        self.ax.clear()
        self.ax.set_aspect("equal")
        self.ax.set_xlabel("x (m, Gazebo world)")
        self.ax.set_ylabel("y (m, Gazebo world)")
        self.ax.text(0.5, 0.5, text, ha="center", va="center", transform=self.ax.transAxes, color="#999")

    def _panel(self, title, text):
        f = ttk.Frame(self.side)
        ttk.Label(f, text=title, style="Head.TLabel").pack(anchor="w", pady=(0, 6))
        if text:
            ttk.Label(f, text=text, wraplength=370, justify=tk.LEFT).pack(anchor="w", pady=(0, 10))
        return f

    # ---------------------------------------------------------------- step 1
    def _panel_field(self):
        f = self._panel("1. Load the velocity field",
                        "Fields are read from velocity_fields/<name>/raw (Velocity2d.pvd and its .vtu "
                        "snapshots), as written by the flow generator.")
        ttk.Label(f, text="Velocity field").pack(anchor="w")
        names = en.list_fields()
        self.field_var = tk.StringVar(value=names[0] if names else "")
        ttk.Combobox(f, textvariable=self.field_var, values=names, state="readonly").pack(fill=tk.X)
        ttk.Label(f, text="Seconds between snapshots (empty = automatic)").pack(anchor="w", pady=(10, 0))
        self.dt_var = tk.StringVar()
        ttk.Entry(f, textvariable=self.dt_var).pack(fill=tk.X)
        ttk.Button(f, text="Load", command=self.load_field).pack(fill=tk.X, pady=10)
        self.field_desc = ttk.Label(f, wraplength=370, justify=tk.LEFT, foreground="#555")
        self.field_desc.pack(anchor="w")

        ttk.Separator(f).pack(fill=tk.X, pady=10)
        ttk.Label(f, text="Show the field at time (s)").pack(anchor="w")
        self.view_t = tk.DoubleVar(value=0.0)
        self.view_scale = ttk.Scale(f, from_=0.0, to=1.0, variable=self.view_t,
                                    command=lambda _v: self._show_field_time(self.view_t.get()))
        self.view_scale.pack(fill=tk.X)
        self.view_t_label = ttk.Label(f, foreground="#555")
        self.view_t_label.pack(anchor="w")
        self.spin_desc = ttk.Label(f, wraplength=370, justify=tk.LEFT, foreground="#555")
        self.spin_desc.pack(anchor="w", pady=(10, 0))
        return f

    def load_field(self):
        name = self.field_var.get()
        if not name:
            messagebox.showerror("Velocity field", "No velocity fields found under velocity_fields/.")
            return
        try:
            dt = float(self.dt_var.get()) if self.dt_var.get().strip() else None
        except ValueError:
            messagebox.showerror("Snapshot interval", "Enter a number of seconds, or leave it empty.")
            return
        if dt is not None and dt <= 0:
            messagebox.showerror("Snapshot interval", "The interval must be greater than 0.")
            return

        def work():
            try:
                fld = en.Field(name, dt)
                self.msgs.put(("field_ready", fld))
                self.msgs.put(("spin_up", fld.spin_up(fld.new_series())))
            except Exception as exc:  # shown to the user
                self.msgs.put(("error", f"Could not load {name}: {exc}"))

        self.next_btn.state(["disabled"])
        self.status.set(f"Loading {name}…")
        threading.Thread(target=work, daemon=True).start()

    def _field_ready(self, fld: en.Field):
        self.field, self.series = fld, fld.new_series()
        self.spin, self.t_spin = None, 0.0
        self._draw_spin_plot()
        self.crop_desc.config(text="")
        self.selected = {i for i, e in enumerate(fld.edges) if e.kind == "inflow"}
        self.sim = None
        g = fld.geometry
        self.dt_var.set(f"{fld.field_dt:g}")
        self.field_desc.config(text=(
            f"{len(fld.files)} snapshots, {fld.field_dt:g} s apart ({fld.duration:g} s).\n"
            f"Mesh: {len(g.xy)} nodes, {len(g.triangulation.triangles)} triangles.\n"
            f"Boundary: {len(fld.edges)} edges, "
            f"{sum(e.kind == 'inflow' for e in fld.edges)} with inflow, "
            f"{sum(e.kind == 'outflow' for e in fld.edges)} with outflow."))
        self.view_scale.configure(to=fld.duration)
        self.spin_desc.config(text="Checking the spin-up…")

        later, _ = self.series.velocity(g.xy, fld.duration)
        self.vmax = float(np.nanpercentile(np.linalg.norm(later, axis=1), 99)) or 1.0
        self._build_plot()
        self.view_t.set(fld.duration)
        self._show_field_time(fld.duration)
        self.toolbar.update()
        self.next_btn.state(["!disabled"])
        self.status.set(f"Loaded {fld.name}. Press Next to choose the spawn edge.")

    def _spin_up_ready(self, result: en.SpinUp):
        if self.field is None:
            return
        self.spin = result
        t_spin = result.t_spin
        if result.periodic:
            txt = (f"The inflow varies with a period of {result.period_s:g} s. The flow repeats one period later "
                   f"(change below {100 * result.band:g}% of the mean speed {result.level:.2f} m/s) from t = {t_spin:g} s.")
        else:
            txt = (f"Mean speed starts at {result.speed[0]:.2f} m/s and settles at {result.level:.2f} m/s; "
                   f"it stays within ±{100 * result.band:.1f}% of that from t = {t_spin:g} s.")
        if self.field.cropped_from:
            c = self.field.cropped_from
            txt = (f"This field is {c['field']} from t = {c['start_s']:g} s on (spin-up already cropped). "
                   f"Mean speed {result.level:.2f} m/s.")
        elif t_spin > 0:
            txt += (" Earlier snapshots are the spin-up from rest: the particle start time is set after "
                    "it, and step 4 can write a copy of the field without it.")
        self.spin_desc.config(text=txt)
        self.t_spin = t_spin
        self.vars["field_start_s"].set(f"{t_spin:g}")
        self.crop_var.set(f"{t_spin:g}")
        self._draw_spin_plot()

    # ---------------------------------------------------------------- step 2
    def _panel_edges(self):
        f = self._panel("2. Choose where particles enter",
                        "Click an edge on the map or in the list. Shift-click adds or removes edges. "
                        "Edges with water flowing in are green, out orange, shore grey; inflow edges "
                        "are selected to start with.\n\nUse the toolbar under the map to zoom; turn "
                        "the tool off again before clicking edges.")
        row = ttk.Frame(f)
        row.pack(fill=tk.X)
        ttk.Button(row, text="Select inflow edges", command=self.select_inflow).pack(side=tk.LEFT, expand=True, fill=tk.X)
        ttk.Button(row, text="Clear", command=lambda: self._select(set())).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=4)

        cols = ("kind", "length", "flow")
        self.tree = ttk.Treeview(f, columns=cols, show="tree headings", height=16, selectmode="extended")
        self.tree.heading("#0", text="Edge")
        self.tree.column("#0", width=60, anchor="w")
        for c, label, w in (("kind", "Type", 80), ("length", "Length (m)", 90), ("flow", "In (m/s)", 90)):
            self.tree.heading(c, text=label)
            self.tree.column(c, width=w, anchor="e" if c != "kind" else "w")
        self.tree.pack(fill=tk.BOTH, expand=True, pady=8)
        self.tree.bind("<<TreeviewSelect>>", self._tree_selected)
        self.edge_desc = ttk.Label(f, foreground="#555", justify=tk.LEFT, wraplength=370)
        self.edge_desc.pack(anchor="w")
        return f

    def _refresh_tree(self):
        self.tree.delete(*self.tree.get_children())
        flow = self.field.edge_inflow(self.series, self.field.duration)
        for i, e in enumerate(self.field.edges):
            self.tree.insert("", tk.END, iid=str(i), text=str(i + 1),
                             values=(e.kind, f"{e.length:.0f}", f"{flow[i]:+.2f}"))
        self._select(self.selected)

    def _tree_selected(self, _e=None):
        sel = {int(i) for i in self.tree.selection()}
        if sel != self.selected:
            self._select(sel)

    def _select(self, sel):
        self.selected = set(sel)
        if self.step == 1 and {int(i) for i in self.tree.selection()} != self.selected:
            self.tree.selection_set([str(i) for i in sorted(self.selected)])
        length = sum(self.field.edges[i].length for i in self.selected) if self.field else 0.0
        self.edge_desc.config(text=f"{len(self.selected)} edge(s) selected, {length:.0f} m in total.")
        self._draw_edges()

    def select_inflow(self):
        self._select({i for i, e in enumerate(self.field.edges) if e.kind == "inflow"})

    # ---------------------------------------------------------------- step 3
    def _panel_settings(self):
        f = self._panel("3. Release and settings",
                        "Every release puts the given number of particles along the chosen edges. "
                        "Particles are moved with the second-order midpoint method; the velocity is "
                        "interpolated linearly in space and time. The diffusivity adds a random walk.")
        grid = ttk.Frame(f)
        grid.pack(fill=tk.X)
        self.vars: dict[str, tk.StringVar] = {}
        for r, (attr, label, _lo, _strict) in enumerate(en.Settings.FIELDS):
            ttk.Label(grid, text=label).grid(row=r, column=0, sticky="w", pady=2)
            v = tk.StringVar(value=f"{getattr(self.settings, attr):g}")
            ttk.Entry(grid, textvariable=v, width=12).grid(row=r, column=1, sticky="e", pady=2)
            v.trace_add("write", lambda *_: self._settings_changed())
            self.vars[attr] = v
        r = len(en.Settings.FIELDS)
        ttk.Label(grid, text="Placement along the edge").grid(row=r, column=0, sticky="w", pady=2)
        self.placement = tk.StringVar(value="even")
        cb = ttk.Combobox(grid, textvariable=self.placement, values=["even", "random"], state="readonly", width=10)
        cb.grid(row=r, column=1, sticky="e", pady=2)
        cb.bind("<<ComboboxSelected>>", lambda _e: self._settings_changed())
        grid.columnconfigure(0, weight=1)

        ttk.Label(f, text="Output CSV").pack(anchor="w", pady=(12, 0))
        row = ttk.Frame(f)
        row.pack(fill=tk.X)
        self.out_var = tk.StringVar()
        ttk.Entry(row, textvariable=self.out_var).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(row, text="…", width=3, command=self.choose_output).pack(side=tk.LEFT, padx=(4, 0))
        self.settings_desc = ttk.Label(f, foreground="#555", justify=tk.LEFT, wraplength=370)
        self.settings_desc.pack(anchor="w", pady=10)
        return f

    def _read_settings(self, show_errors=True) -> en.Settings | None:
        vals = {}
        for attr, label, lo, strict in en.Settings.FIELDS:
            kind = type(getattr(en.Settings, attr))
            try:
                v = kind(float(self.vars[attr].get()))
            except ValueError:
                if show_errors:
                    messagebox.showerror("Settings", f"'{label}' must be a number.")
                return None
            if (v <= lo) if strict else (v < lo):
                if show_errors:
                    messagebox.showerror("Settings", f"'{label}' must be {'>' if strict else '>='} {lo:g}.")
                return None
            vals[attr] = v
        names = {f.name for f in dc_fields(en.Settings)}
        return en.Settings(**{k: v for k, v in vals.items() if k in names}, placement=self.placement.get())

    def _settings_changed(self):
        if self.step != 2 or self.field is None:
            return
        s = self._read_settings(show_errors=False)
        if s is None:
            self.settings_desc.config(text="Check the values above.")
            self._draw_seed_preview(None)
            return
        length = sum(self.field.edges[i].length for i in self.selected)
        txt = (f"{s.releases} release(s), up to {s.releases * s.per_release} particles in total"
               + (f", {s.per_release / s.interval_s:.2f} particles/s" if s.interval_s > 0 else "")
               + f"; about one particle every {length / max(s.per_release, 1):.1f} m of edge per release."
               f"\n{s.steps} steps of {s.dt_s:g} s; the field is read from t = {s.field_start_s:g} s.")
        end = s.field_start_s + s.duration_s
        if end > self.field.duration + 1e-9:
            txt += (f"\n⚠ The field ends at {self.field.duration:g} s; after particle time "
                    f"{max(self.field.duration - s.field_start_s, 0):g} s its last snapshot is held.")
        if s.field_start_s < getattr(self, "t_spin", 0.0):
            txt += f"\n⚠ The field is still spinning up before t = {self.t_spin:g} s."
        self.settings_desc.config(text=txt)
        self._draw_seed_preview(s)

    def choose_output(self):
        p = Path(self.out_var.get() or self._default_output())
        path = filedialog.asksaveasfilename(title="Output CSV", initialdir=p.parent, initialfile=p.name,
                                            defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if path:
            self.out_var.set(path)

    def _default_output(self) -> str:
        return str(en.OUTPUT_DIR / self.field.name / f"{self.field.name}_particles.csv")

    # ---------------------------------------------------------------- step 4
    def _panel_run(self):
        f = self._panel("4. Run and view", "Particles are coloured by release. Run writes the CSV and a "
                                           ".json with the settings next to it.")
        row = ttk.Frame(f)
        row.pack(fill=tk.X)
        self.run_btn = ttk.Button(row, text="Run", command=self.run_sim)
        self.run_btn.pack(side=tk.LEFT, expand=True, fill=tk.X)
        self.stop_btn = ttk.Button(row, text="Stop", command=self.stop_sim)
        self.stop_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=4)
        self.progress = ttk.Progressbar(f, maximum=1.0)
        self.progress.pack(fill=tk.X, pady=8)
        self.run_desc = ttk.Label(f, foreground="#555", justify=tk.LEFT, wraplength=370)
        self.run_desc.pack(anchor="w")

        ttk.Separator(f).pack(fill=tk.X, pady=10)
        ttk.Label(f, text="Playback").pack(anchor="w")
        row = ttk.Frame(f)
        row.pack(fill=tk.X)
        self.play_btn = ttk.Button(row, text="▶ Play", command=self.toggle_play)
        self.play_btn.pack(side=tk.LEFT)
        self.speed_var = tk.StringVar(value="20×")
        ttk.Combobox(row, textvariable=self.speed_var, values=list(PLAY_SPEEDS), state="readonly",
                     width=5).pack(side=tk.LEFT, padx=6)
        self.follow = tk.BooleanVar(value=True)
        ttk.Checkbutton(row, text="Follow the run", variable=self.follow).pack(side=tk.LEFT)
        self.frame_var = tk.DoubleVar(value=0)
        self.frame_scale = ttk.Scale(f, from_=0, to=0, variable=self.frame_var,
                                     command=lambda _v: self._scrub())
        self.frame_scale.pack(fill=tk.X, pady=6)
        self.show_speed = tk.BooleanVar(value=True)
        ttk.Checkbutton(f, text="Show flow speed under the particles", variable=self.show_speed,
                        command=lambda: self._show_frame(self.frame, force=True)).pack(anchor="w")
        self.frame_desc = ttk.Label(f, foreground="#555", justify=tk.LEFT)
        self.frame_desc.pack(anchor="w", pady=6)
        ttk.Button(f, text="Open output folder", command=self.open_output).pack(fill=tk.X, pady=(10, 0))

        ttk.Separator(f).pack(fill=tk.X, pady=10)
        ttk.Label(f, text="Crop the spin-up from the field").pack(anchor="w")
        ttk.Label(f, wraplength=370, justify=tk.LEFT, foreground="#555", text=(
            "Writes a new field that starts at the chosen time, renumbered from 0, so the particle "
            "simulator and the Gazebo current both start after the spin-up. Snapshots are hard links "
            "(no extra disk space).")).pack(anchor="w")
        self.spin_fig = Figure(figsize=(3.8, 1.5), dpi=90)
        self.spin_ax = self.spin_fig.add_axes((0.16, 0.30, 0.80, 0.62))
        self.spin_canvas = FigureCanvasTkAgg(self.spin_fig, master=f)
        self.spin_canvas.get_tk_widget().pack(fill=tk.X, pady=4)
        row = ttk.Frame(f)
        row.pack(fill=tk.X)
        ttk.Label(row, text="Start the new field at t (s)").pack(side=tk.LEFT)
        self.crop_var = tk.StringVar(value="0")
        self.crop_var.trace_add("write", lambda *_: self._draw_spin_plot())
        ttk.Entry(row, textvariable=self.crop_var, width=8).pack(side=tk.RIGHT)
        ttk.Button(f, text="Write cropped field", command=self.crop_field).pack(fill=tk.X, pady=6)
        self.crop_desc = ttk.Label(f, wraplength=370, justify=tk.LEFT, foreground="#555")
        self.crop_desc.pack(anchor="w")
        return f

    def _draw_spin_plot(self):
        ax = self.spin_ax
        ax.clear()
        sp = getattr(self, "spin", None)
        if sp is None or self.field is None:
            ax.text(0.5, 0.5, "spin-up not computed yet", ha="center", va="center",
                    transform=ax.transAxes, fontsize=8, color="#999")
            ax.set_axis_off()
        else:
            ax.set_axis_on()
            if sp.periodic:
                ax.semilogy(sp.mismatch_t, 100 * np.maximum(sp.mismatch, 1e-4), color="#1f6fae", lw=1)
                ax.axhline(100 * sp.band, color="#1f6fae", ls="--", lw=0.8)
                ax.set_ylabel(f"change over\n{sp.period_s:g} s (%)", fontsize=7)
            else:
                ax.plot(sp.t, sp.speed, color="#1f6fae", lw=1)
                ax.axhspan(sp.level * (1 - sp.band), sp.level * (1 + sp.band), color="#1f6fae", alpha=0.12, lw=0)
                ax.set_ylabel("mean speed\n(m/s)", fontsize=7)
            try:
                tc = float(self.crop_var.get())
                ax.axvline(tc, color="#c0392b", lw=1)
                ax.axvspan(0, tc, color="#999", alpha=0.15, lw=0)
            except ValueError:
                pass
            ax.set_xlim(0, self.field.duration)
            ax.set_xlabel("field time (s)", fontsize=7)
            ax.tick_params(labelsize=7)
        self.spin_canvas.draw_idle()

    def crop_field(self):
        if self.field is None:
            return
        try:
            tc = float(self.crop_var.get())
        except ValueError:
            messagebox.showerror("Crop", "Enter the start time in seconds.")
            return
        name = f"{self.field.name}_from{round(tc / self.field.field_dt) * self.field.field_dt:g}s"
        if not messagebox.askyesno("Crop the spin-up",
                                   f"Write velocity_fields/{name} with the snapshots from t = {tc:g} s on?"):
            return
        try:
            out = self.field.crop(tc, name)
        except Exception as exc:  # shown to the user
            messagebox.showerror("Crop", str(exc))
            return
        n = len(list((out / "raw" / "Velocity2d").glob("*.vtu")))
        self.crop_desc.config(text=f"Wrote {out.name}: {n} snapshots, t = 0 is t = {tc:g} s of {self.field.name}.\n"
                                   "Load it in step 1 for particles, and point vtu_converter at its "
                                   "raw/Velocity2d for the Gazebo current.")
        self.status.set(f"Wrote {out}")

    def run_sim(self):
        if self.sim_thread and self.sim_thread.is_alive():
            return
        out = Path(self.out_var.get()).expanduser()
        if out.exists() and not messagebox.askyesno("Output exists", f"{out} exists. Replace it?"):
            return
        self.sim = en.Simulation(self.field, sorted(self.selected), self.settings, out)
        self.frame, self.playing = 0, False
        self.play_btn.config(text="▶ Play")
        self.sim_reported = False
        self.sim_thread = threading.Thread(target=self.sim.run, daemon=True)
        self.sim_thread.start()
        self._update_run_buttons()
        self.status.set("Running…")

    def stop_sim(self):
        if self.sim:
            self.sim.stop.set()

    def _sim_done(self):
        s, sim = self.settings, self.sim
        if sim.error:
            messagebox.showerror("Simulation", sim.error)
            self.status.set("The simulation failed.")
        else:
            last = sim.snapshots[-1] if sim.snapshots else None
            stopped = sim.stop.is_set()
            self.status.set(("Stopped" if stopped else "Done") + f": wrote {sim.csv_path}")
            if last:
                self.run_desc.config(text=f"{'Stopped at' if stopped else 'Finished'} t = {last.time:g} s of "
                                          f"{s.duration_s:g} s.\n{last.released} released, {last.exited} left "
                                          f"the domain, {len(last.xy)} still in the water.\n"
                                          f"CSV: {sim.csv_path}\nSettings: {sim.settings_path().name}")
        self._update_run_buttons()

    def toggle_play(self):
        if not self.sim or not self.sim.snapshots:
            return
        self.playing = not self.playing
        if self.playing and self.frame >= len(self.sim.snapshots) - 1:
            self.frame = 0
        self.play_pos, self.play_tick = float(self.frame), time.monotonic()
        self.follow.set(False)
        self.play_btn.config(text="⏸ Pause" if self.playing else "▶ Play")

    def _scrub(self):
        if self.sim and self.sim.snapshots:
            self.follow.set(False)
            self._show_frame(int(round(self.frame_var.get())))

    def open_output(self):
        folder = Path(self.out_var.get()).expanduser().parent
        folder.mkdir(parents=True, exist_ok=True)
        subprocess.Popen(["xdg-open", str(folder)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def _update_run_buttons(self):
        running = bool(self.sim_thread and self.sim_thread.is_alive())
        self.run_btn.state(["disabled"] if running else ["!disabled"])
        self.stop_btn.state(["!disabled"] if running else ["disabled"])
        self.back_btn.state(["disabled"] if running or self.step == 0 else ["!disabled"])

    # ================================================================ drawing
    def _build_plot(self):
        fld, g = self.field, self.field.geometry
        self.ax.clear()
        self.cax.clear()
        self.cax.set_visible(True)
        # flat shading (one colour per triangle) draws about a third faster than gouraud
        self.speed_art = self.ax.tripcolor(g.triangulation, facecolors=np.zeros(len(g.triangulation.triangles)),
                                           cmap="Blues", vmin=0.0, vmax=self.vmax, zorder=0)
        self.fig.colorbar(self.speed_art, cax=self.cax, label="flow speed (m/s)")
        segs, colors, widths, self.seg_edge = [], [], [], []
        for i, e in enumerate(fld.edges):
            for a, b in zip(e.points[:-1], e.points[1:]):
                segs.append([a, b])
                colors.append(EDGE_COLORS[e.kind])
                widths.append(1.5 if e.kind == "shore" else 3.0)
                self.seg_edge.append(i)
        self.seg_xy = np.array(segs)
        self.seg_edge = np.array(self.seg_edge)
        self.ax.add_collection(LineCollection(segs, colors=colors, linewidths=widths, zorder=3))
        self.sel_art = LineCollection([], colors="#ffd400", linewidths=8, alpha=0.9, capstyle="round", zorder=2)
        self.ax.add_collection(self.sel_art)
        self.edge_labels = []
        for i, e in enumerate(fld.edges):
            m = e.points[len(e.points) // 2]
            self.edge_labels.append(self.ax.text(
                m[0], m[1], str(i + 1), fontsize=7, ha="center", va="center", zorder=6, visible=False,
                color=EDGE_COLORS[e.kind], bbox=dict(boxstyle="round,pad=0.15", fc="white",
                                                     ec=EDGE_COLORS[e.kind], lw=0.6)))
        self.seed_art = self.ax.scatter([], [], s=6, c="#111", zorder=4)
        self.part_art = self.ax.scatter([], [], s=5, c=[], cmap="plasma", vmin=0, vmax=1, zorder=5, linewidths=0)
        self.time_txt = self.ax.text(0.01, 0.99, "", transform=self.ax.transAxes, va="top", fontsize=10,
                                     bbox=dict(boxstyle="round", fc="white", ec="#ccc", alpha=0.85), zorder=7)
        pad = 0.03 * np.ptp(g.xy, axis=0)
        self.ax.set_xlim(g.xy[:, 0].min() - pad[0], g.xy[:, 0].max() + pad[0])
        self.ax.set_ylim(g.xy[:, 1].min() - pad[1], g.xy[:, 1].max() + pad[1])
        self.ax.set_aspect("equal")
        self.ax.set_xlabel("x (m, Gazebo world)")
        self.ax.set_ylabel("y (m, Gazebo world)")
        self.ax.set_title(fld.name, fontsize=10)
        self.shown_field_t = None
        self._draw_edges()

    def _show_field_time(self, t, label=True):
        """Colour the mesh by flow speed at field time t (re-reads only when t moves)."""
        if self.field is None:
            return
        t = min(max(float(t), 0.0), self.field.duration)
        if label:
            self.view_t_label.config(text=f"t = {t:.0f} s of {self.field.duration:g} s")
        if self.shown_field_t is None or abs(t - self.shown_field_t) >= 0.5 * self.field.field_dt:
            v, _ = self.series.velocity(self.field.geometry.xy, t)
            speed = np.nan_to_num(np.linalg.norm(v, axis=1))
            self.speed_art.set_array(speed[self.field.geometry.triangulation.triangles].mean(axis=1))
            self.shown_field_t = t
        self.speed_art.set_visible(self.step != 3 or self.show_speed.get())
        self.canvas.draw_idle()

    def _draw_edges(self):
        if self.field is None:
            return
        mask = np.isin(self.seg_edge, list(self.selected))
        self.sel_art.set_segments(self.seg_xy[mask])
        for lab in self.edge_labels:
            lab.set_visible(self.step == 1)
        self.canvas.draw_idle()

    def _draw_seed_preview(self, s):
        if self.field is None:
            return
        pts = np.zeros((0, 2))
        if s is not None and self.step == 2 and self.selected:
            pts = en.seed_on_edges(self.field, sorted(self.selected), s.per_release, s.placement,
                                   s.offset_m, np.random.default_rng(s.seed))
        self.seed_art.set_offsets(pts)
        self.canvas.draw_idle()

    def _show_frame(self, i, force=False):
        snaps = self.sim.snapshots if self.sim else []
        if not snaps:
            return
        i = int(min(max(i, 0), len(snaps) - 1))
        if i == self.frame and not force and self.part_art.get_offsets().shape[0]:
            return
        self.frame = i
        sn = snaps[i]
        n_rel = max(self.settings.releases - 1, 1)
        self.part_art.set_offsets(sn.xy)
        self.part_art.set_array(sn.generation / n_rel)
        self.frame_scale.configure(to=max(len(snaps) - 1, 0))
        self.frame_var.set(i)
        self.time_txt.set_text(f"t = {sn.time:.0f} s   (field t = {self.settings.field_start_s + sn.time:.0f} s)")
        self.frame_desc.config(text=f"{len(sn.xy)} particles in the water, {sn.released} released, "
                                    f"{sn.exited} left the domain.")
        if self.show_speed.get():
            self._show_field_time(self.settings.field_start_s + sn.time, label=False)
        else:
            self.speed_art.set_visible(False)
            self.canvas.draw_idle()

    # ================================================================ events
    def _poll(self):
        try:
            while True:
                kind, data = self.msgs.get_nowait()
                if kind == "field_ready":
                    self._field_ready(data)
                elif kind == "spin_up":
                    self._spin_up_ready(data)
                elif kind == "error":
                    self.status.set("")
                    self.next_btn.state(["!disabled"])
                    messagebox.showerror("Error", data)
        except queue.Empty:
            pass

        if self.sim is not None and self.step == 3:
            running = bool(self.sim_thread and self.sim_thread.is_alive())
            snaps = self.sim.snapshots
            if running:
                self.progress["value"] = self.sim.progress
                if snaps:
                    sn = snaps[-1]
                    self.run_desc.config(text=f"t = {sn.time:.0f} of {self.settings.duration_s:g} s, "
                                              f"{len(sn.xy)} particles in the water.")
            elif self.sim_thread is not None and not self.sim_reported:
                self.sim_reported = True
                self.progress["value"] = self.sim.progress
                self._sim_done()
            if self.playing and snaps:
                # advance by wall-clock time so the chosen speed holds even when drawing is slow
                now = time.monotonic()
                stride_s = self.sim.view_stride * self.settings.dt_s
                self.play_pos += PLAY_SPEEDS[self.speed_var.get()] * (now - self.play_tick) / stride_s
                self.play_tick = now
                nxt = int(self.play_pos)
                if nxt >= len(snaps):
                    nxt = len(snaps) - 1
                    if not running:
                        self.playing = False
                        self.play_btn.config(text="▶ Play")
                self._show_frame(nxt)
            elif self.follow.get() and snaps:
                self._show_frame(len(snaps) - 1)
        self.root.after(100, self._poll)

    def on_click(self, ev):
        if (self.step != 1 or self.field is None or ev.inaxes is not self.ax or self.toolbar.mode
                or ev.button != 1):
            return
        i = self._nearest_edge(ev.x, ev.y)
        if i is None:
            return
        self._select(self.selected ^ {i} if ev.key in ("shift", "control", "ctrl+shift") else {i})

    def _nearest_edge(self, px, py):
        a = self.ax.transData.transform(self.seg_xy[:, 0])
        b = self.ax.transData.transform(self.seg_xy[:, 1])
        p = np.array([px, py])
        ab = b - a
        t = np.clip(np.einsum("ij,ij->i", p - a, ab) / np.maximum(np.einsum("ij,ij->i", ab, ab), 1e-12), 0, 1)
        d = np.linalg.norm(a + t[:, None] * ab - p, axis=1)
        k = int(np.argmin(d))
        return int(self.seg_edge[k]) if d[k] < PICK_PX else None

    # ================================================================ steps
    def show_step(self, i):
        self.step = i
        for k, lab in enumerate(self.step_labels):
            lab.configure(style="StepOn.TLabel" if k == i else "Step.TLabel")
        for p in self.panels:
            p.pack_forget()
        self.panels[i].pack(fill=tk.BOTH, expand=True)
        self.next_btn.state(["disabled"] if i == len(STEPS) - 1 or self.field is None else ["!disabled"])
        self._update_run_buttons()
        if self.field is not None:
            self.part_art.set_visible(i == 3)
            self.time_txt.set_visible(i == 3)
            self._draw_seed_preview(None)
            if i == 0:
                self._show_field_time(self.view_t.get())
        if i == 1:
            self._refresh_tree()
        if i == 2:
            if not self.out_var.get():
                self.out_var.set(self._default_output())
            self._settings_changed()
        if i == 3:
            self.run_desc.config(text=f"{self.settings.releases} release(s) of {self.settings.per_release} "
                                      f"particles from {len(self.selected)} edge(s), "
                                      f"{self.settings.duration_s:g} s.\nOutput: {self.out_var.get()}")
            self._show_frame(self.frame, force=True)
        self.status.set({0: "Load a field, then press Next.",
                         1: "Click an edge to choose where particles enter.",
                         2: "The black dots show one release.",
                         3: "Press Run."}[i])
        self._draw_edges()

    def go_back(self):
        if self.sim_thread and self.sim_thread.is_alive():
            messagebox.showinfo("Busy", "Wait for the run to finish, or press Stop.")
            return
        if self.step > 0:
            self.show_step(self.step - 1)

    def go_next(self):
        s = self.step
        if s == 0:
            if self.field is None or self.field.name != self.field_var.get():
                messagebox.showinfo("Velocity field", "Press Load first.")
                return
            self.show_step(1)
        elif s == 1:
            if not self.selected:
                messagebox.showinfo("Spawn edge", "Choose at least one edge.")
                return
            self.show_step(2)
        elif s == 2:
            settings = self._read_settings()
            if settings is None:
                return
            if not self.out_var.get().strip().endswith(".csv"):
                messagebox.showerror("Output CSV", "The output must be a .csv file.")
                return
            if not len(en.seed_on_edges(self.field, sorted(self.selected), 1, settings.placement,
                                        settings.offset_m, np.random.default_rng(0))):
                messagebox.showerror("Spawn edge", "No start point lands inside the water. Reduce the distance.")
                return
            self.settings = settings
            self.show_step(3)


def main():
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
