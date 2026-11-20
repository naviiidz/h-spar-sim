#!/usr/bin/env python3
"""H-SPAR flow generator: a step-by-step UI.

    1 Map           pick the Gazebo world and an output name
    2 Draw region   click the water region's outline on the map
    3 Boundaries    say what happens on every edge (shore, inflow, outflow, open)
    4 Depth         water depth and simulation settings
    5 Mesh & run    build the gmsh mesh, then run Thetis/Firedrake

Start it with launch.sh (sets up the Firedrake environment), or:
    FIREDRAKE_PYTHON=/path/to/venv-firedrake/bin/python python3 app.py
"""
from __future__ import annotations

import os
import queue
import re
import subprocess
import sys
import threading
from pathlib import Path

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText

import matplotlib
import numpy as np

matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import maps  # noqa: E402
import region as rg  # noqa: E402
from build_mesh import read_msh_triangles  # noqa: E402

STEPS = ["Map", "Draw region", "Boundaries", "Depth & settings", "Mesh & run"]
EDGE_COLORS = {"wall": "#4d4d4d", "inflow": "#1a9850", "inflow_open": "#66bd63", "outflow": "#f46d43", "open": "#2166ac"}
PICK_PX = 10  # click tolerance in screen pixels
SECONDS_PER_CELL_STEP = 1.25e-3  # measured: Sydney, 3640 triangles, 2 s step: ~4.5 s per step, one core


def firedrake_env():
    """Python and environment used for the Firedrake run."""
    python = os.environ.get("FIREDRAKE_PYTHON", sys.executable)
    env = dict(os.environ, PYTHONUNBUFFERED="1", OMP_NUM_THREADS="1")
    petsc_dir, petsc_arch = env.get("PETSC_DIR"), env.get("PETSC_ARCH")
    if petsc_dir and petsc_arch:
        lib = str(Path(petsc_dir) / petsc_arch / "lib")
        env["LD_LIBRARY_PATH"] = lib + (":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
    return python, env


def free_output_name(base: str) -> str:
    """base, base_2, base_3 ... whichever has no velocity fields yet."""
    name, k = base, 2
    while (rg.REPO_ROOT / "velocity_fields" / name / "raw").exists():
        name, k = f"{base}_{k}", k + 1
    return name


class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("H-SPAR flow generator")
        root.geometry("1400x880")
        root.minsize(1100, 700)

        self.step = 0
        self.map_info: maps.MapInfo | None = None
        self.loaded_map: str | None = None
        self.poly: list[list[float]] = []
        self.closed = False
        self.edges: list[rg.EdgeBC] = []
        self.selected: set[int] = set()
        self.sim = rg.SimSettings()
        self.mesh_xy = self.mesh_tris = None
        self.mesh_ready = False
        self.overlay = []
        self.proc: subprocess.Popen | None = None
        self.msgs: queue.Queue = queue.Queue()

        self._build_layout()
        self.panels = [self._panel_map(), self._panel_draw(), self._panel_bcs(),
                       self._panel_settings(), self._panel_run()]
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
        self.ax = self.fig.add_axes((0.07, 0.07, 0.9, 0.9))
        self.ax.set_aspect("equal")
        self.ax.set_xlabel("x (m, Gazebo world)")
        self.ax.set_ylabel("y (m, Gazebo world)")
        self.ax.text(0.5, 0.5, "Choose a map and press Next", ha="center", va="center",
                     transform=self.ax.transAxes, color="#999")
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot)
        self.toolbar = NavigationToolbar2Tk(self.canvas, plot, pack_toolbar=False)
        self.toolbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.canvas.mpl_connect("button_press_event", self.on_click)
        self.canvas.mpl_connect("motion_notify_event", self.on_move)
        (self.rubber,) = self.ax.plot([], [], "--", color="#d62728", lw=1)

    def _panel(self, title, text):
        f = ttk.Frame(self.side)
        ttk.Label(f, text=title, style="Head.TLabel").pack(anchor="w", pady=(0, 6))
        if text:
            ttk.Label(f, text=text, wraplength=370, justify=tk.LEFT).pack(anchor="w", pady=(0, 10))
        return f

    # ---------------------------------------------------------------- step 1
    def _panel_map(self):
        f = self._panel("1. Choose the map",
                        "The map's shoreline is shown as the background, in Gazebo world "
                        "coordinates. A map is downloaded from Gazebo Fuel only if it is not "
                        "already in your local Fuel cache.")
        ttk.Label(f, text="Map").pack(anchor="w")
        self.map_var = tk.StringVar(value="sydney_regatta")
        cb = ttk.Combobox(f, textvariable=self.map_var, values=list(maps.MAPS), state="readonly")
        cb.pack(fill=tk.X)
        cb.bind("<<ComboboxSelected>>", lambda _e: self._on_map_change())
        self.map_desc = ttk.Label(f, wraplength=370, justify=tk.LEFT, foreground="#555")
        self.map_desc.pack(anchor="w", pady=(4, 12))

        ttk.Label(f, text="Output name").pack(anchor="w")
        self.name_var = tk.StringVar()
        ttk.Entry(f, textvariable=self.name_var).pack(fill=tk.X)
        self.name_hint = ttk.Label(f, wraplength=370, justify=tk.LEFT, foreground="#555")
        self.name_hint.pack(anchor="w", pady=(4, 12))
        self.name_var.trace_add("write", lambda *_: self._update_name_hint())

        ttk.Separator(f).pack(fill=tk.X, pady=8)
        ttk.Button(f, text="Load a saved region (region.json)…", command=self.load_region).pack(fill=tk.X)
        self._on_map_change()
        return f

    def _on_map_change(self):
        info = maps.MAPS[self.map_var.get()]
        cached = maps.find_model_dir(info)
        self.map_desc.config(text=info.description + ("\n\nIn local cache." if cached else
                                                       "\n\nNot cached yet: will be downloaded."))
        self.name_var.set(free_output_name(info.name))

    def _update_name_hint(self):
        name = self.name_var.get().strip()
        if not rg.NAME_RE.match(name or ""):
            txt = "Use letters, digits, '_' or '-'."
        else:
            txt = f"Mesh → meshes/{name}/\nVelocity fields → velocity_fields/{name}/raw/"
            fields = rg.REPO_ROOT / "velocity_fields" / name
            if (fields / "raw").exists():
                txt += ("\n⚠ Fields made by this tool already exist and will be replaced."
                        if (fields / "region.json").is_file() else
                        "\n⛔ Hand-made fields already exist under this name. Pick another name.")
        self.name_hint.config(text=txt)

    def _check_map_step(self) -> bool:
        name = self.name_var.get().strip()
        if not rg.NAME_RE.match(name or ""):
            messagebox.showerror("Output name", "Use only letters, digits, '_' or '-'.")
            return False
        fields = rg.REPO_ROOT / "velocity_fields" / name
        if (fields / "raw").exists() and not (fields / "region.json").is_file():
            messagebox.showerror("Output name", f"velocity_fields/{name}/raw already holds velocity fields "
                                                "that this tool did not create. Choose another name.")
            return False
        return True

    def load_map_then(self, on_ready):
        info = maps.MAPS[self.map_var.get()]
        if self.loaded_map == info.name:
            on_ready()
            return
        model_dir = maps.find_model_dir(info)
        if model_dir is None and not messagebox.askyesno(
                "Download map", f"'{info.model}' is not in your local Gazebo Fuel cache.\n\n"
                                f"Download it from {maps.FUEL_SERVER}?"):
            return

        def work():
            try:
                d = model_dir
                if d is None:
                    d = maps.download_model(info, progress=lambda got, tot: self.msgs.put(
                        ("status", f"Downloading {info.model}: {got / 1e6:.0f}"
                                   + (f" / {tot / 1e6:.0f} MB" if tot else " MB"))))
                self.msgs.put(("status", "Preparing map background (first time only, may take a minute)…"))
                img, extent = maps.background(info, d)
                self.msgs.put(("map_ready", (info, img, extent, on_ready)))
            except Exception as exc:  # shown to the user
                self.msgs.put(("error", f"Could not load map: {exc}"))

        self.next_btn.state(["disabled"])
        self.status.set("Loading map…")
        threading.Thread(target=work, daemon=True).start()

    def _show_map(self, info, img, extent):
        self.ax.clear()
        self.ax.imshow(img, extent=extent, origin="upper", interpolation="bilinear", zorder=0)
        self.ax.set_xlim(extent[0], extent[1])
        self.ax.set_ylim(extent[2], extent[3])
        self.ax.set_aspect("equal")
        self.ax.set_xlabel("x (m, Gazebo world)")
        self.ax.set_ylabel("y (m, Gazebo world)")
        self.ax.set_title(f"{info.name}   (pale blue = no terrain, usually open water)", fontsize=10)
        (self.rubber,) = self.ax.plot([], [], "--", color="#d62728", lw=1, zorder=5)
        self.overlay = []
        self.toolbar.update()  # make 'home' return to this view
        self.map_info, self.loaded_map = info, info.name
        self.redraw()

    # ---------------------------------------------------------------- step 2
    def _panel_draw(self):
        f = self._panel("2. Draw the region",
                        "Left-click to add points along the water region's outline. Click the first "
                        "point again (or press 'Close polygon') to finish. Right-click removes the "
                        "last point.\n\nUse the toolbar under the map to zoom or pan; turn the tool "
                        "off again before clicking points.")
        row = ttk.Frame(f)
        row.pack(fill=tk.X)
        ttk.Button(row, text="Undo point", command=self.undo_point).pack(side=tk.LEFT, expand=True, fill=tk.X)
        ttk.Button(row, text="Clear", command=self.clear_poly).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=4)
        ttk.Button(f, text="Close polygon", command=self.close_poly).pack(fill=tk.X, pady=(6, 0))
        ttk.Separator(f).pack(fill=tk.X, pady=10)
        ttk.Button(f, text="Load outline from a .geo or region.json…",
                   command=self.load_outline).pack(fill=tk.X)
        self.poly_info = ttk.Label(f, foreground="#555", justify=tk.LEFT, wraplength=370)
        self.poly_info.pack(anchor="w", pady=10)
        return f

    def undo_point(self):
        if self.closed:
            self.closed = False
        elif self.poly:
            self.poly.pop()
        self._poly_changed()

    def clear_poly(self):
        self.poly, self.closed = [], False
        self._poly_changed()

    def close_poly(self):
        if len(self.poly) < 3:
            messagebox.showinfo("Close polygon", "Add at least 3 points first.")
            return
        if rg.self_intersects(self.poly):
            messagebox.showerror("Close polygon", "The outline crosses itself. Undo some points and try again.")
            return
        self.closed = True
        self._poly_changed()
        self.go_next()  # straight on to boundary conditions

    def _poly_changed(self, keep_edges=False):
        if not keep_edges:
            old = self.edges
            self.edges = [old[i] if i < len(old) else rg.EdgeBC() for i in range(len(self.poly))] \
                if self.closed else []
        self.selected.clear()
        self.mesh_ready, self.mesh_xy, self.mesh_tris = False, None, None
        self.rubber.set_data([], [])
        n = len(self.poly)
        txt = f"{n} point{'s' if n != 1 else ''}"
        if self.closed:
            per = sum(rg.edge_length(self.poly, i) for i in range(n))
            txt += f", closed\nArea {rg.polygon_area(self.poly) / 1e6:.3f} km², perimeter {per:.0f} m"
        self.poly_info.config(text=txt)
        self.redraw()

    def load_outline(self):
        path = filedialog.askopenfilename(title="Outline file", initialdir=rg.REPO_ROOT / "meshes",
                                          filetypes=[("gmsh geo / region", "*.geo *.json"), ("All", "*")])
        if not path:
            return
        try:
            if path.endswith(".json"):
                r = rg.Region.load(Path(path))
                poly, edges = r.polygon, r.edges
            else:
                poly, edges = rg.polygon_from_geo(Path(path))
        except Exception as exc:
            messagebox.showerror("Load outline", f"Could not read {path}:\n{exc}")
            return
        if len(poly) < 3:
            messagebox.showerror("Load outline", "No polygon found in that file.")
            return
        self.poly, self.closed = poly, not rg.self_intersects(poly)
        self.edges = edges if (edges and len(edges) == len(poly)) else [rg.EdgeBC() for _ in poly]
        self._poly_changed(keep_edges=True)
        self.status.set(f"Loaded {len(poly)} points from {Path(path).name}"
                        + ("; boundary types taken from its physical groups" if edges else ""))

    # ---------------------------------------------------------------- step 3
    def _panel_bcs(self):
        f = self._panel("3. Boundary conditions",
                        "Every edge starts as shore (no flow through it). Select edges by clicking them "
                        "on the map (Shift-click adds) or in the list, then set their type.")
        cols = ("edge", "type", "value", "length")
        self.tree = ttk.Treeview(f, columns=cols, show="headings", height=11, selectmode="extended")
        for c, w in zip(cols, (50, 130, 80, 80)):
            self.tree.heading(c, text=c.capitalize())
            self.tree.column(c, width=w, anchor="center")
        self.tree.pack(fill=tk.X)
        self.tree.bind("<<TreeviewSelect>>", self._tree_selected)
        ttk.Button(f, text="Select all", command=lambda: self._select(set(range(len(self.edges))))
                   ).pack(anchor="e", pady=4)

        box = ttk.LabelFrame(f, text="Set selected edges to", padding=8)
        box.pack(fill=tk.X, pady=6)
        self.kind_var = tk.StringVar(value="inflow")
        for kind, (label, expl, _unit) in rg.EDGE_KINDS.items():
            r = ttk.Frame(box)
            r.pack(fill=tk.X)
            tk.Label(r, text="■", fg=EDGE_COLORS[kind]).pack(side=tk.LEFT)
            ttk.Radiobutton(r, text=f"{label}: {expl}", value=kind, variable=self.kind_var,
                            command=self._kind_changed).pack(side=tk.LEFT)
        vr = ttk.Frame(box)
        vr.pack(fill=tk.X, pady=(6, 0))
        self.value_label = ttk.Label(vr, text="Value")
        self.value_label.pack(side=tk.LEFT)
        self.value_var = tk.StringVar()
        self.value_entry = ttk.Entry(vr, textvariable=self.value_var, width=10)
        self.value_entry.pack(side=tk.LEFT, padx=6)
        ttk.Button(box, text="Apply", command=self.apply_bc).pack(fill=tk.X, pady=(8, 0))
        self.bc_info = ttk.Label(f, foreground="#555", wraplength=370, justify=tk.LEFT)
        self.bc_info.pack(anchor="w", pady=6)
        self._kind_changed()
        return f

    def _kind_changed(self):
        kind = self.kind_var.get()
        unit = rg.EDGE_KINDS[kind][2]
        if unit is None:
            self.value_var.set("")
            self.value_entry.state(["disabled"])
            self.value_label.config(text="No value needed")
        else:
            self.value_entry.state(["!disabled"])
            self.value_var.set(f"{rg.DEFAULT_VALUES[kind]:g}")
            self.value_label.config(text={"m/s": "Speed (m/s)", "m": "Water level (m)"}[unit])

    def apply_bc(self):
        if not self.selected:
            messagebox.showinfo("Boundary conditions", "Select one or more edges first.")
            return
        kind = self.kind_var.get()
        value = 0.0
        if rg.EDGE_KINDS[kind][2] is not None:
            try:
                value = float(self.value_var.get())
            except ValueError:
                messagebox.showerror("Boundary conditions", "The value must be a number.")
                return
            if kind in rg.SPEED_KINDS and value <= 0:
                messagebox.showerror("Boundary conditions", "Speeds must be greater than 0.")
                return
        for i in self.selected:
            self.edges[i] = rg.EdgeBC(kind, value)
        self.mesh_ready = False
        self._refresh_tree()
        self.redraw()

    def _refresh_tree(self):
        self.tree.delete(*self.tree.get_children())
        for i, e in enumerate(self.edges):
            unit = rg.EDGE_KINDS[e.kind][2]
            self.tree.insert("", tk.END, iid=str(i), values=(
                i + 1, rg.EDGE_KINDS[e.kind][0], f"{e.value:g} {unit}" if unit else "–",
                f"{rg.edge_length(self.poly, i):.0f} m"))
        self.tree.selection_set([str(i) for i in sorted(self.selected)])
        counts = {k: sum(e.kind == k for e in self.edges) for k in rg.EDGE_KINDS}
        txt = ", ".join(f"{n} {rg.EDGE_KINDS[k][0].lower()}" for k, n in counts.items() if n)
        if not any(e.kind in rg.LEVEL_KINDS for e in self.edges) and any(e.kind != "wall" for e in self.edges):
            txt += ("\n\nTip: with no open water-level edge, the water level can only stay steady if "
                    "inflow and outflow balance exactly. Usually at least one edge should be open.")
        self.bc_info.config(text=txt)

    def _tree_selected(self, _e=None):
        sel = {int(i) for i in self.tree.selection()}
        if sel != self.selected:
            self.selected = sel
            self.redraw()

    def _select(self, sel):
        self.selected = set(sel)
        self.tree.selection_set([str(i) for i in sorted(sel)])
        self.redraw()

    # ---------------------------------------------------------------- step 4
    def _panel_settings(self):
        f = self._panel("4. Depth and simulation settings",
                        "The depth is used everywhere in the region (flat bed). The other settings have "
                        "workable defaults.")
        grid = ttk.Frame(f)
        grid.pack(fill=tk.X)
        self.sim_vars = {}
        for r, (attr, label, _lo, _strict) in enumerate(rg.SimSettings.FIELDS):
            lab = ttk.Label(grid, text=label + (" *" if attr == "depth_m" else ""))
            if attr == "depth_m":
                lab.config(font=("TkDefaultFont", 10, "bold"))
            lab.grid(row=r, column=0, sticky="w", pady=3)
            v = tk.StringVar(value=f"{getattr(self.sim, attr):g}")
            v.trace_add("write", lambda *_: self._update_estimate())
            ttk.Entry(grid, textvariable=v, width=10).grid(row=r, column=1, sticky="e", padx=6)
            self.sim_vars[attr] = v
        grid.columnconfigure(0, weight=1)
        self.estimate = ttk.Label(f, foreground="#555", wraplength=370, justify=tk.LEFT)
        self.estimate.pack(anchor="w", pady=10)
        ttk.Label(f, foreground="#777", wraplength=370, justify=tk.LEFT, text=(
            "Boundary ramp-up: inflow, outflow and open water levels grow from 0 to their set values over "
            "this time, which avoids a start-up shock.")).pack(anchor="w")
        return f

    def _read_sim(self, show_errors=True):
        values = {}
        for attr, label, lo, strict in rg.SimSettings.FIELDS:
            try:
                v = float(self.sim_vars[attr].get())
            except ValueError:
                if show_errors:
                    messagebox.showerror("Settings", f"{label}: please enter a number.")
                return None
            if (v <= lo) if strict else (v < lo):
                if show_errors:
                    messagebox.showerror("Settings", f"{label} must be {'>' if strict else '>='} {lo:g}.")
                return None
            values[attr] = v
        sim = rg.SimSettings(**values)
        if not rg.export_fits_timestep(sim):
            if show_errors:
                messagebox.showerror("Settings", "'Save every' must be a whole multiple of the solver time step.")
            return None
        return sim

    def _update_estimate(self):
        s = self._read_sim(show_errors=False)
        if s is None or len(self.poly) < 3:
            self.estimate.config(text="")
            return
        cells = rg.polygon_area(self.poly) / (0.433 * s.mesh_size_m ** 2)
        exports = int(s.duration_h * 3600 / s.export_interval_s) + 1
        steps = int(s.duration_h * 3600 / s.timestep_s)
        minutes = steps * cells * SECONDS_PER_CELL_STEP / 60
        self.estimate.config(text=(
            f"≈ {cells:,.0f} triangles, {steps:,} time steps, {exports} saved snapshots.\n"
            f"Rough run time on one core: {minutes:,.0f} min (plus a few minutes of one-off "
            f"code compilation the first time a setup is used)."))

    # ---------------------------------------------------------------- step 5
    def _panel_run(self):
        f = self._panel("5. Build the mesh and run", "")
        self.summary = ttk.Label(f, justify=tk.LEFT, wraplength=370)
        self.summary.pack(anchor="w")
        ttk.Separator(f).pack(fill=tk.X, pady=8)
        self.mesh_btn = ttk.Button(f, text="1. Build mesh (gmsh)", command=self.build_mesh)
        self.mesh_btn.pack(fill=tk.X)
        self.run_btn = ttk.Button(f, text="2. Run flow simulation (Firedrake)", command=self.run_flow)
        self.run_btn.pack(fill=tk.X, pady=4)
        row = ttk.Frame(f)
        row.pack(fill=tk.X)
        self.stop_btn = ttk.Button(row, text="Stop", command=self.stop_proc)
        self.stop_btn.pack(side=tk.LEFT, expand=True, fill=tk.X)
        ttk.Button(row, text="Open output folder", command=self.open_output).pack(
            side=tk.LEFT, expand=True, fill=tk.X, padx=(4, 0))
        self.progress = ttk.Progressbar(f, maximum=100)
        self.progress.pack(fill=tk.X, pady=6)
        self.log = ScrolledText(f, height=14, font=("TkFixedFont", 8), state="disabled")
        self.log.pack(fill=tk.BOTH, expand=True)
        python, env = firedrake_env()
        ttk.Label(f, foreground="#777", wraplength=370, justify=tk.LEFT,
                  text=f"Firedrake Python: {python}\nPETSC_DIR: {env.get('PETSC_DIR', '(not set)')}"
                  ).pack(anchor="w", pady=(6, 0))
        return f

    def region(self) -> rg.Region:
        info = self.map_info or maps.MAPS[self.map_var.get()]
        return rg.Region(map_name=info.name, output_name=self.name_var.get().strip(),
                         world_offset=info.world_offset, polygon=[list(p) for p in self.poly],
                         edges=list(self.edges), sim=self.sim)

    def _update_run_panel(self):
        r = self.region()
        kinds = ", ".join(f"{i + 1}: {rg.EDGE_KINDS[e.kind][0].lower()}"
                          + (f" {e.value:g}" if rg.EDGE_KINDS[e.kind][2] else "")
                          for i, e in enumerate(r.edges) if e.kind != "wall")
        self.summary.config(text=(
            f"Map: {r.map_name}    Output: {r.output_name}\n"
            f"Region: {len(r.polygon)} edges, {rg.polygon_area(r.polygon) / 1e6:.3f} km²\n"
            f"Open edges: {kinds or 'none'} (all others shore)\n"
            f"Depth {r.sim.depth_m:g} m, cells {r.sim.mesh_size_m:g} m, {r.sim.duration_h:g} h, "
            f"save every {r.sim.export_interval_s:g} s"))
        busy = self.proc is not None
        self.mesh_btn.state(["disabled"] if busy else ["!disabled"])
        self.run_btn.state(["disabled"] if busy or not self.mesh_ready else ["!disabled"])
        self.stop_btn.state(["!disabled"] if busy else ["disabled"])

    def build_mesh(self):
        r = self.region()
        problems = r.problems()
        if problems:
            messagebox.showerror("Region", "\n".join(problems))
            return
        path = r.save()
        self._log_clear()
        self._log(f"Saved {path}\n")
        self.progress.config(mode="indeterminate")
        self.progress.start(10)
        self._start([sys.executable, str(HERE / "build_mesh.py"), str(path)], dict(os.environ), "mesh")

    def run_flow(self):
        r = self.region()
        args = []
        if (r.fields_dir / "raw").exists():
            if not messagebox.askyesno("Replace fields?", f"velocity_fields/{r.output_name}/raw already exists "
                                                          "(made by this tool). Replace it?"):
                return
            args.append("--overwrite")
        python, env = firedrake_env()
        self.progress.stop()
        self.progress.config(mode="determinate", value=0)
        self._log(f"\n$ {python} run_flow.py {r.region_file} {' '.join(args)}\n")
        self._start([python, str(HERE / "run_flow.py"), str(r.region_file), *args], env, "flow")

    def _start(self, cmd, env, kind):
        try:
            self.proc = subprocess.Popen(cmd, cwd=rg.REPO_ROOT, env=env, text=True, bufsize=1,
                                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        except OSError as exc:
            messagebox.showerror("Run", f"Could not start {cmd[0]}:\n{exc}")
            return
        self.proc_kind, self.proc_tail = kind, []
        proc = self.proc

        def reader():
            for line in proc.stdout:
                self.msgs.put(("line", line))
            self.msgs.put(("done", (kind, proc.wait())))

        threading.Thread(target=reader, daemon=True).start()
        self._update_run_panel()

    def stop_proc(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            self._log("\n[stopped by user]\n")

    def _proc_done(self, kind, code):
        self.proc = None
        self.progress.stop()
        self.progress.config(mode="determinate")
        r = self.region()
        if kind == "mesh" and code == 0:
            self.mesh_xy, self.mesh_tris = read_msh_triangles(r.mesh_file)
            self.mesh_ready = True
            self.progress.config(value=100)
            self.status.set(f"Mesh ready: {len(self.mesh_tris):,} triangles. Check it on the map, then run the flow.")
            self.redraw()
        elif kind == "flow" and code == 0:
            self.progress.config(value=100)
            self.status.set(f"Done. Velocity fields in velocity_fields/{r.output_name}/raw/")
            messagebox.showinfo("Finished", f"Velocity fields written to\nvelocity_fields/{r.output_name}/raw/\n\n"
                                            "Next: tools/vtu_converter to build the HDF5 lookup.")
        else:
            tail = "".join(self.proc_tail)
            hint = ""
            if "cannot open shared object file" in tail:
                hint = ("\n\nFiredrake could not load a system library. See 'Firedrake environment' "
                        "in tools/flow_generator/README.md.")
            self.status.set(f"{kind} step failed (exit code {code}).")
            if code not in (-15,):  # not a user stop
                messagebox.showerror("Failed", f"The {kind} step failed (exit code {code}). "
                                               f"See the log for details.{hint}")
        self._update_run_panel()

    def open_output(self):
        r = self.region()
        target = r.fields_dir if r.fields_dir.exists() else r.mesh_dir
        if target.exists():
            subprocess.Popen(["xdg-open", str(target)])
        else:
            messagebox.showinfo("Output", "Nothing has been written yet.")

    def _log(self, text):
        self.log.config(state="normal")
        self.log.insert(tk.END, text)
        self.log.see(tk.END)
        self.log.config(state="disabled")

    def _log_clear(self):
        self.log.config(state="normal")
        self.log.delete("1.0", tk.END)
        self.log.config(state="disabled")

    # ================================================================ events
    def _poll(self):
        try:
            while True:
                kind, payload = self.msgs.get_nowait()
                if kind == "status":
                    self.status.set(payload)
                elif kind == "error":
                    self.status.set(payload)
                    self.next_btn.state(["!disabled"])
                    messagebox.showerror("Error", payload)
                elif kind == "map_ready":
                    info, img, extent, on_ready = payload
                    self._show_map(info, img, extent)
                    self.next_btn.state(["!disabled"])
                    self.status.set("")
                    on_ready()
                elif kind == "line":
                    self._log(payload)
                    self.proc_tail = (self.proc_tail + [payload])[-40:]
                    m = re.search(r"\[flow\] export (\d+)/(\d+)", payload)
                    if m:
                        done, total = int(m.group(1)), int(m.group(2))
                        self.progress.config(value=100 * done / total)
                        self.status.set(f"Running flow: snapshot {done} of {total}")
                elif kind == "done":
                    self._proc_done(*payload)
        except queue.Empty:
            pass
        self.root.after(100, self._poll)

    def _px(self, x, y):
        return self.ax.transData.transform((x, y))

    def on_click(self, ev):
        if ev.inaxes is not self.ax or self.toolbar.mode or ev.xdata is None or self.map_info is None:
            return
        if self.step == 1:
            if ev.button == 3:
                self.undo_point()
            elif ev.button == 1 and not self.closed:
                if len(self.poly) >= 3 and np.hypot(*(self._px(*self.poly[0]) - np.array([ev.x, ev.y]))) < PICK_PX:
                    self.close_poly()
                else:
                    self.poly.append([float(ev.xdata), float(ev.ydata)])
                    self._poly_changed()
        elif self.step == 2 and ev.button == 1:
            i = self._nearest_edge(ev.x, ev.y)
            if i is None:
                return
            add = ev.key in ("shift", "control", "ctrl+shift")
            if add:
                self._select(self.selected ^ {i})
            else:
                self._select({i})

    def on_move(self, ev):
        if self.step != 1 or self.closed or not self.poly or ev.inaxes is not self.ax or ev.xdata is None:
            if self.rubber.get_xdata() is not None and len(self.rubber.get_xdata()):
                self.rubber.set_data([], [])
                self.canvas.draw_idle()
            return
        self.rubber.set_data([self.poly[-1][0], ev.xdata], [self.poly[-1][1], ev.ydata])
        self.canvas.draw_idle()

    def _nearest_edge(self, px, py):
        best, best_d = None, PICK_PX
        p = np.array([px, py])
        n = len(self.poly)
        for i in range(n):
            a = self._px(*self.poly[i])
            b = self._px(*self.poly[(i + 1) % n])
            ab = b - a
            t = np.clip(np.dot(p - a, ab) / max(np.dot(ab, ab), 1e-12), 0, 1)
            d = np.hypot(*(a + t * ab - p))
            if d < best_d:
                best, best_d = i, d
        return best

    # ================================================================ drawing
    def redraw(self):
        for a in self.overlay:
            a.remove()
        self.overlay = []
        ax, P = self.ax, np.array(self.poly)
        if self.mesh_ready and self.mesh_tris is not None and self.step == 4:
            self.overlay.append(ax.triplot(self.mesh_xy[:, 0], self.mesh_xy[:, 1], self.mesh_tris,
                                           color="#333", lw=0.3, alpha=0.6, zorder=2)[0])
        if len(P):
            if self.closed:
                n = len(P)
                show_labels = n <= 80
                for i in range(n):
                    a, b = P[i], P[(i + 1) % n]
                    kind = self.edges[i].kind if i < len(self.edges) else "wall"
                    if i in self.selected:
                        self.overlay.append(ax.plot(*np.c_[a, b], color="#ffd400", lw=9, alpha=0.9,
                                                    solid_capstyle="round", zorder=3)[0])
                    self.overlay.append(ax.plot(*np.c_[a, b], color=EDGE_COLORS[kind],
                                                lw=3.5 if kind != "wall" else 2, zorder=4)[0])
                    if show_labels and self.step >= 2:
                        m = (a + b) / 2
                        self.overlay.append(ax.text(m[0], m[1], str(i + 1), fontsize=7, ha="center", va="center",
                                                    zorder=6, color=EDGE_COLORS[kind],
                                                    bbox=dict(boxstyle="round,pad=0.15", fc="white",
                                                              ec=EDGE_COLORS[kind], lw=0.6)))
            else:
                self.overlay.append(ax.plot(P[:, 0], P[:, 1], "-", color="#d62728", lw=2, zorder=4)[0])
                self.overlay.append(ax.plot(*P[0], "o", ms=11, mfc="none", mec="#d62728", mew=2, zorder=5)[0])
            self.overlay.append(ax.plot(P[:, 0], P[:, 1], "o", ms=3.5, color="#d62728", zorder=5)[0])
        self.canvas.draw_idle()

    # ================================================================ steps
    def show_step(self, i):
        self.step = i
        for k, lab in enumerate(self.step_labels):
            lab.configure(style="StepOn.TLabel" if k == i else "Step.TLabel")
        for p in self.panels:
            p.pack_forget()
        self.panels[i].pack(fill=tk.BOTH, expand=True)
        self.back_btn.state(["disabled"] if i == 0 else ["!disabled"])
        self.next_btn.state(["disabled"] if i == len(STEPS) - 1 else ["!disabled"])
        if i == 2:
            self._refresh_tree()
        if i == 3:
            self._update_estimate()
        if i == 4:
            self._update_run_panel()
        self.status.set({0: "", 1: "Click on the map to add points.",
                         2: "Click an edge on the map or in the list to select it.",
                         3: "", 4: "Build the mesh first, check it, then run the flow."}[i])
        self.redraw()

    def go_back(self):
        if self.proc is not None:
            messagebox.showinfo("Busy", "Wait for the running step to finish, or press Stop.")
            return
        if self.step > 0:
            self.show_step(self.step - 1)

    def go_next(self):
        s = self.step
        if s == 0:
            if self._check_map_step():
                if self.loaded_map and self.loaded_map != self.map_var.get() and self.poly:
                    if not messagebox.askyesno("Change map", "Switching map clears the region you drew. Continue?"):
                        return
                    self.clear_poly()
                self.load_map_then(lambda: self.show_step(1))
        elif s == 1:
            if not self.closed:
                self.close_poly() if len(self.poly) >= 3 else messagebox.showinfo(
                    "Draw region", "Draw the region and close it first.")
                return
            self.show_step(2)
        elif s == 2:
            if all(e.kind == "wall" for e in self.edges):
                messagebox.showerror("Boundary conditions", "All edges are shore. Set at least one edge "
                                                            "to inflow, outflow or open water level.")
                return
            if not any(e.kind in rg.LEVEL_KINDS for e in self.edges) and not messagebox.askyesno(
                    "No open edge", "No edge has an open water level, so the water level will drift unless "
                                    "inflow and outflow balance exactly. Continue anyway?"):
                return
            self.show_step(3)
        elif s == 3:
            sim = self._read_sim()
            if sim is None:
                return
            if sim != self.sim:
                self.sim, self.mesh_ready = sim, False
            self.show_step(4)

    # ================================================================ files
    def load_region(self):
        path = filedialog.askopenfilename(title="Saved region", initialdir=rg.REPO_ROOT / "meshes",
                                          filetypes=[("region", "region.json *.json")])
        if not path:
            return
        try:
            r = rg.Region.load(Path(path))
        except Exception as exc:
            messagebox.showerror("Load region", f"Could not read {path}:\n{exc}")
            return
        if r.map_name not in maps.MAPS:
            messagebox.showerror("Load region", f"Unknown map '{r.map_name}'.")
            return
        self.map_var.set(r.map_name)
        self._on_map_change()
        self.name_var.set(r.output_name)
        self.sim = r.sim
        for attr, v in self.sim_vars.items():
            v.set(f"{getattr(r.sim, attr):g}")

        def ready():
            self.poly, self.closed, self.edges = r.polygon, True, r.edges
            self._poly_changed(keep_edges=True)
            self.show_step(2)

        self.load_map_then(ready)


def main():
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
