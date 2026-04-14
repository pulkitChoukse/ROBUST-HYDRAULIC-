Integrates:
  - MOC simulation engine  (moc_module.py)
  - Trained RandomForest   (model.pkl)
  - Safe-range scanner
  - Tkinter GUI with matplotlib plots
"""

import os
import sys
import pickle
import tkinter as tk
from tkinter import messagebox, ttk
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

from moc_module import PenstockParams, SimulationEngine

# ── constants ────────────────────────────────────────────────────────────────
G               = 9.81
FIXED_FRICTION  = 0.015
MODEL_PATH      = os.path.join(os.path.dirname(__file__), "model.pkl")

# Input ranges (min, max, default)
PARAM_CONFIG = {
    "Penstock Length (m)":      (100,  5000, 1000),
    "Pipe Diameter (m)":        (0.5,  10,   2.0),
    "Wave Speed (m/s)":         (500,  2000, 1000),
    "Initial Velocity (m/s)":   (0.5,  10,   3.0),
    "Initial Pressure Head (m)":(50,   500,  200),
    "Max Pressure Head (m)":    (100,  600,  280),
    "Min Pressure Head (m)":    (0,    100,  20),
    "Closure Time (s)":         (1,    60,   15),
}


# ═══════════════════════════════════════════════════════════════════════════════
#  ML MODEL LOADER
# ═══════════════════════════════════════════════════════════════════════════════

class MLPredictor:
    """
    Wraps the trained RandomForest pipeline from model.pkl.
    Falls back gracefully if the file is missing or sklearn is absent.
    """

    FEATURE_ORDER = [
        "length", "diameter", "wave_speed", "initial_velocity",
        "initial_pressure_head", "max_pressure_head", "min_pressure_head",
        "joukowsky_rise", "wave_period", "head_margin",
        "velocity_ratio", "pipe_aspect",
    ]

    def __init__(self):
        self._pipeline   = None
        self._loaded     = False
        self._load_error = ""
        self._load()

    def _load(self):
        if not os.path.exists(MODEL_PATH):
            self._load_error = f"model.pkl not found at {MODEL_PATH}"
            return
        try:
            with open(MODEL_PATH, "rb") as f:
                payload = pickle.load(f)
            self._pipeline = payload["pipeline"]
            self._loaded   = True
        except Exception as e:
            self._load_error = str(e)

    @property
    def loaded(self):
        return self._loaded

    @property
    def load_error(self):
        return self._load_error

    def _build_feature_vector(self, p: PenstockParams) -> np.ndarray:
        """Build the 12-feature vector the model was trained on."""
        joukowsky_rise = p.wave_speed * p.initial_velocity / G
        wave_period    = 2.0 * p.length / p.wave_speed
        head_margin    = (p.max_pressure_head - p.initial_pressure_head) / max(p.initial_pressure_head, 1e-6)
        velocity_ratio = p.initial_velocity / (p.wave_speed / G)
        pipe_aspect    = p.length / max(p.diameter, 1e-6)

        return np.array([[
            p.length,
            p.diameter,
            p.wave_speed,
            p.initial_velocity,
            p.initial_pressure_head,
            p.max_pressure_head,
            p.min_pressure_head,
            joukowsky_rise,
            wave_period,
            head_margin,
            velocity_ratio,
            pipe_aspect,
        ]])

    def predict(self, p: PenstockParams) -> float:
        """Returns predicted minimum safe Tc in seconds."""
        if not self._loaded:
            raise RuntimeError(self._load_error or "Model not loaded.")
        X      = self._build_feature_vector(p)
        tc     = float(self._pipeline.predict(X)[0])
        return max(1.0, round(tc, 2))

    def predict_with_uncertainty(self, p: PenstockParams) -> dict:
        """Returns mean, std, p10, p90 from per-tree predictions."""
        if not self._loaded:
            raise RuntimeError(self._load_error or "Model not loaded.")
        X       = self._build_feature_vector(p)
        scaler  = self._pipeline.named_steps["scaler"]
        rf      = self._pipeline.named_steps["rf"]
        X_sc    = scaler.transform(X)
        preds   = np.array([t.predict(X_sc)[0] for t in rf.estimators_])
        mean_tc = float(np.mean(preds))
        std_tc  = float(np.std(preds))
        return {
            "tc":   max(1.0, round(mean_tc, 2)),
            "std":  round(std_tc, 2),
            "p10":  round(max(1.0, float(np.percentile(preds, 10))), 2),
            "p90":  round(float(np.percentile(preds, 90)), 2),
            "conf": "High" if std_tc < 1.5 else ("Medium" if std_tc < 4 else "Low"),
        }


# ═══════════════════════════════════════════════════════════════════════════════
#  SAFE RANGE SCANNER
# ═══════════════════════════════════════════════════════════════════════════════

class SafeRangeFinder:
    def __init__(self, n_points: int = 30):
        self.n_points = n_points
        self._engine  = SimulationEngine()

    def scan(self, params: PenstockParams):
        wave_period = 2.0 * params.length / params.wave_speed
        tc_lo = max(1.0, wave_period * 0.25)
        tc_hi = min(60.0, wave_period * 8.0)
        tc_vals    = np.linspace(tc_lo, tc_hi, self.n_points)
        peak_heads = []
        is_safe    = []
        for tc in tc_vals:
            res = self._engine.run_simulation(params, float(tc))
            peak_heads.append(res.head)
            is_safe.append(res.is_safe)
        return tc_vals, np.array(peak_heads), np.array(is_safe)


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════

class Dashboard:

    def __init__(self, root: tk.Tk):
        self.root      = root
        self.root.title("Hydraulic Transient Analysis — EG4")
        self.root.resizable(True, True)

        self._engine    = SimulationEngine()
        self._predictor = MLPredictor()
        self._scanner   = SafeRangeFinder(n_points=30)
        self._last_result = None

        self._build_ui()
        self._update_ml_status()

    # ── UI construction ──────────────────────────────────────────────────────

    def _build_ui(self):
        # ── top status bar ──────────────────────────────────────────────────
        top = tk.Frame(self.root, bg="#1a1a2e")
        top.pack(fill="x")
        tk.Label(top, text="Hydraulic Transient Analysis Dashboard",
                 font=("Consolas", 13, "bold"), bg="#1a1a2e", fg="#e0e0ff",
                 pady=8).pack(side="left", padx=12)
        self._ml_status_lbl = tk.Label(top, text="", font=("Consolas", 10),
                                       bg="#1a1a2e", fg="#aaffaa", pady=8)
        self._ml_status_lbl.pack(side="right", padx=12)

        # ── main body ───────────────────────────────────────────────────────
        body = tk.Frame(self.root)
        body.pack(fill="both", expand=True, padx=10, pady=8)

        # Left column: inputs
        left = tk.LabelFrame(body, text="Input Parameters",
                             font=("Consolas", 10, "bold"), padx=8, pady=8)
        left.grid(row=0, column=0, sticky="ns", padx=(0, 8))

        self._entries: dict[str, tk.Entry] = {}

        for i, (label, (lo, hi, default)) in enumerate(PARAM_CONFIG.items()):
            tk.Label(left, text=f"{label}", font=("Consolas", 9),
                     anchor="w").grid(row=i, column=0, sticky="w", pady=2)
            tk.Label(left, text=f"[{lo} – {hi}]", font=("Consolas", 8),
                     fg="#888", anchor="w").grid(row=i, column=1, sticky="w", padx=(4, 8))
            entry = tk.Entry(left, width=10, font=("Consolas", 10))
            entry.insert(0, str(default))
            entry.grid(row=i, column=2, pady=2)
            self._entries[label] = entry

        # Closure time row gets a highlight
        ct_entry = self._entries["Closure Time (s)"]
        ct_entry.config(bg="#fffde7")

        # ── ML prediction panel ─────────────────────────────────────────────
        ml_frame = tk.LabelFrame(left, text="ML Prediction",
                                 font=("Consolas", 9, "bold"), padx=6, pady=6)
        ml_frame.grid(row=len(PARAM_CONFIG), column=0, columnspan=3,
                      sticky="ew", pady=(10, 2))

        tk.Button(ml_frame, text="Predict Tc (ML Model) →",
                  font=("Consolas", 9, "bold"), bg="#1565C0", fg="white",
                  relief="flat", padx=6, pady=4,
                  command=self._on_predict).pack(fill="x", pady=(0, 6))

        self._pred_lbl = tk.Label(ml_frame, text="Predicted Tc: —",
                                  font=("Consolas", 9), fg="#1565C0", anchor="w")
        self._pred_lbl.pack(fill="x")
        self._conf_lbl = tk.Label(ml_frame, text="Confidence: —",
                                  font=("Consolas", 9), fg="#555", anchor="w")
        self._conf_lbl.pack(fill="x")
        self._range_lbl = tk.Label(ml_frame, text="P10–P90: —",
                                   font=("Consolas", 9), fg="#555", anchor="w")
        self._range_lbl.pack(fill="x")
        self._juko_lbl = tk.Label(ml_frame, text="Joukowsky ΔH: —",
                                  font=("Consolas", 9), fg="#555", anchor="w")
        self._juko_lbl.pack(fill="x")
        self._wp_lbl = tk.Label(ml_frame, text="Wave period: —",
                                font=("Consolas", 9), fg="#555", anchor="w")
        self._wp_lbl.pack(fill="x")

        # ── action buttons ──────────────────────────────────────────────────
        btn_frame = tk.Frame(left)
        btn_frame.grid(row=len(PARAM_CONFIG)+1, column=0,
                       columnspan=3, pady=(10, 0), sticky="ew")

        tk.Button(btn_frame, text="Run Simulation",
                  font=("Consolas", 10, "bold"), bg="#2e7d32", fg="white",
                  relief="flat", padx=8, pady=6,
                  command=self._on_run).pack(side="left", fill="x", expand=True, padx=(0,4))
        tk.Button(btn_frame, text="Scan Safe Range",
                  font=("Consolas", 9), bg="#6a1b9a", fg="white",
                  relief="flat", padx=6, pady=6,
                  command=self._on_scan).pack(side="left", fill="x", expand=True, padx=(0,4))
        tk.Button(btn_frame, text="Clear",
                  font=("Consolas", 9), bg="#555", fg="white",
                  relief="flat", padx=6, pady=6,
                  command=self._on_clear).pack(side="left", fill="x", expand=True, padx=(0,4))
        tk.Button(btn_frame, text="Exit",
                  font=("Consolas", 9), bg="#b71c1c", fg="white",
                  relief="flat", padx=6, pady=6,
                  command=self.root.quit).pack(side="left", fill="x", expand=True)

        # ── safety status ───────────────────────────────────────────────────
        self._safety_lbl = tk.Label(left, text="",
                                    font=("Consolas", 11, "bold"), pady=6)
        self._safety_lbl.grid(row=len(PARAM_CONFIG)+2, column=0,
                              columnspan=3, sticky="ew")

        # Right column: plot area
        right = tk.Frame(body)
        right.grid(row=0, column=1, sticky="nsew")
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)

        self._fig = plt.Figure(figsize=(9, 7), tight_layout=True)
        self._canvas = FigureCanvasTkAgg(self._fig, master=right)
        self._canvas.get_tk_widget().pack(fill="both", expand=True)

        self._draw_placeholder()

    # ── helpers ──────────────────────────────────────────────────────────────

    def _update_ml_status(self):
        if self._predictor.loaded:
            self._ml_status_lbl.config(
                text="ML model: LOADED ✓", fg="#aaffaa")
        else:
            self._ml_status_lbl.config(
                text=f"ML model: NOT FOUND  ({self._predictor.load_error})",
                fg="#ff8888")

    def _read_params(self):
        """Parse all entry fields. Returns (values_dict, error_str)."""
        vals = {}
        for label, (lo, hi, _) in PARAM_CONFIG.items():
            raw = self._entries[label].get().strip()
            try:
                v = float(raw)
            except ValueError:
                return None, f"'{label}' is not a valid number."
            if not (lo <= v <= hi):
                return None, f"'{label}' must be between {lo} and {hi}."
            vals[label] = v
        return vals, None

    def _vals_to_params(self, vals: dict) -> PenstockParams:
        return PenstockParams(
            length               = vals["Penstock Length (m)"],
            diameter             = vals["Pipe Diameter (m)"],
            wave_speed           = vals["Wave Speed (m/s)"],
            initial_velocity     = vals["Initial Velocity (m/s)"],
            initial_pressure_head= vals["Initial Pressure Head (m)"],
            max_pressure_head    = vals["Max Pressure Head (m)"],
            min_pressure_head    = vals["Min Pressure Head (m)"],
            friction_factor      = FIXED_FRICTION,
            n_segments           = 100,
        )

    def _draw_placeholder(self):
        self._fig.clear()
        ax = self._fig.add_subplot(111)
        ax.text(0.5, 0.5,
                "Enter parameters and press\n'Predict Tc (ML Model)' then 'Run Simulation'",
                ha="center", va="center", fontsize=12, color="#aaa",
                transform=ax.transAxes)
        ax.axis("off")
        self._canvas.draw()

    # ── event handlers ────────────────────────────────────────────────────────

    def _on_predict(self):
        """Run ML prediction, fill Tc entry, show physics info."""
        vals, err = self._read_params()
        if err:
            messagebox.showerror("Input Error", err)
            return

        # Build params without closure_time
        p = self._vals_to_params(vals)
        if not p.validate():
            messagebox.showerror("Validation Error",
                                 "Check that Hmax > H0 and Hmin < H0.")
            return

        if not self._predictor.loaded:
            messagebox.showwarning("ML Model",
                                   f"Model not loaded:\n{self._predictor.load_error}\n\n"
                                   "You can still run simulation manually.")
            return

        try:
            result = self._predictor.predict_with_uncertainty(p)
        except Exception as e:
            messagebox.showerror("Prediction Error", str(e))
            return

        tc = result["tc"]

        # Fill Tc entry
        self._entries["Closure Time (s)"].delete(0, tk.END)
        self._entries["Closure Time (s)"].insert(0, str(tc))

        # Update labels
        juko = p.wave_speed * p.initial_velocity / G
        wp   = 2.0 * p.length / p.wave_speed

        self._pred_lbl.config(
            text=f"Predicted Tc: {tc} s")
        self._conf_lbl.config(
            text=f"Confidence: {result['conf']}  (σ = {result['std']} s)")
        self._range_lbl.config(
            text=f"P10–P90: {result['p10']} s – {result['p90']} s")
        self._juko_lbl.config(
            text=f"Joukowsky ΔH: {juko:.1f} m")
        self._wp_lbl.config(
            text=f"Wave period 2L/c: {wp:.3f} s")

        self._safety_lbl.config(
            text=f"ML suggested Tc = {tc} s  →  Press 'Run Simulation'",
            fg="#1565C0")

    def _on_run(self):
        """Run MOC simulation and display all plots."""
        vals, err = self._read_params()
        if err:
            messagebox.showerror("Input Error", err)
            return

        p  = self._vals_to_params(vals)
        tc = vals["Closure Time (s)"]

        if not p.validate():
            messagebox.showerror("Validation Error",
                                 "Invalid parameters. Check Hmax > H0 and Hmin < H0.")
            return

        self._safety_lbl.config(text="Simulating...", fg="#888")
        self.root.update_idletasks()

        try:
            result = self._engine.run_simulation(p, tc)
        except Exception as e:
            messagebox.showerror("Simulation Error", str(e))
            return

        self._last_result = result
        self._plot_simulation(result, p, tc)

        if result.is_safe:
            self._safety_lbl.config(
                text=f"SAFE  ✓   Peak Head = {result.head:.1f} m   "
                     f"Min Head = {result.min_head:.1f} m   Tc = {tc} s",
                fg="#2e7d32")
        else:
            self._safety_lbl.config(
                text=f"UNSAFE  ✗   Peak Head = {result.head:.1f} m "
                     f"exceeds Hmax = {p.max_pressure_head} m   Tc = {tc} s",
                fg="#b71c1c")

    def _on_scan(self):
        """Scan safe Tc range and display chart."""
        vals, err = self._read_params()
        if err:
            messagebox.showerror("Input Error", err)
            return

        p = self._vals_to_params(vals)
        if not p.validate():
            messagebox.showerror("Validation Error",
                                 "Check that Hmax > H0 and Hmin < H0.")
            return

        self._safety_lbl.config(text="Scanning safe Tc range...", fg="#888")
        self.root.update_idletasks()

        tc_vals, peaks, safe_flags = self._scanner.scan(p)

        # ML predicted Tc (if model loaded)
        ml_tc = None
        if self._predictor.loaded:
            try:
                ml_tc = self._predictor.predict(p)
            except Exception:
                pass

        self._plot_safe_range(tc_vals, peaks, safe_flags,
                              p.max_pressure_head, ml_tc)

        safe_tcs = tc_vals[safe_flags]
        if len(safe_tcs):
            self._safety_lbl.config(
                text=f"Safe range: Tc ≥ {safe_tcs.min():.1f} s   "
                     f"(ML predicted: {ml_tc} s)",
                fg="#6a1b9a")
        else:
            self._safety_lbl.config(
                text="No safe Tc found in scanned range. "
                     "Check Hmax or reduce velocity.",
                fg="#b71c1c")

    def _on_clear(self):
        for label, (lo, hi, default) in PARAM_CONFIG.items():
            self._entries[label].delete(0, tk.END)
            self._entries[label].insert(0, str(default))
        self._safety_lbl.config(text="")
        self._pred_lbl.config(text="Predicted Tc: —")
        self._conf_lbl.config(text="Confidence: —")
        self._range_lbl.config(text="P10–P90: —")
        self._juko_lbl.config(text="Joukowsky ΔH: —")
        self._wp_lbl.config(text="Wave period: —")
        self._draw_placeholder()

    # ── plotting ──────────────────────────────────────────────────────────────

    def _plot_simulation(self, result, params: PenstockParams, tc: float):
        self._fig.clear()
        gs = gridspec.GridSpec(2, 2, figure=self._fig,
                               hspace=0.42, wspace=0.35)

        # 1. Pressure head vs time
        ax1 = self._fig.add_subplot(gs[0, :])
        ax1.plot(result.time_arr, result.pressure_head,
                 color="#1565C0", lw=1.4, label="Pressure Head")
        ax1.axhline(params.max_pressure_head, color="#b71c1c",
                    lw=1.2, ls="--", label=f"Hmax = {params.max_pressure_head} m")
        ax1.axhline(params.min_pressure_head, color="#2e7d32",
                    lw=1.2, ls="--", label=f"Hmin = {params.min_pressure_head} m")
        ax1.axhline(params.initial_pressure_head, color="#888",
                    lw=0.8, ls=":", label=f"H0 = {params.initial_pressure_head} m")
        ax1.set_title(f"Pressure Head vs Time   (Tc = {tc} s)", fontsize=10)
        ax1.set_xlabel("Time (s)", fontsize=9)
        ax1.set_ylabel("Pressure Head (m)", fontsize=9)
        ax1.legend(fontsize=8, loc="upper right")
        ax1.tick_params(labelsize=8)
        peak_t = result.time_arr[np.argmax(result.pressure_head)]
        ax1.annotate(f"Peak = {result.head:.1f} m",
                     xy=(peak_t, result.head),
                     xytext=(peak_t + result.time_arr[-1]*0.03, result.head * 1.01),
                     fontsize=8, color="#b71c1c",
                     arrowprops=dict(arrowstyle="->", color="#b71c1c", lw=0.8))

        # 2. Velocity vs time
        ax2 = self._fig.add_subplot(gs[1, 0])
        ax2.plot(result.time_arr, result.velocity,
                 color="#2e7d32", lw=1.4, label="Velocity")
        ax2.axhline(0, color="#aaa", lw=0.8, ls=":")
        ax2.set_title("Velocity vs Time", fontsize=10)
        ax2.set_xlabel("Time (s)", fontsize=9)
        ax2.set_ylabel("Velocity (m/s)", fontsize=9)
        ax2.tick_params(labelsize=8)

        # 3. Key metrics summary
        ax3 = self._fig.add_subplot(gs[1, 1])
        ax3.axis("off")
        juko      = params.wave_speed * params.initial_velocity / G
        wave_p    = 2.0 * params.length / params.wave_speed
        head_marg = params.max_pressure_head - params.initial_pressure_head
        safe_str  = "SAFE ✓" if result.is_safe else "UNSAFE ✗"
        safe_col  = "#2e7d32" if result.is_safe else "#b71c1c"

        rows = [
            ("Parameter", "Value"),
            ("─"*18, "─"*12),
            ("Closure time Tc",     f"{tc} s"),
            ("Wave period 2L/c",    f"{wave_p:.2f} s"),
            ("Joukowsky ΔH",        f"{juko:.1f} m"),
            ("Head margin",         f"{head_marg:.1f} m"),
            ("Peak head",           f"{result.head:.1f} m"),
            ("Min head",            f"{result.min_head:.1f} m"),
            ("Simulation steps",    f"{result.n_steps}"),
            ("Status",              safe_str),
        ]
        y = 0.98
        for r0, r1 in rows:
            color = safe_col if r0 == "Status" else "black"
            weight = "bold" if r0 in ("Parameter", "Status") else "normal"
            ax3.text(0.02, y, r0, transform=ax3.transAxes,
                     fontsize=8.5, va="top", color="#444", fontweight=weight,
                     fontfamily="monospace")
            ax3.text(0.60, y, r1, transform=ax3.transAxes,
                     fontsize=8.5, va="top", color=color, fontweight=weight,
                     fontfamily="monospace")
            y -= 0.105

        self._canvas.draw()

    def _plot_safe_range(self, tc_vals, peaks, safe_flags,
                         hmax: float, ml_tc=None):
        self._fig.clear()
        ax = self._fig.add_subplot(111)

        safe_tc   = tc_vals[safe_flags]
        safe_pk   = peaks[safe_flags]
        unsafe_tc = tc_vals[~safe_flags]
        unsafe_pk = peaks[~safe_flags]

        ax.scatter(safe_tc, safe_pk, color="#2e7d32", zorder=3,
                   label="Safe closure times", s=50)
        ax.scatter(unsafe_tc, unsafe_pk, color="#b71c1c", zorder=3,
                   label="Unsafe closure times", s=50, marker="x")
        ax.plot(tc_vals, peaks, color="#aaa", lw=0.8, zorder=1)
        ax.axhline(hmax, color="#b71c1c", ls="--", lw=1.2,
                   label=f"Hmax = {hmax} m")

        if ml_tc is not None:
            ax.axvline(ml_tc, color="#1565C0", ls="-.", lw=1.5,
                       label=f"ML predicted Tc = {ml_tc} s")

        if len(safe_tc):
            min_safe = safe_tc.min()
            ax.axvline(min_safe, color="#2e7d32", ls=":", lw=1.2,
                       label=f"Min safe Tc = {min_safe:.1f} s")

        ax.set_title("Peak Pressure Head vs Closure Time — Safe Range Scan",
                     fontsize=11)
        ax.set_xlabel("Closure Time Tc (s)", fontsize=10)
        ax.set_ylabel("Peak Head (m)", fontsize=10)
        ax.legend(fontsize=9)
        ax.tick_params(labelsize=9)
        self._canvas.draw()


# ═══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    root = tk.Tk()
    app  = Dashboard(root)
    root.mainloop()
