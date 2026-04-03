"""
ELEC 292 — Final Project Desktop Application
Step 7: Deploy trained classifier in a GUI

Layout (matches wireframe):
  Left panel  — Data Visualizer
    • X / Y / Z animated time-series plots
    • Stats panel (live min/max/avg per axis)
    • 3-D scatter trajectory
  Right panel — Raw vs. Preprocessed
    • X / Y / Z overlay plots
    • Jumping / Walking window counts

Controls (top bar):
  [Import CSV]  [Download Output CSV]

Theme: Pink Cyberpunk
"""

import os, sys, threading, tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.cm as mplcm
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401  (registers 3-D projection)
from scipy.stats import skew, kurtosis
from scipy.interpolate import interp1d
import joblib

# ---------------------------------------------------------------------------
# ── PATHS  (all relative to wherever app.py lives) ──────────────────────────
# ---------------------------------------------------------------------------
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH  = os.path.join(BASE_DIR, "final_model.joblib")

# ---------------------------------------------------------------------------
# ── THEME ───────────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------
C = {
    "bg":          "#0d0015",   # near-black purple-black
    "bg2":         "#12001f",   # panel background
    "panel":       "#1a0030",   # raised panel
    "border":      "#ff2d9b",   # hot pink border
    "accent":      "#ff2d9b",   # hot pink
    "accent2":     "#bf00ff",   # electric violet
    "accent3":     "#00f5ff",   # cyan pop
    "text":        "#ffe6f5",   # soft white-pink
    "text_dim":    "#aa7799",   # dimmed label
    "grid":        "#2a0040",   # plot grid lines
    "raw":         "#ff0084",   # raw signal — hot pink
    "filt":        "#00f5ff",   # filtered signal — cyan
    "x_col":       "#ff2d9b",
    "y_col":       "#bf00ff",
    "z_col":       "#00f5ff",
    "btn_bg":      "#2d0050",
    "btn_active":  "#ff2d9b",
}

FONT_TITLE  = ("Courier New", 20, "bold")
FONT_HEAD   = ("Courier New", 11, "bold")
FONT_BODY   = ("Courier New",  9)
FONT_STAT   = ("Courier New", 10, "bold")
FONT_BIG    = ("Courier New", 28, "bold")

# ---------------------------------------------------------------------------
# ── SIGNAL PROCESSING HELPERS  (mirrors the training pipeline) ───────────────
# ---------------------------------------------------------------------------

def _get_sr(data: np.ndarray) -> float:
    diffs = np.diff(data[:, 0])
    return float(1.0 / np.median(diffs[diffs > 0]))


def _moving_average(arr: np.ndarray, w: int = 15) -> np.ndarray:
    out = arr.copy()
    for col in range(arr.shape[1]):
        s = pd.Series(arr[:, col])
        out[:, col] = s.rolling(window=w, center=True, min_periods=1).mean().values
    return out


def _remove_duplicates(data: np.ndarray) -> np.ndarray:
    df = pd.DataFrame(data)
    return df.drop_duplicates().values


def _fill_gaps(data: np.ndarray, gap_thresh: float = 3.0) -> np.ndarray:
    t      = data[:, 0]
    diffs  = np.diff(t)
    pos    = diffs[diffs > 0]
    if len(pos) == 0:
        return data
    mdt    = float(np.median(pos))
    gaps   = np.where(diffs > gap_thresh * mdt)[0]
    if len(gaps) == 0:
        return data
    new_rows = []
    for idx in gaps:
        t0, t1   = t[idx], t[idx + 1]
        n_fill   = max(1, round((t1 - t0) / mdt) - 1)
        t_fill   = np.linspace(t0, t1, n_fill + 2)[1:-1]
        for tf in t_fill:
            alpha = (tf - t0) / (t1 - t0)
            row   = np.zeros(data.shape[1])
            row[0] = tf
            for c in range(1, data.shape[1]):
                row[c] = data[idx, c] + alpha * (data[idx + 1, c] - data[idx, c])
            new_rows.append(row)
    if new_rows:
        data = np.vstack([data, np.array(new_rows)])
        data = data[np.argsort(data[:, 0])]
    return data


def _safe_skew(s):
    if np.allclose(s, s[0], atol=1e-8):
        return 0.0
    v = skew(s)
    return 0.0 if np.isnan(v) else float(v)


def _safe_kurtosis(s):
    if np.allclose(s, s[0], atol=1e-8):
        return 0.0
    v = kurtosis(s)
    return 0.0 if np.isnan(v) else float(v)


def _extract_features(window: np.ndarray) -> np.ndarray:
    """window shape: (500, 4)  cols=[time, x, y, z]"""
    feats = []
    for i in range(1, 4):        # columns 1=x  2=y  3=z
        s = window[:, i].astype(float)
        feats += [
            np.mean(s), np.std(s), np.var(s),
            np.min(s),  np.max(s), np.max(s) - np.min(s),
            np.median(s),
            _safe_skew(s), _safe_kurtosis(s),
            np.sqrt(np.mean(s ** 2)),
        ]
    return np.array(feats)


def preprocess_csv(path: str):
    """
    Load a CSV, preprocess, segment, classify.

    Returns
    -------
    raw_data   : (N, 4) float32  [time, x, y, z]   original signal
    prep_data  : (N, 4) float32  [time, x, y, z]   filtered signal
    predictions: list of str     per-window labels
    sr         : float           detected sample rate
    """
    df = pd.read_csv(path)
    df = df.iloc[:, :4]
    df.columns = ["time", "x", "y", "z"]

    raw_arr  = df.values.astype(np.float32)
    sr       = _get_sr(raw_arr)

    # Preprocessing steps
    clean   = _remove_duplicates(raw_arr)
    clean   = _fill_gaps(clean)
    prep_arr = _moving_average(clean, w=15).astype(np.float32)

    # Load model
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}.\n"
            "Run train_model.py first to generate final_model.joblib."
        )
    clf = joblib.load(MODEL_PATH)

    # Segment and predict
    window_size = int(round(sr * 5))
    acc_data    = prep_arr[:, 1:4]          # (N, 3)  x/y/z only
    n_windows   = len(acc_data) // window_size

    # Rebuild 4-col windows (time, x, y, z) so _extract_features works
    prep_4col = prep_arr                    # still (N, 4)

    predictions = []
    for i in range(n_windows):
        s   = i * window_size
        e   = s + window_size
        win = prep_4col[s:e]                # (window_size, 4)
        feat = _extract_features(win).reshape(1, -1)
        pred = clf.predict(feat)[0]
        predictions.append("jumping" if pred == 1 else "walking")

    return raw_arr, prep_arr, predictions, sr


# ---------------------------------------------------------------------------
# ── MAIN APPLICATION ─────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

class ELEC292App(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("ELEC 292 — Final Project")
        self.configure(bg=C["bg"])
        self.state("zoomed")          # start maximised
        self.minsize(1200, 700)

        # State
        self.raw_data    = None       # (N,4)
        self.prep_data   = None       # (N,4)
        self.predictions = []
        self.csv_path    = None
        self._anim_idx   = 0
        self._anim_job   = None
        self._anim_step  = 200        # samples per frame
        self.scatter3d   = [None]

        self._build_ui()

    # ------------------------------------------------------------------ UI --

    def _build_ui(self):
        # ── Top bar ──────────────────────────────────────────────────────────
        top = tk.Frame(self, bg=C["bg"], pady=6)
        top.pack(side="top", fill="x", padx=16)

        self._btn(top, "⬆  IMPORT CSV",          self._import_csv).pack(side="left", padx=(0, 12))
        self._btn(top, "⬇  DOWNLOAD OUTPUT CSV", self._download_csv).pack(side="left")

        title_lbl = tk.Label(
            top, text="ELEC 292  ·  FINAL PROJECT",
            bg=C["bg"], fg=C["accent"], font=FONT_TITLE,
        )
        title_lbl.pack(side="left", expand=True)

        # ── Divider ──────────────────────────────────────────────────────────
        tk.Frame(self, bg=C["border"], height=2).pack(fill="x", padx=0)

        # ── Main body ────────────────────────────────────────────────────────
        body = tk.Frame(self, bg=C["bg"])
        body.pack(fill="both", expand=True, padx=0, pady=0)

        # Left panel (60 %) — Data Visualizer
        left = tk.Frame(body, bg=C["bg2"], bd=0)
        left.pack(side="left", fill="both", expand=True, padx=(10, 5), pady=10)
        self._build_left(left)

        # Vertical divider
        tk.Frame(body, bg=C["border"], width=2).pack(side="left", fill="y", pady=10)

        # Right panel (40 %) — Raw vs Preprocessed
        right = tk.Frame(body, bg=C["bg2"], bd=0)
        right.pack(side="left", fill="both", expand=False, padx=(5, 10), pady=10,
                   ipadx=4)
        right.configure(width=520)
        right.pack_propagate(False)
        self._build_right(right)

    def _btn(self, parent, text, cmd):
        return tk.Button(
            parent, text=text, command=cmd,
            bg=C["btn_bg"], fg=C["accent"], activebackground=C["accent"],
            activeforeground=C["bg"], font=FONT_HEAD,
            relief="flat", bd=0, padx=14, pady=6,
            cursor="hand2",
        )

    # ── Left panel ──────────────────────────────────────────────────────────

    def _build_left(self, parent):
        tk.Label(parent, text="DATA VISUALIZER",
                 bg=C["bg2"], fg=C["accent2"], font=FONT_HEAD
                 ).pack(anchor="w", padx=10, pady=(8, 2))

        # Build the matplotlib figure for left panel
        self.fig_left = plt.Figure(figsize=(10, 6), facecolor=C["bg2"])
        self.fig_left.subplots_adjust(
            left=0.07, right=0.97, top=0.95, bottom=0.07,
            hspace=0.45, wspace=0.35,
        )

        gs = GridSpec(3, 2, figure=self.fig_left,
                      width_ratios=[2.6, 1.4],
                      hspace=0.5, wspace=0.35,
                      left=0.07, right=0.97, top=0.95, bottom=0.07)

        self.ax_x    = self.fig_left.add_subplot(gs[0, 0])
        self.ax_y    = self.fig_left.add_subplot(gs[1, 0])
        self.ax_z    = self.fig_left.add_subplot(gs[2, 0])
        self.ax_stat = self.fig_left.add_subplot(gs[0, 1])
        self.ax_3d   = self.fig_left.add_subplot(gs[1:, 1], projection="3d")

        self._style_2d_axes()
        self._style_stat_panel()
        self._style_3d_axis()

        self.canvas_left = FigureCanvasTkAgg(self.fig_left, parent)
        self.canvas_left.get_tk_widget().pack(fill="both", expand=True,
                                               padx=6, pady=4)
        self.canvas_left.draw()

    def _style_2d_axes(self):
        axes_cfg = [
            (self.ax_x, "X (m/s²)", C["x_col"]),
            (self.ax_y, "Y (m/s²)", C["y_col"]),
            (self.ax_z, "Z (m/s²)", C["z_col"]),
        ]
        for ax, label, col in axes_cfg:
            ax.set_facecolor(C["panel"])
            ax.tick_params(colors=C["text_dim"], labelsize=7)
            ax.set_ylabel(label, color=col, fontsize=8, fontweight="bold")
            ax.grid(True, color=C["grid"], linewidth=0.5, linestyle="--")
            for sp in ax.spines.values():
                sp.set_edgecolor(C["border"])
                sp.set_linewidth(0.8)
        self.ax_x.set_xticklabels([])
        self.ax_y.set_xticklabels([])
        self.ax_z.set_xlabel("Time (s)", color=C["text_dim"], fontsize=7)

        self.line_x, = self.ax_x.plot([], [], color=C["x_col"], lw=0.9, alpha=0.95)
        self.line_y, = self.ax_y.plot([], [], color=C["y_col"], lw=0.9, alpha=0.95)
        self.line_z, = self.ax_z.plot([], [], color=C["z_col"], lw=0.9, alpha=0.95)

        self._placeholder_text(self.ax_x, "Import a CSV to begin")
        self._placeholder_text(self.ax_y, "")
        self._placeholder_text(self.ax_z, "")

    def _placeholder_text(self, ax, msg):
        ax.text(0.5, 0.5, msg, transform=ax.transAxes,
                ha="center", va="center", color=C["text_dim"],
                fontsize=8, style="italic")

    def _style_stat_panel(self):
        self.ax_stat.set_facecolor(C["panel"])
        self.ax_stat.set_xticks([])
        self.ax_stat.set_yticks([])
        for sp in self.ax_stat.spines.values():
            sp.set_edgecolor(C["border"])
            sp.set_linewidth(0.8)
        self.ax_stat.set_title("STATS", color=C["accent"], fontsize=9,
                                fontweight="bold", pad=4)

        self._stat_texts = {}
        y_positions = np.linspace(0.90, 0.05, 12)
        axes_label  = ["X", "Y", "Z"]
        colors      = [C["x_col"], C["y_col"], C["z_col"]]
        for i in range(3):
            base = i * 4
            t = self.ax_stat.text(
                0.5, y_positions[base], f"── {axes_label[i]} ──",
                transform=self.ax_stat.transAxes,
                ha="center", va="center", fontsize=8,
                fontweight="bold", color=colors[i],
            )
            self._stat_texts[f"head_{i}"] = t
            for j, lbl in enumerate(["Min", "Max", "Avg"]):
                t = self.ax_stat.text(
                    0.5, y_positions[base + 1 + j], f"{lbl}: —",
                    transform=self.ax_stat.transAxes,
                    ha="center", va="center", fontsize=7.5,
                    color=C["text"],
                )
                self._stat_texts[f"{lbl}_{i}"] = t

    def _style_3d_axis(self):
        ax = self.ax_3d
        ax.set_facecolor(C["panel"])
        for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
            pane.fill = False
            pane.set_edgecolor(C["grid"])
        ax.tick_params(colors=C["text_dim"], labelsize=5)
        ax.set_xlabel("X", color=C["x_col"], fontsize=7, labelpad=1)
        ax.set_ylabel("Y", color=C["y_col"], fontsize=7, labelpad=1)
        ax.set_zlabel("Z", color=C["z_col"], fontsize=7, labelpad=1)
        ax.set_title("3D TRAJECTORY", color=C["accent2"],
                     fontsize=8, fontweight="bold", pad=3)

    # ── Right panel ─────────────────────────────────────────────────────────

    def _build_right(self, parent):
        tk.Label(parent, text="RAW vs. PREPROCESSED",
                 bg=C["bg2"], fg=C["accent2"], font=FONT_HEAD,
                 ).pack(anchor="w", padx=10, pady=(8, 2))

        # matplotlib figure for right panel
        self.fig_right = plt.Figure(figsize=(5.2, 5.8), facecolor=C["bg2"])

        self.ax_rx = self.fig_right.add_subplot(3, 1, 1)
        self.ax_ry = self.fig_right.add_subplot(3, 1, 2)
        self.ax_rz = self.fig_right.add_subplot(3, 1, 3)
        self.fig_right.subplots_adjust(
            left=0.14, right=0.97, top=0.94, bottom=0.10, hspace=0.55
        )

        for ax, label, col in [
            (self.ax_rx, "X (m/s²)", C["x_col"]),
            (self.ax_ry, "Y (m/s²)", C["y_col"]),
            (self.ax_rz, "Z (m/s²)", C["z_col"]),
        ]:
            ax.set_facecolor(C["panel"])
            ax.tick_params(colors=C["text_dim"], labelsize=6)
            ax.set_ylabel(label, color=col, fontsize=7, fontweight="bold")
            ax.grid(True, color=C["grid"], linewidth=0.4, linestyle="--")
            for sp in ax.spines.values():
                sp.set_edgecolor(C["border"])
                sp.set_linewidth(0.8)
            ax.text(0.5, 0.5, "—", transform=ax.transAxes,
                    ha="center", va="center", color=C["text_dim"], fontsize=8)

        self.ax_rx.set_xticklabels([])
        self.ax_ry.set_xticklabels([])
        self.ax_rz.set_xlabel("Time (s)", color=C["text_dim"], fontsize=6)

        # Legend patch
        import matplotlib.patches as mpatches
        raw_patch  = mpatches.Patch(color=C["raw"],  label="Raw")
        filt_patch = mpatches.Patch(color=C["filt"], label="Preprocessed")
        self.fig_right.legend(
            handles=[raw_patch, filt_patch],
            loc="upper right", fontsize=7,
            framealpha=0.25, facecolor=C["panel"],
            edgecolor=C["border"], labelcolor=C["text"],
        )

        self.canvas_right = FigureCanvasTkAgg(self.fig_right, parent)
        self.canvas_right.get_tk_widget().pack(fill="both", expand=True,
                                                padx=6, pady=(2, 4))
        self.canvas_right.draw()

        # ── Window count bar ─────────────────────────────────────────────
        count_frame = tk.Frame(parent, bg=C["panel"],
                               highlightbackground=C["border"],
                               highlightthickness=1)
        count_frame.pack(fill="x", padx=6, pady=(0, 8), ipady=6)

        self.lbl_jump = tk.Label(
            count_frame,
            text="Jumping Windows:  —",
            bg=C["panel"], fg=C["accent"], font=FONT_STAT,
        )
        self.lbl_jump.pack(side="left", expand=True)

        tk.Frame(count_frame, bg=C["border"], width=1).pack(
            side="left", fill="y", pady=4)

        self.lbl_walk = tk.Label(
            count_frame,
            text="Walking Windows:  —",
            bg=C["panel"], fg=C["accent3"], font=FONT_STAT,
        )
        self.lbl_walk.pack(side="left", expand=True)

    # ---------------------------------------------------------------- Actions

    def _import_csv(self):
        path = filedialog.askopenfilename(
            title="Select accelerometer CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if not path:
            return

        self.csv_path = path
        self._stop_animation()

        # Run in background thread so UI stays responsive
        self._set_status("Processing…")
        t = threading.Thread(target=self._run_pipeline, args=(path,), daemon=True)
        t.start()

    def _run_pipeline(self, path):
        try:
            raw, prep, preds, sr = preprocess_csv(path)
            # Schedule UI update on main thread
            self.after(0, lambda: self._on_pipeline_done(raw, prep, preds, sr))
        except Exception as exc:
            self.after(0, lambda: messagebox.showerror("Error", str(exc)))

    def _on_pipeline_done(self, raw, prep, preds, sr):
        self.raw_data    = raw
        self.prep_data   = prep
        self.predictions = preds
        self._set_status("")

        self._update_right_panel()
        self._update_window_counts()
        self._start_animation()

    def _download_csv(self):
        if not self.predictions:
            messagebox.showwarning("No data", "Import and process a CSV first.")
            return
        path = filedialog.asksaveasfilename(
            title="Save output CSV",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
            initialfile="output.csv",
        )
        if not path:
            return
        df = pd.DataFrame({
            "window": range(1, len(self.predictions) + 1),
            "label":  self.predictions,
        })
        df.to_csv(path, index=False)
        messagebox.showinfo("Saved", f"Output saved to:\n{path}")

    # -------------------------------------------------------------- Right panel

    def _update_right_panel(self):
        raw  = self.raw_data
        prep = self.prep_data

        # Downsample for performance (max 4000 points)
        N    = len(raw)
        step = max(1, N // 4000)
        t_r  = raw[::step,  0];  t_p  = prep[::step, 0]

        for ax, col_r, col_p in [
            (self.ax_rx, 1, 1),
            (self.ax_ry, 2, 2),
            (self.ax_rz, 3, 3),
        ]:
            ax.cla()
            ax.set_facecolor(C["panel"])
            ax.tick_params(colors=C["text_dim"], labelsize=6)
            ax.grid(True, color=C["grid"], linewidth=0.4, linestyle="--")
            for sp in ax.spines.values():
                sp.set_edgecolor(C["border"])
                sp.set_linewidth(0.8)
            ax.plot(t_r, raw[::step,  col_r], color=C["raw"],
                    lw=0.6, alpha=0.45, label="Raw")
            ax.plot(t_p, prep[::step, col_p], color=C["filt"],
                    lw=0.9, alpha=0.9,  label="Preprocessed")

        self.ax_rx.set_ylabel("X (m/s²)", color=C["x_col"], fontsize=7, fontweight="bold")
        self.ax_ry.set_ylabel("Y (m/s²)", color=C["y_col"], fontsize=7, fontweight="bold")
        self.ax_rz.set_ylabel("Z (m/s²)", color=C["z_col"], fontsize=7, fontweight="bold")
        self.ax_rx.set_xticklabels([])
        self.ax_ry.set_xticklabels([])
        self.ax_rz.set_xlabel("Time (s)", color=C["text_dim"], fontsize=6)

        self.canvas_right.draw()

    def _update_window_counts(self):
        jump = self.predictions.count("jumping")
        walk = self.predictions.count("walking")
        self.lbl_jump.config(text=f"Jumping Windows:  {jump}")
        self.lbl_walk.config(text=f"Walking Windows:  {walk}")

    # -------------------------------------------------------------- Animation

    def _start_animation(self):
        self._anim_idx = 0
        raw = self.raw_data
        acc = raw[:, 1:4]

        # Fix y-limits from full data so axes don't jump
        for i, ax in enumerate([self.ax_x, self.ax_y, self.ax_z]):
            pad = max(0.5, (acc[:, i].max() - acc[:, i].min()) * 0.15)
            ax.set_ylim(acc[:, i].min() - pad, acc[:, i].max() + pad)

        # Fix 3-D limits
        self.ax_3d.set_xlim(acc[:, 0].min(), acc[:, 0].max())
        self.ax_3d.set_ylim(acc[:, 1].min(), acc[:, 1].max())
        self.ax_3d.set_zlim(acc[:, 2].min(), acc[:, 2].max())

        self._anim_step = max(50, len(raw) // 300)   # adaptive speed
        self.scatter3d  = [None]
        self._tick_animation()

    def _stop_animation(self):
        if self._anim_job:
            self.after_cancel(self._anim_job)
            self._anim_job = None

    def _tick_animation(self):
        raw = self.raw_data
        if raw is None:
            return

        end = min(self._anim_idx + self._anim_step, len(raw))
        t   = raw[:end, 0]
        acc = raw[:end, 1:4]

        MAX_PTS = 800
        if len(t) > MAX_PTS:
            s = len(t) // MAX_PTS
            t_plot = t[::s]; a_plot = acc[::s]
        else:
            t_plot = t; a_plot = acc

        # Update 2-D lines
        for i, (line, ax) in enumerate(zip(
                [self.line_x, self.line_y, self.line_z],
                [self.ax_x,   self.ax_y,   self.ax_z])):
            line.set_data(t_plot, a_plot[:, i])
            x0 = t_plot[0]; x1 = max(t_plot[-1], x0 + 0.5)
            ax.set_xlim(x0, x1)

        # Update stats
        colors = [C["x_col"], C["y_col"], C["z_col"]]
        for i in range(3):
            mn  = acc[:, i].min()
            mx  = acc[:, i].max()
            avg = acc[:, i].mean()
            self._stat_texts[f"Min_{i}"].set_text(f"Min:  {mn:+.2f}")
            self._stat_texts[f"Max_{i}"].set_text(f"Max: {mx:+.2f}")
            self._stat_texts[f"Avg_{i}"].set_text(f"Avg:  {avg:+.2f}")

        # Update 3-D trail
        TRAIL = 300
        t_start = max(0, end - TRAIL)
        x3 = raw[t_start:end, 1]
        y3 = raw[t_start:end, 2]
        z3 = raw[t_start:end, 3]
        cmap   = mplcm.get_cmap("plasma")
        n      = len(x3)
        cols   = cmap(np.linspace(0.15, 1.0, max(n, 1)))
        if self.scatter3d[0] is not None:
            self.scatter3d[0].remove()
        self.scatter3d[0] = self.ax_3d.scatter(
            x3, y3, z3, c=cols, s=3, alpha=0.8, depthshade=True
        )

        self.canvas_left.draw()

        self._anim_idx = end
        if end < len(raw):
            self._anim_job = self.after(16, self._tick_animation)   # ~60 fps
        else:
            self._anim_job = None

    # ----------------------------------------------------------------- Helpers

    def _set_status(self, msg):
        """Flash a brief message in the title bar."""
        if msg:
            self.title(f"ELEC 292 — Final Project  ·  {msg}")
        else:
            self.title("ELEC 292 — Final Project")


# ---------------------------------------------------------------------------
# ── ENTRY POINT ──────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app = ELEC292App()
    app.mainloop()