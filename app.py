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
from mpl_toolkits.mplot3d import Axes3D 
from scipy.stats import skew, kurtosis
from scipy.interpolate import interp1d
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "final_model.joblib")

C = {
    "bg": "#0d0015",   
    "bg2": "#12001f",   
    "panel": "#1a0030",   
    "border": "#ff2d9b",   
    "accent": "#ff2d9b",   
    "accent2": "#bf00ff",   
    "accent3": "#00f5ff",   
    "text": "#ffe6f5",   
    "text_dim": "#aa7799",  
    "grid": "#2a0040",   
    "raw": "#ff0084",   
    "filt": "#00f5ff",   
    "x_col": "#ff2d9b",
    "y_col": "#bf00ff",
    "z_col":"#00f5ff",
    "btn_bg": "#2d0050",
    "btn_active":"#ff2d9b",
}

FONT_TITLE = ("Courier New", 20, "bold")
FONT_HEAD = ("Courier New", 11, "bold")
FONT_BODY = ("Courier New",  9)
FONT_STAT = ("Courier New", 10, "bold")
FONT_BIG = ("Courier New", 28, "bold")

# Data processing helpers, very similar to the ones used in other files

# get sample rate of dataset based on time
def _get_sr(data: np.ndarray) -> float:
    diffs = np.diff(data[:, 0])
    return float(1.0 / np.median(diffs[diffs > 0]))

# Use a centered moving average to smooth data
def _moving_average(arr: np.ndarray, w: int = 15) -> np.ndarray:
    out = arr.copy()
    for col in range(arr.shape[1]):
        s = pd.Series(arr[:, col])
        out[:, col] = s.rolling(window = w, center = True, min_periods = 1).mean().values
    return out

# remove duplicate rows based on all cols
def _remove_duplicates(data: np.ndarray) -> np.ndarray:
    df = pd.DataFrame(data)
    return df.drop_duplicates().values

# fill gaps in time by linear interpolation, only applies if gap > 3x median gap
def _fill_gaps(data: np.ndarray, gap_thresh: float = 3.0) -> np.ndarray:
    t = data[:, 0]
    diffs = np.diff(t)
    pos = diffs[diffs > 0]
    if len(pos) == 0:
        return data
    mdt = float(np.median(pos))
    gaps = np.where(diffs > gap_thresh * mdt)[0]
    
    if len(gaps) == 0:
        return data
    
    # linear interpolation for gaps
    new_rows = []
    for idx in gaps:
        t0, t1 = t[idx], t[idx + 1]
        n_fill = max(1, round((t1 - t0) / mdt) - 1)
        t_fill = np.linspace(t0, t1, n_fill + 2)[1: -1]
        for tf in t_fill:
            alpha = (tf - t0) / (t1 - t0)
            row = np.zeros(data.shape[1])
            row[0] = tf
            for c in range(1, data.shape[1]):
                row[c] = data[idx, c] + alpha * (data[idx + 1, c] - data[idx, c])
            new_rows.append(row)
    if new_rows:
        data = np.vstack([data, np.array(new_rows)])
        data = data[np.argsort(data[:, 0])]
    return data

# safe skew and kurtosis that return 0 for constant signals instead of NaN, since sklearn doesnt like NaNs in features
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

# feature extraction
def _extract_features(window: np.ndarray) -> np.ndarray:
    feats = []
    for i in range(1, 4):
        s = window[:, i].astype(float)
        feats += [
            np.mean(s), np.std(s), np.var(s),
            np.min(s), np.max(s), np.max(s) - np.min(s),
            np.median(s), _safe_skew(s), _safe_kurtosis(s),
            np.sqrt(np.mean(s ** 2)),]
    return np.array(feats)

# preprocess imported csv
def preprocess_csv(path: str):
    df = pd.read_csv(path)
    df = df.iloc[:, :4]
    df.columns = ["time", "x", "y", "z"]

    raw_arr = df.values.astype(np.float32)
    sr = _get_sr(raw_arr)

    # Preprocessing
    clean = _remove_duplicates(raw_arr)
    clean = _fill_gaps(clean)
    prep_arr = _moving_average(clean, w = 15).astype(np.float32)

    # load model
    clf = joblib.load(MODEL_PATH)

    # Segment and predict
    window_size = int(round(sr * 5))
    acc_data = prep_arr[:, 1:4]
    n_windows   = len(acc_data) // window_size

    # Rebuild 4-col windows (time, x, y, z) so _extract_features works
    prep_4col = prep_arr

    predictions = []
    for i in range(n_windows):
        s   = i * window_size
        e   = s + window_size
        win = prep_4col[s:e]
        feat = _extract_features(win).reshape(1, -1)
        pred = clf.predict(feat)[0]
        predictions.append("jumping" if pred == 1 else "walking")

    return raw_arr, prep_arr, predictions, sr

# main function, runs everything
class ELEC292App(tk.Tk):

    # initialization and GUI setup
    def __init__(self):
        super().__init__()
        self.title("ELEC 292 — Final Project")
        self.configure(bg = C["bg"])
        self.state("zoomed") # start maximized
        self.minsize(1200, 700)

        self.raw_data = None 
        self.prep_data = None 
        self.predictions = []
        self.csv_path = None
        self._anim_idx = 0
        self._anim_job = None
        self._anim_step = 200
        self.scatter3d = [None]

        self._build_ui()

    # UI building methods
    def _build_ui(self):
        # top bar with buttons and title
        top = tk.Frame(self, bg = C["bg"], pady = 6)
        top.pack(side = "top", fill = "x", padx = 16)

        # buttons
        self._btn(top, "^  IMPORT CSV", self._import_csv).pack(side = "left", padx = (0, 12))
        self._btn(top, "v  DOWNLOAD OUTPUT CSV", self._download_csv).pack(side = "left")

        # title in the center
        title_lbl = tk.Label(top, text = "ELEC 292  ·  FINAL PROJECT", bg = C["bg"], fg = C["accent"], font = FONT_TITLE,)
        title_lbl.pack(side="left", expand=True)

        # divider 
        tk.Frame(self, bg = C["border"], height = 2).pack(fill = "x", padx = 0)

        # body
        body = tk.Frame(self, bg = C["bg"])
        body.pack(fill = "both", expand = True, padx = 0, pady = 0)

        # Left data visualizer panel 
        left = tk.Frame(body, bg = C["bg2"], bd = 0)
        left.pack(side = "left", fill = "both", expand = True, padx = (10, 5), pady = 10)
        self._build_left(left)

        # Vertical divider
        tk.Frame(body, bg = C["border"], width = 2).pack(side = "left", fill = "y", pady = 10)

        # Right panel, raw vs preprocessed
        right = tk.Frame(body, bg = C["bg2"], bd = 0)
        right.pack(side = "left", fill = "both", expand = False, padx = (5, 10), pady = 10, ipadx = 4)
        right.configure(width=520)
        right.pack_propagate(False)
        self._build_right(right)

    # buttons
    def _btn(self, parent, text, cmd):
        return tk.Button(
            parent, text=text, command=cmd,
            bg = C["btn_bg"], fg = C["accent"], activebackground = C["accent"],
            activeforeground = C["bg"], font = FONT_HEAD,
            relief = "flat", bd = 0, padx = 14, pady = 6,
            cursor = "hand2",)

    # left panel with 2D and 3D visualizer
    def _build_left(self, parent):
        tk.Label(parent, text = "DATA VISUALIZER", bg = C["bg2"], fg = C["accent2"], font = FONT_HEAD).pack(anchor = "w", padx = 10, pady = (8, 2))

        # Build the matplotlib figure for left panel
        self.fig_left = plt.Figure(figsize = (10, 6), facecolor = C["bg2"])
        self.fig_left.subplots_adjust(left = 0.07, right = 0.97, top = 0.95, bottom = 0.07, hspace = 0.45, wspace = 0.35,)

        # grid configuration
        gs = GridSpec(3, 2, figure = self.fig_left,
                      width_ratios = [2.6, 1.4],
                      hspace = 0.5, wspace = 0.35,
                      left = 0.07, right = 0.97, top = 0.95, bottom = 0.07)

        self.ax_x = self.fig_left.add_subplot(gs[0, 0])
        self.ax_y = self.fig_left.add_subplot(gs[1, 0])
        self.ax_z = self.fig_left.add_subplot(gs[2, 0])
        self.ax_stat = self.fig_left.add_subplot(gs[0, 1])
        self.ax_3d = self.fig_left.add_subplot(gs[1:, 1], projection = "3d")

        self._style_2d_axes()
        self._style_stat_panel()
        self._style_3d_axis()

        self.canvas_left = FigureCanvasTkAgg(self.fig_left, parent)
        self.canvas_left.get_tk_widget().pack(fill = "both", expand = True, padx = 6, pady = 4)
        self.canvas_left.draw()

    # 2D and 3D plot styling
    def _style_2d_axes(self):
        axes_cfg = [(self.ax_x, "X (m/s²)", C["x_col"]), (self.ax_y, "Y (m/s²)", C["y_col"]), (self.ax_z, "Z (m/s²)", C["z_col"]),]
        
        # style each 2D axis
        for ax, label, col in axes_cfg:
            ax.set_facecolor(C["panel"])
            ax.tick_params(colors = C["text_dim"], labelsize = 7)
            ax.set_ylabel(label, color = col, fontsize = 8, fontweight = "bold")
            ax.grid(True, color = C["grid"], linewidth = 0.5, linestyle = "--")
            for sp in ax.spines.values():
                sp.set_edgecolor(C["border"])
                sp.set_linewidth(0.8)
        self.ax_x.set_xticklabels([])
        self.ax_y.set_xticklabels([])
        self.ax_z.set_xlabel("Time (s)", color = C["text_dim"], fontsize = 7)

        self.line_x, = self.ax_x.plot([], [], color = C["x_col"], lw = 0.9, alpha = 0.95)
        self.line_y, = self.ax_y.plot([], [], color = C["y_col"], lw = 0.9, alpha = 0.95)
        self.line_z, = self.ax_z.plot([], [], color = C["z_col"], lw = 0.9, alpha = 0.95)

        self._placeholder_text(self.ax_x, "Import a CSV to begin")
        self._placeholder_text(self.ax_y, "")
        self._placeholder_text(self.ax_z, "")

    # placeholder text in empty plots
    def _placeholder_text(self, ax, msg):
        ax.text(0.5, 0.5, msg, transform = ax.transAxes, ha = "center", va = "center", color = C["text_dim"], fontsize = 8, style = "italic")

    # stats panel styling
    def _style_stat_panel(self):
        self.ax_stat.set_facecolor(C["panel"])
        self.ax_stat.set_xticks([])
        self.ax_stat.set_yticks([])
        for sp in self.ax_stat.spines.values():
            sp.set_edgecolor(C["border"])
            sp.set_linewidth(0.8)
        self.ax_stat.set_title("STATS", color = C["accent"], fontsize = 9, fontweight = "bold", pad = 4)

        self._stat_texts = {}
        y_positions = np.linspace(0.90, 0.05, 12)
        axes_label = ["X", "Y", "Z"]
        colors = [C["x_col"], C["y_col"], C["z_col"]]
        
        # craete text objects for stats
        for i in range(3):
            base = i * 4
            t = self.ax_stat.text(
                0.5, y_positions[base], f"── {axes_label[i]} ──",
                transform = self.ax_stat.transAxes,
                ha = "center", va = "center", fontsize = 8,
                fontweight = "bold", color = colors[i],
            )
            self._stat_texts[f"head_{i}"] = t
            for j, lbl in enumerate(["Min", "Max", "Avg"]):
                t = self.ax_stat.text(
                    0.5, y_positions[base + 1 + j], f"{lbl}: —",
                    transform = self.ax_stat.transAxes,
                    ha = "center", va = "center", fontsize = 7.5, color = C["text"],)
                self._stat_texts[f"{lbl}_{i}"] = t


    def _style_3d_axis(self):
        ax = self.ax_3d
        ax.set_facecolor(C["panel"])
        for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
            pane.fill = False
            pane.set_edgecolor(C["grid"])
        ax.tick_params(colors = C["text_dim"], labelsize = 5)
        ax.set_xlabel("X", color = C["x_col"], fontsize = 7, labelpad = 1)
        ax.set_ylabel("Y", color = C["y_col"], fontsize = 7, labelpad = 1)
        ax.set_zlabel("Z", color = C["z_col"], fontsize = 7, labelpad = 1)
        ax.set_title("3D TRAJECTORY", color = C["accent2"], fontsize = 8, fontweight = "bold", pad = 3)

    # right panel with raw vs preprocessed plots
    def _build_right(self, parent):
        tk.Label(parent, text = "RAW vs. PREPROCESSED", bg = C["bg2"], fg = C["accent2"], font = FONT_HEAD,).pack(anchor = "w", padx = 10, pady = (8, 2))

        # matplotlib figure for right panel
        self.fig_right = plt.Figure(figsize = (5.2, 5.8), facecolor = C["bg2"])
        self.ax_rx = self.fig_right.add_subplot(3, 1, 1)
        self.ax_ry = self.fig_right.add_subplot(3, 1, 2)
        self.ax_rz = self.fig_right.add_subplot(3, 1, 3)
        self.fig_right.subplots_adjust(left=0.14, right=0.97, top=0.94, bottom=0.10, hspace=0.55)

        for ax, label, col in [(self.ax_rx, "X (m/s²)", C["x_col"]), (self.ax_ry, "Y (m/s²)", C["y_col"]),(self.ax_rz, "Z (m/s²)", C["z_col"]),]:
            ax.set_facecolor(C["panel"])
            ax.tick_params(colors = C["text_dim"], labelsize = 6)
            ax.set_ylabel(label, color = col, fontsize = 7, fontweight = "bold")
            ax.grid(True, color = C["grid"], linewidth = 0.4, linestyle = "--")
            for sp in ax.spines.values():
                sp.set_edgecolor(C["border"])
                sp.set_linewidth(0.8)
            ax.text(0.5, 0.5, "—", transform = ax.transAxes, ha = "center", va = "center", color = C["text_dim"], fontsize = 8)

        self.ax_rx.set_xticklabels([])
        self.ax_ry.set_xticklabels([])
        self.ax_rz.set_xlabel("Time (s)", color = C["text_dim"], fontsize = 6)
 
        # Legend patch
        import matplotlib.patches as mpatches
        raw_patch  = mpatches.Patch(color = C["raw"], label = "Raw")
        filt_patch = mpatches.Patch(color = C["filt"], label = "Preprocessed")
        self.fig_right.legend(
            handles = [raw_patch, filt_patch],
            loc = "upper right", fontsize = 7,
            framealpha = 0.25, facecolor = C["panel"],
            edgecolor = C["border"], labelcolor = C["text"],)

        self.canvas_right = FigureCanvasTkAgg(self.fig_right, parent)
        self.canvas_right.get_tk_widget().pack(fill = "both", expand = True, padx = 6, pady = (2, 4))
        self.canvas_right.draw()

        # window counts
        count_frame = tk.Frame(parent, bg = C["panel"], highlightbackground = C["border"], highlightthickness = 1)
        count_frame.pack(fill = "x", padx = 6, pady = (0, 8), ipady = 6)

        self.lbl_jump = tk.Label(count_frame, text = "Jumping Windows:  —",bg = C["panel"], fg = C["accent"], font = FONT_STAT,)
        self.lbl_jump.pack(side = "left", expand = True)

        tk.Frame(count_frame, bg = C["border"], width = 1).pack(side = "left", fill = "y", pady = 4)

        self.lbl_walk = tk.Label(count_frame, text = "Walking Windows:  —", bg = C["panel"], fg = C["accent3"], font = FONT_STAT,)
        self.lbl_walk.pack(side = "left", expand = True)

    # csv import and processing
    def _import_csv(self):
        path = filedialog.askopenfilename(title="Select accelerometer CSV", filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],)
        if not path:
            return

        self.csv_path = path
        self._stop_animation()

        # Run in background thread so UI stays responsive
        self._set_status("Processing…")
        t = threading.Thread(target = self._run_pipeline, args = (path,), daemon = True)
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
        df = pd.DataFrame({ "window": range(1, len(self.predictions) + 1), "label":  self.predictions, })
        df.to_csv(path, index=False)
        messagebox.showinfo("Saved", f"Output saved to:\n{path}")

    # right panel
    def _update_right_panel(self):
        raw = self.raw_data
        prep = self.prep_data

        # Downsample for performance (max 4000 points)
        N = len(raw)
        step = max(1, N // 4000)
        t_r = raw[::step,  0];  t_p  = prep[::step, 0]

        # plot raw vs preprocessed for each axis
        for ax, col_r, col_p in [(self.ax_rx, 1, 1), (self.ax_ry, 2, 2), (self.ax_rz, 3, 3),]:
            ax.cla()
            ax.set_facecolor(C["panel"])
            ax.tick_params(colors = C["text_dim"], labelsize = 6)
            ax.grid(True, color = C["grid"], linewidth = 0.4, linestyle = "--")
            for sp in ax.spines.values():
                sp.set_edgecolor(C["border"])
                sp.set_linewidth(0.8)
            ax.plot(t_r, raw[::step,  col_r], color = C["raw"], lw = 0.6, alpha = 0.45, label = "Raw")
            ax.plot(t_p, prep[::step, col_p], color = C["filt"], lw = 0.9, alpha = 0.9,  label = "Preprocessed")

        self.ax_rx.set_ylabel("X (m/s²)", color = C["x_col"], fontsize = 7, fontweight = "bold")
        self.ax_ry.set_ylabel("Y (m/s²)", color = C["y_col"], fontsize = 7, fontweight = "bold")
        self.ax_rz.set_ylabel("Z (m/s²)", color = C["z_col"], fontsize = 7, fontweight = "bold")
        self.ax_rx.set_xticklabels([])
        self.ax_ry.set_xticklabels([])
        self.ax_rz.set_xlabel("Time (s)", color = C["text_dim"], fontsize = 6)

        self.canvas_right.draw()

    def _update_window_counts(self):
        jump = self.predictions.count("jumping")
        walk = self.predictions.count("walking")
        self.lbl_jump.config(text = f"Jumping Windows: {jump}")
        self.lbl_walk.config(text = f"Walking Windows: {walk}")

    # animation

    def _start_animation(self):
        self._anim_idx = 0
        raw = self.raw_data
        acc = raw[:, 1:4]

        # Fix y-limits from full data so axes don't jump
        for i, ax in enumerate([self.ax_x, self.ax_y, self.ax_z]):
            pad = max(0.5, (acc[:, i].max() - acc[:, i].min()) * 0.15)
            ax.set_ylim(acc[:, i].min() - pad, acc[:, i].max() + pad)

        # Fix 3D limits
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
        t = raw[:end, 0]
        acc = raw[:end, 1:4]

        MAX_PTS = 800

        if len(t) > MAX_PTS:
            s = len(t) // MAX_PTS
            t_plot = t[::s]; a_plot = acc[::s]
        else:
            t_plot = t; a_plot = acc

        # Update 2D lines
        for i, (line, ax) in enumerate(zip(
                [self.line_x, self.line_y, self.line_z], [self.ax_x, self.ax_y, self.ax_z])):
            line.set_data(t_plot, a_plot[:, i])
            x0 = t_plot[0]; x1 = max(t_plot[-1], x0 + 0.5)
            ax.set_xlim(x0, x1)

        # Update stats
        colors = [C["x_col"], C["y_col"], C["z_col"]]
        for i in range(3):
            mn  = acc[:, i].min()
            mx  = acc[:, i].max()
            avg = acc[:, i].mean()
            self._stat_texts[f"Min_{i}"].set_text(f"Min: {mn:+.2f}")
            self._stat_texts[f"Max_{i}"].set_text(f"Max: {mx:+.2f}")
            self._stat_texts[f"Avg_{i}"].set_text(f"Avg: {avg:+.2f}")

        # Update 3D trail
        TRAIL = 300
        t_start = max(0, end - TRAIL)
        x3 = raw[t_start:end, 1]
        y3 = raw[t_start:end, 2]
        z3 = raw[t_start:end, 3]
        cmap = mplcm.get_cmap("plasma")
        n = len(x3)
        cols = cmap(np.linspace(0.15, 1.0, max(n, 1)))
        if self.scatter3d[0] is not None:
            self.scatter3d[0].remove()
        self.scatter3d[0] = self.ax_3d.scatter(x3, y3, z3, c=cols, s=3, alpha=0.8, depthshade=True)

        self.canvas_left.draw()

        self._anim_idx = end
        if end < len(raw):
            self._anim_job = self.after(16, self._tick_animation) 
        else:
            self._anim_job = None

    def _set_status(self, msg):
        if msg:
            self.title(f"ELEC 292 — Final Project  ·  {msg}")
        else:
            self.title("ELEC 292 — Final Project")

if __name__ == "__main__":
    app = ELEC292App()
    app.mainloop()