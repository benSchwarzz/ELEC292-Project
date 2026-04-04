import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
import matplotlib.cm as mplcm

CSV_PATH = "Data/CSV/Data_Jumping_SweaterPocket_Sachin.csv" 
STEP = 500 # samples added/ animation frame
INTERVAL_MS = 10 # ms between frames
VISIBLE_SEC = None # scrolling time window in seconds (None = show all)
MAX_POINTS = 500 # max plotted points per 2-D graph (downsampled if exceeded)
TRAIL_POINTS = 200 # recent points shown on the 3-D scatter

COLOURS = ['#ff2d9b', '#bf00ff', '#00f5ff']
AXES = ['X', 'Y', 'Z']
CMAP = mplcm.get_cmap('plasma') # colour theme for 3D trail
BG = "#0d0015"
GRID_C = "#2a0040"
TEXT_C = "#ffe6f5"

# load data
df = pd.read_csv(CSV_PATH)
time = df.iloc[:, 0].values.astype(np.float32)
acc = df.iloc[:, 1:4].values.astype(np.float32) # columns 1-3: x, y, z, skip col 0 bc its time
N = len(time)

# figure layouy
fig = plt.figure(figsize = (15, 8))
fig.patch.set_facecolor('#12001f')

# customized GridSpec to arrange space for 2D plots, stats panel and 3D scatter plot
gs = GridSpec(
    nrows = 3, ncols = 2,
    figure = fig,
    width_ratios = [3.5, 1.5],
    hspace = 0.45,
    left = 0.06, right = 0.97, top = 0.93, bottom = 0.08,)

# axes and borders
axes = [fig.add_subplot(gs[i, 0]) for i in range(3)]
stat_ax = fig.add_subplot(gs[0, 1])
ax3d = fig.add_subplot(gs[1:, 1], projection = '3d')

for i, ax in enumerate(axes):
    ax.set_facecolor(BG)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_C)
    ax.tick_params(colors = TEXT_C, labelsize = 8)
    ax.set_ylabel(f"{AXES[i]} (m/s²)", color = COLOURS[i], fontsize = 9, fontweight = "bold")
    ax.yaxis.label.set_color(COLOURS[i])
    ax.grid(True, color = GRID_C, linewidth = 0.6, linestyle = "--")
    if i < 2:
        ax.set_xticklabels([])

axes[2].set_xlabel("Time (s)", color = TEXT_C, fontsize = 9, fontweight = "bold")

fig.suptitle(f"Accelerometer — {CSV_PATH.split('/')[-1].split(chr(92))[-1]}", color = TEXT_C, fontsize = 12, fontweight = "bold", y = 0.98,)

# Fixed y axes limits computed from the full dataset so axes don't jump around
for i, ax in enumerate(axes):
    pad = max(0.5, (acc[:, i].max() - acc[:, i].min()) * 0.15)
    ax.set_ylim(acc[:, i].min() - pad, acc[:, i].max() + pad)

# stat panel
stat_ax.set_facecolor(BG)
stat_ax.set_xticks([])
stat_ax.set_yticks([])
for spine in stat_ax.spines.values():
    spine.set_edgecolor(GRID_C)
stat_ax.set_title('LIVE STATS', color = TEXT_C, fontsize = 9, fontweight = "bold", pad = 6)

stat_texts = []
y_pos = np.linspace(0.93, 0.07, 12)

for i in range(3):
    base = i * 4
    stat_texts.append(stat_ax.text(0.5, y_pos[base], f"── {AXES[i]} axis ──",
                     transform = stat_ax.transAxes, ha = "center", va = "center",
                     fontsize = 8, fontweight = "bold", color = COLOURS[i]))
    for j, label in enumerate(["Min", "Max", "Avg"]):
        stat_texts.append(stat_ax.text(0.5, y_pos[base + 1 + j], f"{label}: —",
                         transform = stat_ax.transAxes, ha = "center", va = "center",
                         fontsize = 8, color = TEXT_C))

# 3D scatter plot
ax3d.set_facecolor(BG)
ax3d.xaxis.pane.fill = False
ax3d.yaxis.pane.fill = False
ax3d.zaxis.pane.fill = False
ax3d.xaxis.pane.set_edgecolor(GRID_C)
ax3d.yaxis.pane.set_edgecolor(GRID_C)
ax3d.zaxis.pane.set_edgecolor(GRID_C)
ax3d.tick_params(colors = TEXT_C, labelsize = 6)

# titles and axis labels
ax3d.set_xlabel("X", color = COLOURS[0], fontsize = 8, labelpad = 2)
ax3d.set_ylabel("Y", color = COLOURS[1], fontsize = 8, labelpad = 2)
ax3d.set_zlabel("Z", color = COLOURS[2], fontsize = 8, labelpad = 2)
ax3d.set_title("3D Trajectory", color = TEXT_C, fontsize = 9,fontweight = "bold", pad = 4)

# fixed axes limits, nicer visuals
ax3d.set_xlim(acc[:, 0].min(), acc[:, 0].max())
ax3d.set_ylim(acc[:, 1].min(), acc[:, 1].max())
ax3d.set_zlim(acc[:, 2].min(), acc[:, 2].max())

scatter3d = [None]   # mutable container so update() can replace the scatter object

# create empty line objects for x, y, z axes
lines = []
for i, ax in enumerate(axes):
    line, = ax.plot([], [], color=COLOURS[i], linewidth=0.8, alpha=0.9)
    lines.append(line)

# animation functions

# initializes lines and stats text
def init():
    for line in lines:
        line.set_data([], [])
    return lines + stat_texts

# updates visuals for each animation frame
def update(frame):
    # determine how many samples to show
    end_idx = min((frame + 1) * STEP, N)
    if end_idx == 0:
        return lines + stat_texts

    t_visible = time[ :end_idx]
    t_end     = t_visible[-1]

    # 2D line graphs
    for i, (ax, line) in enumerate(zip(axes, lines)):
        acc_visible = acc[:end_idx, i]

        # scrolling window if VISIBLE_SEC is not None
        if VISIBLE_SEC is not None and t_end > VISIBLE_SEC:
            mask = t_visible >= (t_end - VISIBLE_SEC)
            t_plot = t_visible[mask]
            a_plot = acc_visible[mask]
        else:
            t_plot = t_visible
            a_plot = acc_visible

        # downsample if too many points
        if len(t_plot) > MAX_POINTS:
            s = len(t_plot) // MAX_POINTS
            t_plot = t_plot[::s]
            a_plot = a_plot[::s]

        # update line data
        line.set_data(t_plot, a_plot)

        # dynamically update x-axis limits
        x_min = t_plot[0]
        x_max = max(t_plot[-1], t_plot[0] + 0.5)
        if ax.get_xlim() != (x_min, x_max):
            ax.set_xlim(x_min, x_max)

        # live stats: min, max, avg for visible data
        acc_all = acc[:end_idx, i]
        mn, mx, avg = acc_all.min(), acc_all.max(), acc_all.mean()
        base = i * 4
        stat_texts[base + 1].set_text(f"Min: {mn:+.2f}")
        stat_texts[base + 2].set_text(f"Max: {mx:+.2f}")
        stat_texts[base + 3].set_text(f"Avg: {avg:+.2f}")

    # 3D trail — oldest points dark, newest points bright
    trail_start = max(0, end_idx - TRAIL_POINTS)
    x3 = acc[trail_start:end_idx, 0]
    y3 = acc[trail_start:end_idx, 1]
    z3 = acc[trail_start:end_idx, 2]
    n_trail = len(x3)
    colours = CMAP(np.linspace(0.15, 1.0, n_trail))

    # remove old scatter and plot new one
    if scatter3d[0] is not None:
        scatter3d[0].remove()

    scatter3d[0] = ax3d.scatter(x3, y3, z3,c=colours, s=4, alpha=0.7, depthshade=True,)

    return lines + stat_texts


n_frames = int(np.ceil(N / STEP))
ani = animation.FuncAnimation(
    fig, update,
    frames=n_frames,
    init_func=init,
    blit=False,
    interval=INTERVAL_MS,
    repeat=False,
)

plt.show()