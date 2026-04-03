import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec

CSV_PATH = "ELEC292-Project\\Data\\CSV\\Data_Walking_SweaterPocket_Sachin.csv"
STEP = 50 # number of samples added per frame
INTERVAL_MS = 50 # ms bw frames
VISIBLE_SEC = 10
MAX_POINTS = 5000 # max points to show on graph at once, should improve performance


COLOURS = ['r', 'g', 'b']
AXES = ['X', 'Y', 'Z']

# load data
df = pd.read_csv(CSV_PATH)
time = df.iloc[:, 0].values.astype(np.float32)
acc = df.iloc[:, 1:4].values.astype(np.float32)

N = len(time)

# 3 graphs on left, 1 on right
fig  = plt.figure(figsize=(13, 7))
fig.patch.set_facecolor('#1a1a2e')

gs = GridSpec(nrows = 3, ncols = 2, figure = fig, width_ratios = [4, 1], hspace = 0.45, 
              left = 0.07, right = 0.97, top = 0.95, bottom = 0.08)

axes = [fig.add_subplot(gs[i, 0]) for i in range(3)]
stat_ax = fig.add_subplot(gs[:, 1])

BG      = "#16213e"
GRID_C  = "#2a2a4a"
TEXT_C  = "#e0e0e0"

for i, ax in enumerate(axes):
    ax.set_facecolor(BG)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_C)
    ax.tick_params(colors = TEXT_C, labelsize = 8)
    ax.set_ylabel(f"{AXES[i]} (m/s²)", color=COLOURS[i], fontsize = 9, fontweight = "bold")
    ax.yaxis.label.set_color(COLOURS[i])
    ax.grid(True, color = GRID_C, linewidth = 0.6, linestyle = "--")

    if i < 2:
        ax.set_xticklabels([]) # only show x labels on bottom graph
    
axes[2].set_xlabel("Time (s)", color=TEXT_C, fontsize = 9, fontweight = "bold")

fig.suptitle(f"Accelerometer — {CSV_PATH.split('\\')[-1]}", color = TEXT_C, fontsize = 12, fontweight = "bold", y = 0.97)


# stat panel on right
stat_ax.set_facecolor(BG)
stat_ax.set_xticks([])
stat_ax.set_yticks([])

for spine in stat_ax.spines.values():
    spine.set_edgecolor(GRID_C)

stat_ax.set_title('LIVE STATS', color = TEXT_C, fontsize = 9, fontweight = "bold", pad = 8)

stat_texts = []

y_pos = np.linspace(0.93, 0.07, 12)

for i in range(3):
    base = i * 4
    stat_texts.append(stat_ax.text(0.5, y_pos[base], f"── {AXES[i]} axis ──",
                     transform = stat_ax.transAxes,
                     ha = "center", va = "center",
                     fontsize = 8, fontweight = "bold", color = COLOURS[i]))
    
    # min/max.avg
    for j, label in enumerate(["Min", "Max", "Avg"]):
        stat_texts.append(
            stat_ax.text(0.5, y_pos[base + 1 + j], f"{label}: —",
                         transform=stat_ax.transAxes,
                         ha="center", va="center",
                         fontsize=8, color=TEXT_C))

# create line objects for each axis
lines = []

for i, ax in enumerate(axes):
    line, = ax.plot([], [], color = COLOURS[i], linewidth = 0.8, alpha = 0.9) # , unpacks a single item of the list
    lines.append(line)

# animation
def init():
    for line in lines:
        line.set_data([], [])
    return lines + stat_texts

def update(frame):
    end_idx = min((frame + 1) * STEP, N)
    
    if end_idx == 0:
        return lines + stat_texts
    
    t_visible = time[:end_idx]
    t_end = t_visible[-1]

    for i, (ax, line) in enumerate(zip(axes, lines)):
        acc_visible = acc[:end_idx, i]
        
        if VISIBLE_SEC is not None and t_end > VISIBLE_SEC:
            t_start = t_end - VISIBLE_SEC
            mask = t_visible >= t_start
            t_plot = t_visible[mask]
            acc_plot = acc_visible[mask]
        else:
            t_plot = t_visible
            acc_plot = acc_visible

        if len(t_plot) > MAX_POINTS:
            step = len(t_plot) // MAX_POINTS
            t_plot = t_plot[::step]
            acc_plot = acc_plot[::step]
        
        line.set_data(t_plot, acc_plot)

        # Rescale axes
        x_min = t_plot[0]
        x_max = max(t_plot[-1], t_plot[0] + 0.5)
        if ax.get_xlim() != (x_min, x_max):
            ax.set_xlim(x_min, x_max)
        pad = max(0.5, (acc_plot.max() - acc_plot.min()) * 0.15)

        # update stats w/ all data up to current frame
        acc_all = acc[:end_idx, i]
        mn, mx, avg = acc_all.min(), acc_all.max(), acc_all.mean()
        base = i * 4

        stat_texts[base + 1].set_text(f"Min:  {mn:+.2f}")
        stat_texts[base + 2].set_text(f"Max: {mx:+.2f}")
        stat_texts[base + 3].set_text(f"Avg:  {avg:+.2f}")

    return lines + stat_texts

# before the animation, set fixed y limits
for i, ax in enumerate(axes):
    pad = max(0.5, (acc[:, i].max() - acc[:, i].min()) * 0.15)
    ax.set_ylim(acc[:, i].min() - pad, acc[:, i].max() + pad)

n_frames = int(np.ceil(N / STEP))
ani = animation.FuncAnimation(fig, update, frames = n_frames, init_func = init, blit = False, interval = INTERVAL_MS, repeat = False)
plt.show()