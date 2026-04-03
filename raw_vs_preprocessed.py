"""
raw_vs_preprocessed.py
======================
Step 3 – Visualization: Raw vs Preprocessed comparison

Plots a single figure with 18 subplots arranged as:
    - 3 rows (one per axis: X, Y, Z)
    - 6 columns (one per dataset: Sachin jump/walk, Ben jump/walk, Christian jump/walk)

Each subplot overlays the raw signal (faded) with the preprocessed (filtered)
signal so the smoothing effect of the moving-average filter is clearly visible.

Run data_storage.py then preprocess.py before this script.
"""

import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')   # non-interactive backend; remove this line if running
                        # interactively (e.g. Jupyter / IDE with display)
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os

# FIX: use local filename rather than an absolute path computed from __file__,
# which broke when the script was called from a different working directory.
HDF5_FILE = 'data_storage.h5'

# Column order: (column_title, hdf5_dataset_name)
COLUMNS = [
    ("Sachin\nJumping",    "sachin_jumping_sweater_pocket"),
    ("Sachin\nWalking",    "sachin_walking_sweater_pocket"),
    ("Ben\nJumping",       "ben_jumping"),
    ("Ben\nWalking",       "ben_walking_outside"),
    ("Christian\nJumping", "christian_jumping_right_hand"),
    ("Christian\nWalking", "christian_walking_left_pocket"),
]

AXES      = ['X', 'Y', 'Z']
AXIS_COLS = [1, 2, 3]          # column indices in stored array: 0=time, 1=x, 2=y, 3=z

RAW_COLOUR   = '#8ab4f8'       # light blue – raw signal
FILT_COLOUR  = '#f28b3b'       # orange – preprocessed signal
BG           = '#1a1a2e'
PANEL_BG     = '#16213e'
GRID_C       = '#2a2a4a'
TEXT_C       = '#e0e0e0'
AXIS_COLOURS = ['#e05c5c', '#5ce08a', '#5c9ee0']   # X=red  Y=green  Z=blue


def load_dataset(f: h5py.File, group: str, name: str) -> np.ndarray:
    """Return (N, 4) array [time, x, y, z] from the given HDF5 group."""
    return f[group][name][:]


def main():
    if not os.path.exists(HDF5_FILE):
        print(f"ERROR: {HDF5_FILE} not found. Run data_storage.py and preprocess.py first.")
        return

    n_rows = len(AXES)      # 3
    n_cols = len(COLUMNS)   # 6

    fig, axes = plt.subplots(
        nrows=n_rows,
        ncols=n_cols,
        figsize=(22, 9),
        sharex=False,
    )

    fig.patch.set_facecolor(BG)
    fig.suptitle(
        "Raw vs Preprocessed Accelerometer Data",
        color=TEXT_C, fontsize=14, fontweight='bold', y=1.01,
    )

    with h5py.File(HDF5_FILE, 'r') as f:

        for col_idx, (col_title, ds_name) in enumerate(COLUMNS):

            raw  = load_dataset(f, 'raw',          ds_name)
            prep = load_dataset(f, 'preprocessed', ds_name)

            t_raw  = raw[:,  0]
            t_prep = prep[:, 0]

            for row_idx, (axis_label, col_num) in enumerate(zip(AXES, AXIS_COLS)):

                ax = axes[row_idx, col_idx]
                ax.set_facecolor(PANEL_BG)

                for spine in ax.spines.values():
                    spine.set_edgecolor(GRID_C)

                ax.tick_params(colors=TEXT_C, labelsize=6)
                ax.grid(True, color=GRID_C, linewidth=0.4,
                        linestyle='--', alpha=0.6)
                ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.0f'))
                ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
                ax.tick_params(axis='x', labelsize=5.5)
                ax.tick_params(axis='y', labelsize=5.5)

                # Raw signal (faded, thin) then preprocessed (solid, thicker)
                ax.plot(t_raw,  raw[:,  col_num],
                        color=RAW_COLOUR,  linewidth=0.5, alpha=0.4, label='Raw')
                ax.plot(t_prep, prep[:, col_num],
                        color=FILT_COLOUR, linewidth=0.8, alpha=0.9, label='Preprocessed')

                if col_idx == 0:
                    ax.set_ylabel(
                        f"{axis_label} (m/s²)",
                        color=AXIS_COLOURS[row_idx],
                        fontsize=8, fontweight='bold',
                    )

                if row_idx == 0:
                    ax.set_title(col_title, color=TEXT_C, fontsize=8,
                                 fontweight='bold', pad=6)

                if row_idx == n_rows - 1:
                    ax.set_xlabel("Time (s)", color=TEXT_C, fontsize=7)

    legend_elements = [
        plt.Line2D([0], [0], color=RAW_COLOUR,  linewidth=1.5,
                   alpha=0.7, label='Raw'),
        plt.Line2D([0], [0], color=FILT_COLOUR, linewidth=1.5,
                   alpha=0.9, label='Preprocessed'),
    ]
    fig.legend(
        handles=legend_elements,
        loc='upper right',
        fontsize=9,
        framealpha=0.3,
        facecolor=PANEL_BG,
        edgecolor=GRID_C,
        labelcolor=TEXT_C,
    )

    plt.tight_layout()
    plt.savefig("raw_vs_preprocessed.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Plot saved to raw_vs_preprocessed.png")


if __name__ == "__main__":
    main()