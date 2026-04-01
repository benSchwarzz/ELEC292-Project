"""
visualize_comparison.py
=======================
Step 3 - Visualization: Raw vs Preprocessed comparison

Plots a single figure with 18 subplots arranged as:
    - 3 rows per person (one row per axis: X, Y, Z)
    - 3 columns per person (one column per dataset: jumping, walking)
    - 6 people x 3 axes = 18 graphs total

    Layout (each cell = one subplot):
                  Sachin Jump  |  Sachin Walk  |  Ben Jump  |  Ben Walk  |  Christian Jump  |  Christian Walk
        X axis  |     ...      |      ...      |    ...     |    ...     |       ...        |       ...
        Y axis  |     ...      |      ...      |    ...     |    ...     |       ...        |       ...
        Z axis  |     ...      |      ...      |    ...     |    ...     |       ...        |       ...

Each subplot shows the raw signal (faded) overlaid with the preprocessed
(filtered) signal so the smoothing effect is clearly visible.

Usage
-----
    python visualize_comparison.py

Requires dataset.h5 to have both raw/ and preprocessed/ groups populated.
Run build_dataset.py then preprocess.py before this script.
"""

import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os

BASE = os.path.dirname(os.path.abspath(__file__))
HDF5_FILE = os.path.join(BASE, 'data_storage.h5')

# Order of datasets as columns: (column_title, hdf5_dataset_name)
COLUMNS = [
    ("Sachin\nJumping",  "sachin_jumping_sweater_pocket"),
    ("Sachin\nWalking",  "sachin_walking_sweater_pocket"),
    ("Ben\nJumping",     "ben_jumping"),
    ("Ben\nWalking",     "ben_walking_outside"),
    ("Christian\nJumping", "christian_jumping_right_hand"),
    ("Christian\nWalking", "christian_walking_left_pocket"),
]

AXES       = ['X', 'Y', 'Z']
AXIS_COLS  = [1, 2, 3]   # column indices in the stored array for x, y, z

# Colours
RAW_COLOUR  = '#8ab4f8'   # light blue — raw signal
FILT_COLOUR = '#f28b3b'   # orange — filtered/preprocessed signal
BG          = '#1a1a2e'
PANEL_BG    = '#16213e'
GRID_C      = '#2a2a4a'
TEXT_C      = '#e0e0e0'
AXIS_COLOURS = ['#e05c5c', '#5ce08a', '#5c9ee0']   # X=red  Y=green  Z=blue


def load_dataset(f: h5py.File, group: str, name: str) -> np.ndarray:
    """Return (N, 4) array [time, x, y, z] from the given group."""
    return f[group][name][:]


def main():
    n_rows = len(AXES)       # 3
    n_cols = len(COLUMNS)    # 6

    fig, axes = plt.subplots(
        nrows=n_rows,
        ncols=n_cols,
        figsize=(22, 9),
        sharex=False,   # each dataset has its own time range
    )

    fig.patch.set_facecolor(BG)
    fig.suptitle(
        "Raw vs Preprocessed Accelerometer Data",
        color=TEXT_C, fontsize=14, fontweight='bold', y=1.01,
    )

    with h5py.File(HDF5_FILE, 'r') as f:

        for col_idx, (col_title, ds_name) in enumerate(COLUMNS):

            # Load raw and preprocessed arrays
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
                ax.grid(True, color=GRID_C, linewidth=0.4, linestyle='--', alpha=0.6)
                ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.0f'))
                ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
                ax.tick_params(axis='x', labelsize=5.5)
                ax.tick_params(axis='y', labelsize=5.5)

                # Plot raw (faded, thin) then preprocessed (solid, slightly thicker)
                ax.plot(t_raw,  raw[:,  col_num], color=RAW_COLOUR,
                        linewidth=0.5, alpha=0.4, label='Raw')
                ax.plot(t_prep, prep[:, col_num], color=FILT_COLOUR,
                        linewidth=0.8, alpha=0.9, label='Preprocessed')

                # Y-axis label on leftmost column only
                if col_idx == 0:
                    ax.set_ylabel(
                        f"{axis_label} (m/s²)",
                        color=AXIS_COLOURS[row_idx],
                        fontsize=8, fontweight='bold',
                    )

                # Column title on top row only
                if row_idx == 0:
                    ax.set_title(col_title, color=TEXT_C, fontsize=8,
                                 fontweight='bold', pad=6)

                # X label on bottom row only
                if row_idx == n_rows - 1:
                    ax.set_xlabel("Time (s)", color=TEXT_C, fontsize=7)

    # Single shared legend at the top right of the figure
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
    plt.show()


if __name__ == "__main__":
    main()