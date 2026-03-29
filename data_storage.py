import os
import numpy as np
import pandas as pd
import h5py

OUTPUT_FILE = 'dataset.h5'
WINDOW_SECONDS = 5
TRAIN_RATIO = 0.9
RANDOM_SEED = 292

# 1 = jumping, 0 = walking
DATASETS = [
    # Sachin
    ("sachin_jumping_sweater_pocket",
     "ELEC292-Project\\Data\\CSV\\Data_Jumping_SweaterPocket_Sachin.csv",   1),
    ("sachin_walking_sweater_pocket",
     "ELEC292-Project\\Data\\CSV\\Data_Walking_SweaterPocket_Sachin.csv",   0),

    # Ben
    ("ben_jumping",
     "ELEC292-Project\\Data\\CSV\\Data_Jumping_Ben.csv",                    1),
    ("ben_walking_outside",
     "ELEC292-Project\\Data\\CSV\\Data_WalkingOutside_Ben.csv",             0),

    # Christian
    ("christian_jumping_right_hand",
     "ELEC292-Project\\Data\\CSV\\ELEC292_Jumping_RightHand_Data_Christian.csv",  1),
    ("christian_walking_left_pocket",
     "ELEC292-Project\\Data\\CSV\\ELEC292_Walking_LeftPocket_Data_Christian.csv", 0),
]


def load_csv(path: str) -> np.ndarray:
    """Load CSV, drop absolute acceleration column, return (N, 4) float32 array."""
    df = pd.read_csv(path)
    data = df.iloc[:, :4].values.astype(np.float32)
    return data


def sample_rate(data: np.ndarray) -> float:
    """Estimate sample rate in Hz from the time column (column 0)."""
    diffs = np.diff(data[:, 0])
    diffs = diffs[diffs > 0]        # safety check to remove zero/negative gaps
    return float(1.0 / np.median(diffs))


def segment(data: np.ndarray, window_len: int) -> np.ndarray:
    """
    Cut (N, 4) data into non-overlapping windows of length window_len.
    Returns (n_windows, window_len, 4).
    Trailing samples that don't fill a full window are discarded.
    """
    n_windows = len(data) // window_len     # round down so only full windows
    trimmed   = data[: n_windows * window_len]
    return trimmed.reshape(n_windows, window_len, 4)


def main():
    # Delete previous file if it exists so we always start fresh
    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)
        print(f"Deleted existing {OUTPUT_FILE}")

    all_windows = []
    all_labels  = []

    with h5py.File(OUTPUT_FILE, 'w') as f:

        # Create top-level groups
        # Note: preprocessed/ is intentionally not created here.
        #       It is created and populated by preprocess.py (Step 4).
        #       This way the group only exists once real preprocessing has run.
        raw_grp = f.create_group('raw')
        seg_grp = f.create_group('segmented')

        for ds_name, csv_path, label in DATASETS:
            print(f"Loading {ds_name}...")

            # Load raw data
            data       = load_csv(csv_path)
            sr         = sample_rate(data)
            window_len = int(sr * WINDOW_SECONDS)

            # Store raw data
            raw_grp.create_dataset(ds_name, data=data, compression='gzip', compression_opts=4)
            raw_grp[ds_name].attrs['sample_rate_hz']  = sr
            raw_grp[ds_name].attrs['label']           = label
            raw_grp[ds_name].attrs['label_name']      = 'jumping' if label else 'walking'
            raw_grp[ds_name].attrs['columns']         = ['time_s', 'acc_x', 'acc_y', 'acc_z']
            raw_grp[ds_name].attrs['window_len_used'] = window_len

            # Segment into windows for train/test split
            windows = segment(data, window_len)
            all_windows.append(windows)
            all_labels.append(np.full(len(windows), label, dtype=np.int8))

        # Trim all windows to the minimum window length across all files
        # (necessary because different sample rates produce different window lengths)
        min_wl = min(w.shape[1] for w in all_windows)
        trimmed_windows = [w[:, :min_wl, :] for w in all_windows]
        X = np.concatenate(trimmed_windows, axis=0)
        y = np.concatenate(all_labels,      axis=0)

        # Shuffle
        rng     = np.random.default_rng(RANDOM_SEED)
        indices = rng.permutation(len(X))
        X, y    = X[indices], y[indices]

        # Split 90/10
        split           = int(TRAIN_RATIO * len(X))
        X_train, X_test = X[:split],  X[split:]
        y_train, y_test = y[:split],  y[split:]

        # Store segmented data
        seg_grp.create_dataset('train',        data=X_train, compression='gzip')
        seg_grp.create_dataset('train_labels', data=y_train, compression='gzip')
        seg_grp.create_dataset('test',         data=X_test,  compression='gzip')
        seg_grp.create_dataset('test_labels',  data=y_test,  compression='gzip')

        seg_grp.attrs['window_seconds']     = WINDOW_SECONDS
        seg_grp.attrs['window_len_samples'] = min_wl
        seg_grp.attrs['label_encoding']     = '0 = walking,  1 = jumping'
        seg_grp.attrs['train_ratio']        = TRAIN_RATIO
        seg_grp.attrs['random_seed']        = RANDOM_SEED
        seg_grp['train'].attrs['shape_description'] = \
            '(n_windows, window_len, 4)  columns: time, acc_x, acc_y, acc_z'

        print(f"\nTotal windows : {len(X)}")
        print(f"Train windows : {len(X_train)}")
        print(f"Test windows  : {len(X_test)}")
        print(f"\nDataset written to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()