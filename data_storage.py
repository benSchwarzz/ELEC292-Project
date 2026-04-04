import numpy as np
import pandas as pd
import h5py
from scipy.interpolate import interp1d

OUTPUT_FILE = 'data_storage.h5'
WINDOW_SECONDS = 5
TARGET_SR = 100.0                           # FIX: resample all datasets to a common 100 Hz
WINDOW_LEN = int(TARGET_SR * WINDOW_SECONDS)  # 500 samples per window
TRAIN_RATIO = 0.9
RANDOM_SEED = 292

# 1 = jumping, 0 = walking
DATASETS = [
    # Sachin
    ("sachin_jumping_sweater_pocket",
     "Data_Jumping_SweaterPocket_Sachin.csv",      1),
    ("sachin_walking_sweater_pocket",
     "Data_Walking_SweaterPocket_Sachin.csv",      0),

    # Ben
    ("ben_jumping",
     "Data_Jumping_Ben.csv",                       1),
    ("ben_walking_outside",
     "Data_WalkingOutside_Ben.csv",                0),

    # Christian
    ("christian_jumping_right_hand",
     "ELEC292_Jumping_RightHand_Data_Christian.csv",  1),
    ("christian_walking_left_pocket",
     "ELEC292_Walking_LeftPocket_Data_Christian.csv", 0),
]


def load_csv(path: str) -> np.ndarray:
    """Load CSV and return a (N, 4) float32 array: [time, x, y, z]."""
    df = pd.read_csv(path)
    return df.iloc[:, :4].values.astype(np.float32)


def get_sample_rate(data: np.ndarray) -> float:
    """Estimate sampling frequency from the time column."""
    diffs = np.diff(data[:, 0])
    diffs = diffs[diffs > 0]
    return float(1.0 / np.median(diffs))


def resample_to_target(data: np.ndarray, target_sr: float = 100.0) -> np.ndarray:
    """
    Resample data to target_sr using linear interpolation.
    If the data is already at target_sr (within 1 Hz), it is returned unchanged.
    This is required so that all windows represent the same duration in seconds
    regardless of the phone's recording rate.
    """
    current_sr = get_sample_rate(data)
    if abs(current_sr - target_sr) < 1.0:
        return data                         # already at target rate

    t_orig = data[:, 0]
    n_new  = int(np.round((t_orig[-1] - t_orig[0]) * target_sr)) + 1
    t_new  = np.linspace(t_orig[0], t_orig[-1], n_new)

    result = np.zeros((len(t_new), 4), dtype=np.float32)
    result[:, 0] = t_new
    for col in range(1, 4):
        f = interp1d(t_orig, data[:, col], kind='linear', fill_value='extrapolate')
        result[:, col] = f(t_new).astype(np.float32)

    return result


def segment(data: np.ndarray, window_len: int) -> np.ndarray:
    """Cut data into non-overlapping windows of exactly window_len samples."""
    n_windows = len(data) // window_len
    trimmed   = data[: n_windows * window_len]
    return trimmed.reshape(n_windows, window_len, 4)


def main():
    all_windows = []
    all_labels  = []

    with h5py.File(OUTPUT_FILE, 'w') as f:

        raw_grp   = f.create_group('raw')
        prepr_grp = f.create_group('preprocessed')
        seg_grp   = f.create_group('segmented')

        for ds_name, csv_path, label in DATASETS:

            # --- Load raw data -----------------------------------------------
            data      = load_csv(csv_path)
            sr_orig   = get_sample_rate(data)

            # Store raw data at original sample rate
            raw_grp.create_dataset(ds_name, data=data,
                                   compression='gzip', compression_opts=4)
            raw_grp[ds_name].attrs['sample_rate_hz']    = sr_orig
            raw_grp[ds_name].attrs['label']             = label
            raw_grp[ds_name].attrs['label_name']        = 'jumping' if label else 'walking'
            raw_grp[ds_name].attrs['columns']           = ['time_s', 'acc_x', 'acc_y', 'acc_z']

            # --- Resample to common 100 Hz -----------------------------------
            data_rs = resample_to_target(data, TARGET_SR)
            print(f"{ds_name}: orig SR={sr_orig:.1f} Hz, "
                  f"rows {len(data)} -> {len(data_rs)} after resampling to {TARGET_SR:.0f} Hz")

            # Placeholder preprocessed dataset (filled in by preprocess.py)
            prepr_grp.create_dataset(ds_name, shape=data_rs.shape,
                                     dtype=np.float32, compression='gzip')
            prepr_grp[ds_name].attrs['note'] = (
                "Placeholder – fill with preprocessed data in Step 4"
            )

            # --- Segment into 5-second windows --------------------------------
            windows = segment(data_rs, WINDOW_LEN)
            all_windows.append(windows)
            all_labels.append(np.full(len(windows), label, dtype=np.int8))
            print(f"  -> {len(windows)} windows of {WINDOW_LEN} samples "
                  f"({WINDOW_SECONDS}s each)\n")

        # --- Combine, shuffle, and split ------------------------------------
        X = np.concatenate(all_windows, axis=0)
        Y = np.concatenate(all_labels,  axis=0)

        rng     = np.random.default_rng(RANDOM_SEED)
        indices = rng.permutation(len(X))
        X, Y    = X[indices], Y[indices]

        split   = int(TRAIN_RATIO * len(X))
        X_train, X_test = X[:split],  X[split:]
        Y_train, Y_test = Y[:split],  Y[split:]

        print(f"Total windows : {len(X)}  "
              f"(walking={int((Y==0).sum())}, jumping={int((Y==1).sum())})")
        print(f"Train : {len(X_train)}    Test : {len(X_test)}")

        # --- Store segmented data -------------------------------------------
        seg_grp.create_dataset("train",        data=X_train, compression='gzip')
        seg_grp.create_dataset("train_labels", data=Y_train, compression='gzip')
        seg_grp.create_dataset("test",         data=X_test,  compression='gzip')
        seg_grp.create_dataset("test_labels",  data=Y_test,  compression='gzip')

        seg_grp.attrs["window_seconds"]      = WINDOW_SECONDS
        seg_grp.attrs["window_len_samples"]  = WINDOW_LEN
        seg_grp.attrs["target_sample_rate"]  = TARGET_SR
        seg_grp.attrs["label_encoding"]      = "0 = walking,  1 = jumping"
        seg_grp.attrs["train_ratio"]         = TRAIN_RATIO
        seg_grp.attrs["random_seed"]         = RANDOM_SEED
        seg_grp["train"].attrs["shape_description"] = (
            "(n_windows, window_len, 4)  columns: time, acc_x, acc_y, acc_z"
        )

    print("\ndata_storage.h5 written successfully.")


if __name__ == "__main__":
    main()