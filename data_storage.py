import numpy as np
import pandas as pd
import h5py
from scipy.interpolate import interp1d

OUTPUT_FILE = 'data_storage.h5'
WINDOW_SECONDS = 5
TARGET_SR = 100.0 # resamples all datasets to a 100 Hz
WINDOW_LEN = int(TARGET_SR * WINDOW_SECONDS) # 500 samples/window
TRAIN_RATIO = 0.9
RANDOM_SEED = 292 # preset random seed for reproducibility

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

# returns a N x 4 array with time, x, y and z cols
def load_csv(path: str) -> np.ndarray:
    df = pd.read_csv(path)
    return df.iloc[:, :4].values.astype(np.float32)

# estimate sampling freq of datasets based on time
def get_sample_rate(data: np.ndarray) -> float:
    diffs = np.diff(data[:, 0])
    diffs = diffs[diffs > 0]
    return float(1.0 / np.median(diffs))

# resample data to target_sr using linear interpolation, only applies if not already within 1 Hz of target_sr
def resample_to_target(data: np.ndarray, target_sr: float = 100.0) -> np.ndarray:
    current_sr = get_sample_rate(data)
    
    # if already close to target_sr, return original
    if abs(current_sr - target_sr) < 1.0:
        return data

    t_orig = data[:, 0]
    n_new  = int(np.round((t_orig[-1] - t_orig[0]) * target_sr)) + 1
    t_new  = np.linspace(t_orig[0], t_orig[-1], n_new)

    result = np.zeros((len(t_new), 4), dtype = np.float32)
    result[:, 0] = t_new

    # linear interpolation for x, y, z accelerations
    for col in range(1, 4):
        f = interp1d(t_orig, data[:, col], kind = 'linear', fill_value = 'extrapolate')
        result[:, col] = f(t_new).astype(np.float32)

    return result

# cut data into windows of window_len samples, get rid of leftover samples if they dont fit
def segment(data: np.ndarray, window_len: int) -> np.ndarray:
    n_windows = len(data) // window_len
    trimmed   = data[: n_windows * window_len]
    return trimmed.reshape(n_windows, window_len, 4)

# main function, runs everything
def main():
    all_windows = []
    all_labels  = []

    # opens HDF5 file, creates groups and stores all data from CSVs
    with h5py.File(OUTPUT_FILE, 'w') as f:
        raw_grp   = f.create_group('raw')
        prepr_grp = f.create_group('preprocessed')
        seg_grp   = f.create_group('segmented')

        for ds_name, csv_path, label in DATASETS:
            # load data
            data      = load_csv(csv_path)
            sr_orig   = get_sample_rate(data)

            # store and compress data, compressed with gzip level 4 to balance size and speed
            raw_grp.create_dataset(ds_name, data = data, compression = 'gzip', compression_opts = 4)
            raw_grp[ds_name].attrs['sample_rate_hz'] = sr_orig
            raw_grp[ds_name].attrs['label'] = label
            raw_grp[ds_name].attrs['label_name'] = 'jumping' if label else 'walking' # 0 = walking, 1 = jumping as written above
            raw_grp[ds_name].attrs['columns'] = ['time_s', 'acc_x', 'acc_y', 'acc_z']

            # resample data to 100 Hz if needed
            data_rs = resample_to_target(data, TARGET_SR)

            # create spaces for preprocessed data, will fill in preprocess.py
            prepr_grp.create_dataset(ds_name, shape = data_rs.shape, dtype = np.float32, compression = 'gzip')
            prepr_grp[ds_name].attrs['note'] = ("placeholder for preprocessed data, filled in later") # adds an atribiute to HDF5 file so team still knows that preprocessing needs to be done

            # split data into 5s windows
            windows = segment(data_rs, WINDOW_LEN)
            all_windows.append(windows)
            all_labels.append(np.full(len(windows), label, dtype = np.int8))

        # combine, shuffle, and split data 9 : 1
        X = np.concatenate(all_windows, axis = 0)
        Y = np.concatenate(all_labels, axis = 0)

        rng = np.random.default_rng(RANDOM_SEED)
        indices = rng.permutation(len(X))
        X, Y = X[indices], Y[indices]

        split = int(TRAIN_RATIO * len(X))
        X_train, X_test = X[:split], X[split:]
        Y_train, Y_test = Y[:split], Y[split:]

        # store segmented data in HDF5 file
        seg_grp.create_dataset("train", data=X_train, compression='gzip')
        seg_grp.create_dataset("train_labels", data=Y_train, compression='gzip')
        seg_grp.create_dataset("test", data=X_test,  compression='gzip')
        seg_grp.create_dataset("test_labels", data=Y_test,  compression='gzip')

        # set attributes for future reference
        seg_grp.attrs["window_seconds"] = WINDOW_SECONDS
        seg_grp.attrs["window_len_samples"] = WINDOW_LEN
        seg_grp.attrs["target_sample_rate"] = TARGET_SR
        seg_grp.attrs["label_encoding"]= "0 = walking, 1 = jumping"
        seg_grp.attrs["train_ratio"] = TRAIN_RATIO
        seg_grp.attrs["random_seed"] = RANDOM_SEED
        seg_grp["train"].attrs["shape_description"] = ("n_windows, window_len, 4, cols: time, acc_x, acc_y, acc_z")

    print("\ndata_storage.h5 done.")

# allows this file to be run on its own, but also allows functions to be imported into preprocess.py without running main()
if __name__ == "__main__":
    main()