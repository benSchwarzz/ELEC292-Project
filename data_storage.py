import numpy as np
import pandas as pd
import h5py

OUTPUT_FILE = 'data_storage.h5'
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

# load data from CSV, drop absolute acc, convert to numpy array
def load_csv(path: str) -> np.ndarray:
    df = pd.read_csv(path)
    data = df.iloc[:, :4].values.astype(np.float32)
    return data

# calculates sampling frequency, required bc not all datasets have the same collection frequency, have to normalize data so that all windows are the same legnth of time
def sample_rate(data: np.ndarray) -> float:
    diffs = np.diff(data[:, 0])
    diffs = diffs[diffs > 0] # safety check to remove -ve diffs
    return float(1.0 / np.median(diffs)) # return frequncy 

# cut data into nonoverlapping windows
def segment(data: np.ndarray, window_len: int) -> np.ndarray:
    n_windows = len(data) // window_len # round down so only int number of windows
    trimmed = data[: n_windows * window_len] # trim off extra samples
    return trimmed.reshape(n_windows, window_len, 4) # reshape into 3D array

def main():
    all_windows = []
    all_labels = []

    with h5py.File(OUTPUT_FILE, 'w') as f:

        #create groups
        raw_grp = f.create_group('raw')
        prepr_grp = f.create_group('preprocessed')
        seg_grp = f.create_group('segmented')

        for ds_name, cv_path, label in DATASETS:

            # load raw data
            data = load_csv(cv_path)
            sr = sample_rate(data)
            window_len = int(sr * WINDOW_SECONDS)

            # store raw data
            raw_grp.create_dataset(ds_name, data = data, compression  = 'gzip', compression_opts = 4)
            raw_grp[ds_name].attrs['sample_rate_hz'] = sr
            raw_grp[ds_name].attrs['label'] = label
            raw_grp[ds_name].attrs['label_name'] = 'jumping' if label else 'walking'
            raw_grp[ds_name].attrs['columns'] = ['time_s', 'acc_x', 'acc_y', 'acc_z']
            raw_grp[ds_name].attrs["window_len_used"] = window_len  

            prepr_grp.create_dataset(
                ds_name,
                shape=data.shape,
                dtype=np.float32,
                compression="gzip",
            )
            prepr_grp[ds_name].attrs["note"] = (
                "Placeholder – fill with preprocessed data in Step 4"
            )   

            # segment data into windows
            windows = segment(data, window_len)
            all_windows.append(windows)
            all_labels.append(np.full(len(windows), label, dtype=np.int8))


        # fin min window len and trim all windows to taht size
        min_wl = min(w.shape[1] for w in all_windows)
        trimmed_windows = [w[:, :min_wl, :] for w in all_windows]
        X = np.concatenate(trimmed_windows, axis = 0)
        Y = np.concatenate(all_labels, axis = 0)

        # shuffle data
        rng = np.random.default_rng(RANDOM_SEED)
        indicies = rng.permutation(len(X))
        X, Y = X[indicies], Y[indicies]

        # split data
        split = int(TRAIN_RATIO * len(X))
        X_train, X_test = X[ : split], X[split : ]
        Y_train, Y_test = Y[ : split], Y[split : ]

        # store segmented data
        seg_grp.create_dataset("train", data = X_train, compression = "gzip")
        seg_grp.create_dataset("train_labels", data = Y_train, compression = "gzip")
        seg_grp.create_dataset("test", data = X_test,  compression = "gzip")
        seg_grp.create_dataset("test_labels", data = Y_test,  compression = "gzip")
         
        seg_grp.attrs["window_seconds"] = WINDOW_SECONDS
        seg_grp.attrs["window_len_samples"] = min_wl
        seg_grp.attrs["label_encoding"] = "0 = walking,  1 = jumping"
        seg_grp.attrs["train_ratio"] = TRAIN_RATIO
        seg_grp.attrs["random_seed"] = RANDOM_SEED
        seg_grp["train"].attrs["shape_description"] = "(n_windows, window_len, 4)  columns: time, acc_x, acc_y, acc_z"




if __name__ == "__main__":
    main()