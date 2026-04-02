import os
import numpy as np
import pandas as pd
import h5py

HDF5_FILE = 'data_storage.h5'

# (hdf5_dataset_name, csv_path, moving_avg_window)
# Window chosen to cover ~0.15s of signal for each sample rate:
#   Sachin  ~10  Hz -> window of  2 samples
#   Ben     ~100 Hz -> window of 15 samples
#   Christian ~100 Hz -> window of 15 samples
DATASETS = [
    # Sachin (~10 Hz)
    ("sachin_jumping_sweater_pocket",
     "Data\\CSV\\Data_Jumping_SweaterPocket_Sachin.csv",    2),
    ("sachin_walking_sweater_pocket",
     "Data\\CSV\\Data_Walking_SweaterPocket_Sachin.csv",    2),

    # Ben (~100 Hz)
    ("ben_jumping",
     "Data\\CSV\\Data_Jumping_Ben.csv",                     15),
    ("ben_walking_outside",
     "Data\\CSV\\Data_WalkingOutside_Ben.csv",              15),

    # Christian (~100 Hz)
    ("christian_jumping_right_hand",
     "Data\\CSV\\ELEC292_Jumping_RightHand_Data_Christian.csv",  15),
    ("christian_walking_left_pocket",
     "Data\\CSV\\ELEC292_Walking_LeftPocket_Data_Christian.csv", 15),
]


def load_csv(path: str) -> pd.DataFrame:
    """Load CSV and return a DataFrame with columns [time, x, y, z]."""
    df = pd.read_csv(path)
    df = df.iloc[:, :4]
    df.columns = ['time', 'x', 'y', 'z']
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Drop fully duplicated rows and reset the index."""
    n_before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    n_removed = n_before - len(df)
    if n_removed > 0:
        print(f"    Duplicates removed : {n_removed}")
    return df


def fill_time_gaps(df: pd.DataFrame, gap_threshold: float = 3.0) -> pd.DataFrame:
    """
    Detect gaps in the time column larger than gap_threshold * median_dt
    and fill them by linear interpolation.

    gap_threshold: a gap must be this many times the median timestep
                   to be considered a real gap worth filling.
    """
    diffs     = np.diff(df['time'].values)
    diffs_pos = diffs[diffs > 0]
    median_dt = float(np.median(diffs_pos))
    gap_mask  = diffs > gap_threshold * median_dt
    n_gaps    = gap_mask.sum()

    if n_gaps == 0:
        return df

    print(f"    Time gaps found    : {n_gaps}  (filling by interpolation)")

    new_rows = []
    for idx in np.where(gap_mask)[0]:
        t0     = df['time'].iloc[idx]
        t1     = df['time'].iloc[idx + 1]
        n_fill = max(1, round((t1 - t0) / median_dt) - 1)
        t_fill = np.linspace(t0, t1, n_fill + 2)[1:-1]  # exclude endpoints

        for t in t_fill:
            alpha = (t - t0) / (t1 - t0)   # interpolation weight 0 -> 1
            new_rows.append({
                'time': t,
                'x': df['x'].iloc[idx] + alpha * (df['x'].iloc[idx+1] - df['x'].iloc[idx]),
                'y': df['y'].iloc[idx] + alpha * (df['y'].iloc[idx+1] - df['y'].iloc[idx]),
                'z': df['z'].iloc[idx] + alpha * (df['z'].iloc[idx+1] - df['z'].iloc[idx]),
            })

    inserts = pd.DataFrame(new_rows)
    df      = pd.concat([df, inserts], ignore_index=True)
    df      = df.sort_values('time').reset_index(drop=True)

    print(f"    Rows after fill    : {len(df)}")
    return df


def moving_average(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """
    Apply a uniform moving average filter to x, y, z columns independently.
    min_periods=1 ensures the edges of the signal are smoothed rather than
    producing NaN values.
    """
    df = df.copy()
    for col in ['x', 'y', 'z']:
        df[col] = df[col].rolling(window=window, center=True, min_periods=1).mean()
    return df


def main():
    if not os.path.exists(HDF5_FILE):
        print(f"ERROR: {HDF5_FILE} not found. Run build_dataset.py first.")
        return

    print(f"Opening {HDF5_FILE}\n")

    with h5py.File(HDF5_FILE, 'a') as f:

        # Create the preprocessed group if it doesn't exist yet
        if 'preprocessed' not in f:
            f.create_group('preprocessed')

        for ds_name, csv_path, window in DATASETS:
            print(f"── {ds_name}")

            # Load
            df = load_csv(csv_path)
            print(f"    Rows loaded        : {len(df)}")

            # Step 1: remove duplicates
            df = remove_duplicates(df)

            # Step 2: fill time gaps
            df = fill_time_gaps(df)

            # Step 3: moving average filter
            df = moving_average(df, window)
            print(f"    Moving avg window  : {window} samples")

            # Write to HDF5
            data = df.values.astype(np.float32)
            key  = f'preprocessed/{ds_name}'

            # If dataset already exists (e.g. rerunning the script), delete it first
            if key in f:
                del f[key]

            f.create_dataset(key, data=data, compression='gzip', compression_opts=4)
            f[key].attrs['columns']           = ['time_s', 'acc_x', 'acc_y', 'acc_z']
            f[key].attrs['moving_avg_window'] = window
            f[key].attrs['label']             = f['raw'][ds_name].attrs['label']
            f[key].attrs['label_name']        = f['raw'][ds_name].attrs['label_name']

            print(f"    Written            : {data.shape}\n")

    print("Preprocessing complete.")


if __name__ == "__main__":
    main()