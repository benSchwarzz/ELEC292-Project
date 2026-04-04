import os
import numpy as np
import pandas as pd
import h5py

HDF5_FILE = 'data_storage.h5'

# 15 was the chosen window sizes, not set to a constant because the team was testing differnet windows per dataset
DATASETS = [
    # Sachin
    ("sachin_jumping_sweater_pocket", "Data_Jumping_SweaterPocket_Sachin.csv", 15),
    ("sachin_walking_sweater_pocket", "Data_Walking_SweaterPocket_Sachin.csv", 15),

    # Ben
    ("ben_jumping", "Data_Jumping_Ben.csv", 15),
    ("ben_walking_outside","Data_WalkingOutside_Ben.csv", 15), 

    # Christian
    ("christian_jumping_right_hand", "ELEC292_Jumping_RightHand_Data_Christian.csv", 15),
    ("christian_walking_left_pocket", "ELEC292_Walking_LeftPocket_Data_Christian.csv", 15)]

# returns a N x 4 array with time, x, y and z cols
def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.iloc[:, :4]
    df.columns = ['time', 'x', 'y', 'z']
    return df

# remove duplicates and reset indexes
def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    len_before  = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    n_removed = len_before - len(df)
    if n_removed > 0:
        print(f"Duplicates removed: {n_removed}")
    return df

# fill data/time gaps by linear interpolation, only applies if gaps > 3x median gap
def fill_time_gaps(df: pd.DataFrame, gap_threshold: float = 3.0) -> pd.DataFrame:
    diffs = np.diff(df['time'].values)
    diffs_pos = diffs[diffs > 0]
    median_dt = float(np.median(diffs_pos))
    gap_mask = diffs > gap_threshold * median_dt
    n_gaps = gap_mask.sum()

    if n_gaps == 0:
        return df

    print(f"gaps found: {n_gaps}  (filling by interpolation)")

    # gap filling
    new_rows = []
    for idx in np.where(gap_mask)[0]:
        t0 = df['time'].iloc[idx]
        t1 = df['time'].iloc[idx + 1]
        n_fill = max(1, round((t1 - t0) / median_dt) - 1)
        t_fill = np.linspace(t0, t1, n_fill + 2)[1:-1]

        for t in t_fill:
            alpha = (t - t0) / (t1 - t0)
            new_rows.append({
                'time': t,
                'x': df['x'].iloc[idx] + alpha * (df['x'].iloc[idx + 1] - df['x'].iloc[idx]),
                'y': df['y'].iloc[idx] + alpha * (df['y'].iloc[idx + 1] - df['y'].iloc[idx]),
                'z': df['z'].iloc[idx] + alpha * (df['z'].iloc[idx + 1] - df['z'].iloc[idx])})

    inserts = pd.DataFrame(new_rows)
    df = pd.concat([df, inserts], ignore_index=True)
    df = df.sort_values('time').reset_index(drop=True)
    print(f"Rows after fill: {len(df)}")
    return df

# apply moving average filter of window size, center = True to reduce phase shift, min_periods = 1 to avoid NaNs at edges
def moving_average(df: pd.DataFrame, window: int) -> pd.DataFrame:
    df = df.copy() # edit a copy just to be safe
    for col in ['x', 'y', 'z']:
        df[col] = df[col].rolling(window = window, center = True, min_periods = 1).mean()
    return df

# runs everything
def main():
    # opens HDF5 file, creates groups and stores all data from CSVs
    with h5py.File(HDF5_FILE, 'a') as f:

        # creates preprocessed group incase it was done wrong in datasotrage 
        if 'preprocessed' not in f:
            f.create_group('preprocessed')

        # preprocesses each dataset and stores in HDF5
        for ds_name, csv_path, window in DATASETS:
            df = load_csv(csv_path)

            df = remove_duplicates(df)
            df = fill_time_gaps(df)
            df = moving_average(df, window)

            data = df.values.astype(np.float32)
            key  = f'preprocessed/{ds_name}'

            if key in f:
                del f[key]

            f.create_dataset(key, data = data, compression = 'gzip', compression_opts = 4)
            f[key].attrs['columns'] = ['time_s', 'acc_x', 'acc_y', 'acc_z']
            f[key].attrs['moving_avg_window'] = window
            f[key].attrs['label'] = f['raw'][ds_name].attrs['label']
            f[key].attrs['label_name'] = f['raw'][ds_name].attrs['label_name']

    print("preprocessing done")

# allows file to run on its own, but also allows functions to be imported into other files without running main
if __name__ == "__main__":
    main()