import h5py
import numpy as np

# Name will be as follows: "type_" + dataset_name -> e.g. "train_jumping_ben", "test_jumping_ben", "raw_jumping_ben"
dataset_name = "jumping_christian"
data = np.loadtxt('ELEC292_Jumping_RightHand_Data_Christian.csv', delimiter=',', skiprows=1)

window_len = 500
num_windows = len(data) // window_len
truncated_data = data[:num_windows * window_len]
windows = truncated_data.reshape(-1, window_len, 5)

split = int(0.9 * num_windows)
train_set = windows[:split]
test_set = windows[split:]

with h5py.File('h5py_data.h5', 'w') as f:
    processed = f.create_group('processed')
    raw = f.create_group('raw')

    processed.create_dataset("train_" + dataset_name, data=train_set, compression='gzip')
    processed.create_dataset("test_" + dataset_name, data=test_set, compression='gzip')
    raw.create_dataset("raw_" + dataset_name, data=truncated_data, compression='gzip')