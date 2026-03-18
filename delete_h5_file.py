import h5py
import numpy as np

# Name will be as follows: "type_" + dataset_name -> e.g. "train_jumping_sachin", "test_jumping_sachin", "raw_jumping_sachin"
dataset_name = "jumping_sachin"

with h5py.File('h5py_data.h5', 'a') as f:
    if '/raw/raw_' + dataset_name in f:
        del f['/raw/raw_' + dataset_name]
    if '/processed/train_' + dataset_name in f:
        del f['/processed/train_' + dataset_name]
    if '/processed/test_' + dataset_name in f:
        del f['/processed/test_' + dataset_name]