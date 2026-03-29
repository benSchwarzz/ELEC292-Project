import h5py
import numpy as np

with h5py.File('h5py_data.h5', 'a') as f:
    del f['segmented']

    f.create_group('segmented')
    f.copy('test', 'segmented/test')
    del f['test']
    f.copy('train', 'segmented/train')
    del f['train']

    """f.copy('processed', 'segmented')
    del f['processed']

    f.create_group('test')
    f.create_group('train')

    for dataset in f['segmented']:
        if dataset.startswith('train'):
            f.copy('segmented/' + dataset, 'train/' + dataset)
        elif dataset.startswith('test'):
            f.copy('segmented/' + dataset, 'test/' + dataset)"""
