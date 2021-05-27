
import subprocess
import sys

try:
    import pickle5 as pickle
except:
    import pickle

import tables
from tqdm import tqdm
import numpy as np


def load_hdf5(vecfile, start_offset, chunk_size):
    """reads training sentences(list of int array) from a hdf5 file"""
    table = tables.open_file(vecfile)
    data = table.get_node('/phrases')[:].astype(np.int)
    index = table.get_node('/indices')[:]
    data_len = index.shape[0]
    if chunk_size == -1:  # if chunk_size is set to -1, then, load all data
        chunk_size = data_len
    start_offset = start_offset % data_len
    sents = []
    for offset in tqdm(range(start_offset, start_offset + chunk_size)):
        offset = offset % data_len
        len, pos = index[offset]['length'], index[offset]['pos']
        sents.append(data[pos:pos + len])
    table.close()
    return sents


def pad(data, len=None):
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    return pad_sequences(data, maxlen=len, padding='post', truncating='post', value=0)


def load_pickle(filename):
    file = open(filename, 'rb')
    element = pickle.load(file)
    file.close()
    return element


def save_pickle(filename, element):
    a_file = open(filename, "wb")
    pickle.dump(element, a_file)
    a_file.close()