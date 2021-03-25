import tensorflow.keras as keras
import tables
import numpy as np
import random

class DataGeneratorDCS(keras.utils.Sequence):
    def __init__(self, tokens_path, desc_path, batch_size, init_pos, last_pos, code_length, desc_length):
        self.tokens_path = tokens_path
        self.desc_path = desc_path
        self.batch_size = batch_size
        self.code_length = code_length
        self.desc_length = desc_length

        # code
        code_table = tables.open_file(tokens_path)
        self.code_data = code_table.get_node('/phrases')[:].astype(np.int)
        self.code_index = code_table.get_node('/indices')[:]
        self.full_data_len = self.code_index.shape[0]

        #self.full_data_len = 100 #100000

        # desc
        desc_table = tables.open_file(desc_path)
        self.desc_data = desc_table.get_node('/phrases')[:].astype(np.int)
        self.desc_index = desc_table.get_node('/indices')[:]

        self.init_pos = init_pos
        self.last_pos = min(last_pos, self.full_data_len)

        self.data_len = self.last_pos - self.init_pos
        print("First row", self.init_pos, "last row", self.last_pos, "len", self.__len__())

    def __len__(self):
        return (np.ceil((self.last_pos - self.init_pos) / float(self.batch_size))).astype(np.int)

    def __getitem__(self, idx):

        start_offset = idx * self.batch_size
        start_offset = start_offset % self.data_len
        chunk_size = self.batch_size

        code = []
        desc = []

        for offset in range(self.init_pos + start_offset, self.init_pos + start_offset + chunk_size):
            offset = offset % self.full_data_len

            # CODE
            len, pos = self.code_index[offset]['length'], self.code_index[offset]['pos']
            code.append(self.code_data[pos:pos + len].copy())

            # Desc
            len, pos = self.desc_index[offset]['length'], self.desc_index[offset]['pos']
            desc.append(self.desc_data[pos:pos + len].copy())

        code = self.pad(code, self.code_length)
        desc = self.pad(desc, self.desc_length)

        negative_description_vector = desc.copy()
        random.shuffle(negative_description_vector)

        results = np.zeros((self.batch_size, 1))

        return [np.array(code), np.array(desc), np.array(negative_description_vector)], results

    def test(self, idx):
        start_offset = idx * self.batch_size
        start_offset = start_offset % self.data_len
        chunk_size = self.batch_size

        code = []
        desc = []

        print(self.init_pos + start_offset, self.init_pos + start_offset + chunk_size)

        # return self.__getitem__(idx)

    def len(self):
        return self.__len__()

    def pad(self, data, len=None):
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        return pad_sequences(data, maxlen=len, padding='post', truncating='post', value=0)
