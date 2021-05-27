import tensorflow.keras as keras
import tables
import numpy as np
import random

class DataGeneratorDCSBERT(keras.utils.Sequence):
    def __init__(self, tokens_path, desc_path, batch_size, init_pos, last_pos, max_length, tokenizer, vocab_code, vocab_desc):
        self.tokens_path = tokens_path
        self.desc_path = desc_path
        self.batch_size = batch_size
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.vocab_code = vocab_code
        self.vocab_desc = vocab_desc

        # code
        code_table = tables.open_file(tokens_path)
        self.code_data = code_table.get_node('/phrases')[:].astype(np.int)
        self.code_index = code_table.get_node('/indices')[:]
        self.full_data_len = self.code_index.shape[0]

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

        retokenized_desc = []
        retokenized_mask_desc = []
        retokenized_type_desc = []

        retokenized_code = []
        retokenized_mask_code = []
        retokenized_type_code = []

        bad_retokenized_code = []
        bad_retokenized_mask_code = []
        bad_retokenized_type_code = []

        labels = []

        for offset in range(self.init_pos + start_offset, self.init_pos + start_offset + chunk_size):
            offset = offset % self.full_data_len

            # CODE
            len_, pos = self.code_index[offset]['length'], self.code_index[offset]['pos']
            extracted_code = self.code_data[pos:pos + len_].copy()
            code =  (" ".join([self.vocab_code[x] for x in extracted_code]))


            len_, pos = self.desc_index[offset]['length'], self.desc_index[offset]['pos']
            extracted_desc = self.desc_data[pos:pos + len_].copy()
            desc = ((" ".join([self.vocab_desc[x] for x in extracted_desc])))

            random_index = random.randint(0, self.full_data_len - 1)
            len_, pos = self.code_index[random_index]['length'], self.code_index[random_index]['pos']
            extracted_neg_code = self.code_data[pos:pos + len_].copy()
            neg_code = ((" ".join([self.vocab_code[x] for x in extracted_neg_code])))

            desc_ = self.__tokenize(desc)
            code_ = self.__tokenize(code)
            neg_ = self.__tokenize(neg_code)

            retokenized_desc.append(desc_[0])
            retokenized_mask_desc.append(desc_[1])
            retokenized_type_desc.append(desc_[2])

            retokenized_code.append(code_[0])
            retokenized_mask_code.append(code_[1])
            retokenized_type_code.append(code_[2])

            bad_retokenized_code.append(neg_[0])
            bad_retokenized_mask_code.append(neg_[1])
            bad_retokenized_type_code.append(neg_[2])


        labels = np.zeros((len(bad_retokenized_code), 1))

        return [np.array(retokenized_desc), np.array(retokenized_mask_desc), np.array(retokenized_type_desc),
                np.array(retokenized_code), np.array(retokenized_mask_code), np.array(retokenized_type_code),
                np.array(bad_retokenized_code), np.array(bad_retokenized_mask_code), np.array(bad_retokenized_type_code),
                ], np.array(labels)

    def len(self):
        return self.__len__()

    def __tokenize(self, input_str):
        return DataGeneratorDCSBERT.tokenize_sentences(self.tokenizer, self.max_length, input_str)

    @staticmethod
    def tokenize_sentences(tokenizer, max_length, input_str):
        _input_str1 = input_str
        if isinstance(input_str, str):
            _input_str1 = [input_str]

        tokenized = tokenizer.batch_encode_plus(
            _input_str1,
            add_special_tokens=True,
            max_length=max_length,
            return_attention_mask=True,
            return_token_type_ids=True,
            return_tensors="np",
            padding='max_length'
        )
        if isinstance(input_str, str):
            return tokenized["input_ids"][0], tokenized["attention_mask"][0], tokenized["token_type_ids"][0]
        return tokenized["input_ids"], tokenized["attention_mask"], tokenized["token_type_ids"]
