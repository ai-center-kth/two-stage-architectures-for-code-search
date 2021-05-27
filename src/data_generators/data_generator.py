
import tables
import tensorflow as tf
import numpy as np
import pandas as pd
import random

# It generates a tf.dataset with Deep Code Search dataset (string)
# Format: (desc, code, neg_code)
def get_dcs_dataset(desc_path, code_path, vocab_desc, vocab_code, max_len=-1):

    return tf.data.Dataset.from_generator(
        dcs_generator(desc_path, code_path, vocab_desc, vocab_code, max_len),
            output_signature=(
                    tf.TensorSpec(shape=(), dtype=tf.string, name="description"),
                    tf.TensorSpec(shape=(), dtype=tf.string, name="code"),
                    tf.TensorSpec(shape=(), dtype=tf.string, name="neg_code")))

# Total number elements this dataset
DSC_NUM_ELEMENTS = 18223872

# It generates a tf.dataset with Code-Docstring-corpus dataset (string)
# Format: (desc, code, neg_code)
def get_cdc_dataset(desc_path, code_path, max_len=-1):
    code_df = pd.read_csv(code_path, header=None, sep='\n', quotechar='"', error_bad_lines=False)
    desc_df = pd.read_csv(desc_path, header=None, sep='\n', quotechar='"', error_bad_lines=False)

    if not max_len == -1:
        code_df = code_df.head(max_len)
        desc_df = desc_df.head(max_len)

    # pandas shuffle
    neg_code_df = code_df.sample(frac=1).reset_index(drop=True)

    dataset = tf.data.Dataset.from_tensor_slices((desc_df.values, code_df.values, neg_code_df.values))

    return dataset

# It generates a tf.dataset with CodeSearchChallenge dataset (string)
# Format: (desc, code, neg_code)
def get_csc_dataset(desc_path, code_path, max_len=-1):

    code_df = pd.read_csv(desc_path, compression='gzip', header=None,sep='\n', quotechar='"', error_bad_lines=False)
    desc_df = pd.read_csv(code_path, header=None, sep='\n', quotechar='"', error_bad_lines=False)

    if not max_len == -1:
        code_df = code_df.head(max_len)
        desc_df = desc_df.head(max_len)

    # pandas shuffle
    neg_code_df = code_df.sample(frac=1).reset_index(drop=True)

    dataset = tf.data.Dataset.from_tensor_slices((desc_df.values, code_df.values, neg_code_df.values))

    return dataset

class dcs_generator():
    def __init__(self, desc_path, code_path, vocab_desc, vocab_code, max_len=-1):
        # Load code
        self.vocab_code = vocab_code
        self.vocab_desc = vocab_desc

        code_table = tables.open_file(code_path)
        self.code_data = code_table.get_node('/phrases')[:].astype(np.int)
        self.code_index = code_table.get_node('/indices')[:]

        desc_table = tables.open_file(desc_path)
        self.desc_data = desc_table.get_node('/phrases')[:].astype(np.int)
        self.desc_index = desc_table.get_node('/indices')[:]
        self.total_len = self.code_index.shape[0]
        self.data_len = max_len

        if max_len == -1 or max_len > self.total_len:
            self.data_len = self.total_len

        code_table.close()

    def __call__(self):

        for offset in range(0, self.data_len):
            code_len, code_pos = self.code_index[offset]['length'], self.code_index[offset]['pos']
            desc_len, desc_pos = self.desc_index[offset]['length'], self.desc_index[offset]['pos']

            random_index = random.randint(0, self.total_len - 1)
            neg_len, neg_pos = self.code_index[random_index]['length'], self.code_index[random_index]['pos']

            extracted_code = self.code_data[code_pos:code_pos + code_len]
            extracted_desc = self.desc_data[desc_pos:desc_pos + desc_len]
            extracted_neg_code = self.code_data[neg_pos:neg_pos + neg_len]

            desc = (" ".join([self.vocab_desc[x] for x in extracted_desc]))
            code = (" ".join([self.vocab_code[x] for x in extracted_code]))
            neg_code = (" ".join([self.vocab_code[x] for x in extracted_neg_code]))

            yield desc, code, neg_code

# This class will allow to use any dataset in a Sentece-Bert model
class sentece_bert_tokenizer_map():
    # Tokenize is a function that tokenizes (desc,code) that must return id, att_mask, seg_type
    def __init__(self, tokenize, max_len):
        self.tokenize = tokenize
        self.max_len = max_len

    def __call__(self, desc, code, neg):
        # Tokenize the description (desc is a tensor here)
        desc_ = tf.reshape(desc, (1,))

        desc_tokenized = tf.py_function(lambda x: tf.constant(self.tokenize(x[0].numpy().decode('utf-8'))), [desc_],tf.int32)

        # desc_tokenized is a tensor with shape [3, max_len]
        # Here split the first dimension in id, attention and seg_type and remove this empty dimension
        desc_ids = tf.squeeze(tf.slice(desc_tokenized, [0, 0], [1, self.max_len]), [0])
        desc_attention = tf.squeeze(tf.slice(desc_tokenized, [1, 0], [1, self.max_len]), [0])
        desc_type = tf.squeeze(tf.slice(desc_tokenized, [2, 0], [1, self.max_len]), [0])

        code_ = tf.reshape(code, (1,))
        code_tokenized = tf.py_function(lambda x: tf.constant(self.tokenize(x[0].numpy().decode('utf-8'))), [code_],
                                        tf.int32)

        code_ids = tf.squeeze(tf.slice(code_tokenized, [0, 0], [1, self.max_len]), [0])
        code_attention = tf.squeeze(tf.slice(code_tokenized, [1, 0], [1, self.max_len]), [0])
        code_type = tf.squeeze(tf.slice(code_tokenized, [2, 0], [1, self.max_len]), [0])

        neg_ = tf.reshape(neg, (1,))
        neg_tokenized = tf.py_function(lambda x: tf.constant(self.tokenize(x[0].numpy().decode('utf-8'))), [neg_],
                                       tf.int32)

        neg_ids = tf.squeeze(tf.slice(neg_tokenized, [0, 0], [1, self.max_len]), [0])
        neg_attention = tf.squeeze(tf.slice(neg_tokenized, [1, 0], [1, self.max_len]), [0])
        neg_type = tf.squeeze(tf.slice(neg_tokenized, [2, 0], [1, self.max_len]), [0])

        return (desc_ids, desc_attention, desc_type,
                code_ids, code_attention, code_type,
                neg_ids, neg_attention, neg_type
                ), tf.constant(0.)

# mono_bert_tokenizer_map generates 1 positive sample and 1 negative sample per call
# this method flats mono_bert_tokenizer_map dataset
def flat_mono_bert_map(ds):
    return ds.flat_map(lambda first, second:
                        tf.data.Dataset.zip(
                            (
                                tf.data.Dataset.zip((
                                    tf.data.Dataset.from_tensor_slices([first[0][0], second[0][0]]),
                                    tf.data.Dataset.from_tensor_slices([first[0][1], second[0][1]]),
                                    tf.data.Dataset.from_tensor_slices([first[0][2], second[0][2]])
                                )),
                                tf.data.Dataset.from_tensor_slices([first[1], second[1]])
                            )))

class mono_bert_tokenizer_map():
    # Tokenize is a function that must return id, att_mask, seg_type
    def __init__(self, tokenize, max_len):
        self.tokenize = tokenize
        self.max_len = max_len

    def __call__(self, desc, code, neg):
        desc_ = tf.reshape(desc, (1,))
        code_ = tf.reshape(code, (1,))
        neg_ = tf.reshape(neg, (1,))

        pos_tokenized = tf.py_function(lambda x, y: tf.constant(self.tokenize(x[0].numpy().decode('utf-8'),
                                                                         y[0].numpy().decode('utf-8')
                                                                         )), [desc_, code_],
                                       tf.int32)

        pos_ids = tf.squeeze(tf.slice(pos_tokenized, [0, 0], [1, self.max_len]), [0])
        pos_attention = tf.squeeze(tf.slice(pos_tokenized, [1, 0], [1, self.max_len]), [0])
        pos_type = tf.squeeze(tf.slice(pos_tokenized, [2, 0], [1, self.max_len]), [0])
        post_label = tf.reshape(tf.constant(1.), (1,))
        neg_tokenized = tf.py_function(lambda x, y: tf.constant(self.tokenize(x[0].numpy().decode('utf-8'),
                                                                         y[0].numpy().decode('utf-8')
                                                                         )), [desc_, neg_],
                                       tf.int32)

        neg_ids = tf.squeeze(tf.slice(neg_tokenized, [0, 0], [1, self.max_len]), [0])
        neg_attention = tf.squeeze(tf.slice(neg_tokenized, [1, 0], [1, self.max_len]), [0])
        neg_type = tf.squeeze(tf.slice(neg_tokenized, [2, 0], [1, self.max_len]), [0])
        neg_label = tf.reshape(tf.constant(0.), (1,))

        return (
            ((pos_ids, pos_attention, pos_type), post_label),
            ((neg_ids, neg_attention, neg_type), neg_label)
        )