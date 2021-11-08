
import sys
import pathlib

import transformers
import tensorflow as tf

from .search_models.mono_bert_model import MonoBERT_SearchModel
from .search_models import models
from . import data_generator
from . import helper

class MONOBERT_DCS(MonoBERT_SearchModel):

    def __init__(self, data_path):
        self.data_path = data_path
        self.max_len = 90

    def get_vocabularies(self):
        self.inverse_vocab_tokens = helper.load_pickle(self.data_path + "vocab.tokens.pkl")
        self.vocab_tokens = {y: x for x, y in self.inverse_vocab_tokens.items()}

        self.inverse_vocab_desc = helper.load_pickle(self.data_path + "vocab.desc.pkl")
        self.vocab_desc = {y: x for x, y in self.inverse_vocab_desc.items()}

        return self.vocab_tokens, self.vocab_desc

    def generate_tokenizer(self):
        self.tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        return self.tokenizer

    def generate_bert_layer(self):
        self.bert_layer = transformers.TFBertModel.from_pretrained('bert-base-uncased')
        return self.bert_layer

    def get_model(self):
        self.training_model = models.mono_bert_model(self.bert_layer, self.max_len)

    def load_dataset(self, batch_size=32):
        # ds output is (desc, code, neg_code) strings
        ds = data_generator.get_dcs_dataset(self.data_path + "train.desc.h5", self.data_path + "train.tokens.h5",
                                        self.vocab_desc, self.vocab_tokens, max_len=-1)

        # Tokenize the dataset
        ds = ds.map(data_generator.mono_bert_tokenizer_map(self.tokenize_sentences, self.max_len))

        ds = data_generator.flat_mono_bert_map(ds)

        ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        ds = ds.batch(batch_size, drop_remainder=True)

        return ds

    def tokenize_sentences(self, input_str1, input_str2):

        if isinstance(input_str1, str):
            input_str1 = [input_str1]
            input_str2 = [input_str2]

        # Tf outputs strings as bytes
        if type(input_str1[0]) is bytes:
            input_str1 = [x.decode('utf-8') for x in input_str1]
            input_str2 = [x.decode('utf-8') for x in input_str2]

        tokenizer_input = list(zip(input_str1, input_str2))

        tokenized = self.tokenizer.batch_encode_plus(
            tokenizer_input,
            add_special_tokens=True,
            max_length=self.max_len,
            return_attention_mask=True,
            return_token_type_ids=True,
            return_tensors="np",
            padding='max_length',
            truncation=True
        )

        return tokenized["input_ids"], tokenized["attention_mask"], tokenized["token_type_ids"]

if __name__ == "__main__":

    args = sys.argv
    data_chunk_id = 0
    if len(args) > 1:
        data_chunk_id = int(args[1])

    script_path = str(pathlib.Path(__file__).parent)

    data_path = script_path + "/../data/deep-code-search/drive/"

    BATCH_SIZE = 16

    monobert = MONOBERT_DCS(data_path)

    vocab_code, vocab_desc = monobert.get_vocabularies()

    monobert.generate_bert_layer()
    monobert.generate_tokenizer()
    monobert.get_model()

    #steps_per_epoch = 2 * monobert.chunk_size // BATCH_SIZE

    monobert.load_dataset(1)
    monobert.load_weights(script_path + "/../kth_w/monobert_00001_dcs_plus_weights")

    ds = data_generator.get_dcs_dataset(data_path + "test.desc.h5", data_path + "test.tokens.h5",
                                        vocab_desc, vocab_code, max_len=-1).batch(100, drop_remainder=True)

    print("Test 500")
    monobert.test(ds, "results/monobert", 100)
