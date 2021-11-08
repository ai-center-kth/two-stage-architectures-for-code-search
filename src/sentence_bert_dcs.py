
import sys
import pathlib

import transformers
import tensorflow as tf

from .search_models.sentence_bert_search_model import SentenceBERT_SearchModel
from .search_models import models
from . import data_generator, helper

class SBERT_DCS(SentenceBERT_SearchModel):
    def __init__(self, data_path):
        self.data_path = data_path
        self.max_len = 90
        self.bert_layer = None
        self.tokenizer = None

    def get_model(self):
        self.training_model, self.model_code, self.model_query, self.dot_model = models.sentence_bert_model(self.bert_layer, self.max_len)

    def generate_tokenizer(self):
        self.tokenizer = transformers.RobertaTokenizer.from_pretrained('roberta-base', do_lower_case=True)
        return self.tokenizer

    def generate_bert_layer(self):
        self.bert_layer = transformers.TFRobertaModel.from_pretrained('roberta-base')
        return self.bert_layer

    def get_vocabularies(self):
        self.inverse_vocab_code = helper.load_pickle(self.data_path + "vocab.merged.pkl")
        self.vocab_code = {y: x for x, y in self.inverse_vocab_code.items()}

        self.inverse_vocab_desc = helper.load_pickle(self.data_path + "vocab.merged.pkl")
        self.vocab_desc = {y: x for x, y in self.inverse_vocab_desc.items()}

        return self.vocab_code, self.vocab_desc


    def tokenize(self, input_str):
        _input_str1 = input_str
        if isinstance(input_str, str):
            _input_str1 = [input_str]

        tokenized = self.tokenizer.batch_encode_plus(
            _input_str1,
            add_special_tokens=True,
            max_length=self.max_len,
            return_attention_mask=True,
            return_token_type_ids=True,
            return_tensors="np",
            padding='max_length',
            truncation=True
        )
        if isinstance(input_str, str):
            return tokenized["input_ids"][0], tokenized["attention_mask"][0], tokenized["token_type_ids"][0]
        return tokenized["input_ids"], tokenized["attention_mask"], tokenized["token_type_ids"]

    def load_dataset(self, desc_path, code_path, vocab_desc, vocab_code, batch_size=16):

        ds = data_generator.get_dcs_dataset(desc_path, code_path, vocab_desc, vocab_code, max_len=-1)

        ds = ds.map(data_generator.sentece_bert_tokenizer_map(self.tokenize, self.max_len))

        ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        ds = ds.batch(batch_size, drop_remainder=True)

        return ds


if __name__ == "__main__":

    args = sys.argv
    data_chunk_id = 0
    if len(args) > 1:
        data_chunk_id = int(args[1])

    script_path = str(pathlib.Path(__file__).parent)

    data_path = script_path + "/../data/deep-code-search/processed/"

    sbert = SBERT_DCS(data_path)

    vocab_code, vocab_desc = sbert.get_vocabularies()

    sbert.generate_tokenizer()
    sbert.generate_bert_layer()
    sbert.get_model()

    BATCH_SIZE = 16

    steps_per_epoch = data_generator.DCS_NUM_ELEMENTS // BATCH_SIZE

    #unif.train(ds, script_path + "/../weights/unif-weights", epochs=1, steps_per_epoch=steps_per_epoch)

    sbert.load_weights(script_path+"/../kth_w/sroberta_600k_00002_m20_dcs_weights")

    test_ds = sbert.load_dataset(data_path+"/test.desc.h5", data_path+"/test.tokens.h5", vocab_desc, vocab_code, 100)

    sbert.test(test_ds, script_path+"/../results/snn-results")
