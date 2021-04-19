import subprocess
import sys

subprocess.check_call([sys.executable, "-m", "pip", "install", "tables"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "tqdm"])

subprocess.check_call([sys.executable, "-m", "pip", "install", "bert-tensorflow==1.0.1"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "tf-hub-nightly"])

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

import tensorflow as tf
import tensorflow_hub as hub
from bert.tokenization import FullTokenizer
import pathlib
from dcs_bert_data_generator import DataGeneratorDCSBERT
from help import *
from code_search_manager import CodeSearchManager

class BERT_DCS(CodeSearchManager):

    def __init__(self, data_path, data_chunk_id=0):
        self.data_path = data_path

        # dataset info
        self.total_length = 18223872
        self.chunk_size = 10000 #// 2  # 18223872  # 10000

        number_chunks = self.total_length / self.chunk_size - 1
        self.number_chunks = int(number_chunks + 1 if number_chunks > int(number_chunks) else number_chunks)

        self.data_chunk_id = min(data_chunk_id, int(self.number_chunks))
        print("### Loading BERT model with DCS chunk number " + str(data_chunk_id) + " [0," + str(number_chunks)+"]")

        vocab_code_pckl = load_pickle(data_path + "vocab.tokens.pkl")
        self.vocab_code = {y: x for x, y in vocab_code_pckl.items()}

        vocab_desc_pckl = load_pickle(data_path + "vocab.desc.pkl")
        self.vocab_desc = {y: x for x, y in vocab_desc_pckl.items()}


    def get_dataset_meta_hardcoded(self):
        return 86, 410, 10001, 10001

    def get_dataset_meta(self):
        # 18223872 (len) #1000000
        code_vector = load_hdf5(data_path + "train.tokens.h5", 0, 18223872)
        desc_vector = load_hdf5(data_path + "train.desc.h5", 0, 18223872)
        vocabulary_merged = load_pickle(data_path + "vocab.merged.pkl")

        longer_code = max(len(t) for t in code_vector)
        print("longer_code", longer_code)
        longer_desc = max(len(t) for t in desc_vector)
        print("longer_desc", longer_desc)

        longer_sentence = max(longer_code, longer_desc)

        number_tokens = len(vocabulary_merged)

        return longer_sentence, number_tokens


    def generate_model(self, max_len, bert_layer):

        input_word_ids = tf.keras.layers.Input(shape=(max_len,),
                                               dtype=tf.int32,
                                               name="input_word_ids")
        input_mask = tf.keras.layers.Input(shape=(max_len,),
                                           dtype=tf.int32,
                                           name="input_mask")
        segment_ids = tf.keras.layers.Input(shape=(max_len,),
                                            dtype=tf.int32,
                                            name="segment_ids")

        pooled_output, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])

        dropout = tf.keras.layers.Dropout(0.3)(pooled_output)

        output = tf.keras.layers.Dense(1, activation="sigmoid")(dropout)

        model = tf.keras.models.Model(
            inputs=[input_word_ids, input_mask, segment_ids], outputs=output
        )

        model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss="binary_crossentropy",
            metrics=["acc"],
        )

        return model, bert_layer

# snn_dcs_weights

    def test(self, model_code, model_query, dot_model, results_path, code_length, desc_length):
        test_tokens = load_hdf5(self.data_path + "test.tokens.h5" , 0, 500)
        test_desc = load_hdf5(self.data_path + "test.desc.h5" , 0, 500) # 10000

        test_tokens = pad(test_tokens, code_length)
        test_desc = pad(test_desc, desc_length)

        embedding_tokens = [None] * len(test_tokens)
        print("Embedding tokens...")
        for idx,token in enumerate(test_tokens):

            embedding_result = model_code(np.array(token).reshape(1,-1))
            embedding_tokens[idx] = embedding_result.numpy()[0]

        embedding_desc = [None] * len(test_desc)
        print("Embedding descs...")
        for idx,desc in enumerate(test_desc):

            embedding_result = model_query(np.array(desc).reshape(1,-1))
            embedding_desc[idx] = embedding_result.numpy()[0]

        self.test_embedded(dot_model, embedding_tokens, embedding_desc, results_path)



    def training_data_chunk(self, id, valid_perc):

        init_trainig = self.chunk_size * id
        init_valid = int(self.chunk_size * id + self.chunk_size * valid_perc)
        end_valid = int(self.chunk_size * id + self.chunk_size)

        return init_trainig, init_valid, end_valid


    def load_dataset(self, data_chunk_id, batch_size, tokenizer, max_len):

        init_trainig, init_valid, end_valid = self.training_data_chunk(data_chunk_id, 0.8)

        longer_code, longer_desc, number_code_tokens, number_desc_tokens= snn_dcs.get_dataset_meta_hardcoded()


        # tokens_path, desc_path, batch_size, init_pos, last_pos, max_length, tokenizer, vocab_code, vocab_desc)
        training_set_generator = DataGeneratorDCSBERT(self.data_path + "train.tokens.h5", self.data_path + "train.desc.h5",
                                                  batch_size, init_trainig, init_valid, max_len, tokenizer, self.vocab_code, self.vocab_desc)
        return training_set_generator

if __name__ == "__main__":

    print("Running SNN Model")

    args = sys.argv
    data_chunk_id = 0
    if len(args) > 1:
        data_chunk_id = int(args[1])

    script_path = str(pathlib.Path(__file__).parent)

    data_path = script_path + "/../data/deep-code-search/drive/"

    snn_dcs = BERT_DCS(data_path, data_chunk_id)

    BATCH_SIZE = 32 * 1


    longer_code, longer_desc, number_code_tokens, number_desc_tokens= snn_dcs.get_dataset_meta_hardcoded()

    max_len = 90
    multi_gpu = False

    print("Building model and loading weights")
    if multi_gpu:
        tf.debugging.set_log_device_placement(False)

        strategy = tf.distribute.MirroredStrategy()

        with strategy.scope():
            bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1",
                                        trainable=False)
            training_model, bert_layer = snn_dcs.generate_model(max_len, bert_layer)
            snn_dcs.load_weights(training_model, script_path+"/../weights/bert_dcs_weights")
    else:
        bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1",
                                    trainable=False)
        training_model, bert_layer = snn_dcs.generate_model(max_len, bert_layer)
        snn_dcs.load_weights(training_model, script_path + "/../weights/bert_dcs_weights")

    # Get tokenizer
    tf.gfile = tf.io.gfile
    bert_layer.resolved_object.vocab_file.asset_path.numpy()

    vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
    do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
    tokenizer = FullTokenizer(vocab_file, do_lower_case)

    dataset = snn_dcs.load_dataset(0, BATCH_SIZE, tokenizer, max_len)

    snn_dcs.train(training_model, dataset, script_path+"/../weights/bert_dcs_weights")

    #snn_dcs.test(model_code, model_query, dot_model, script_path+"/../results", longer_code, longer_desc)


