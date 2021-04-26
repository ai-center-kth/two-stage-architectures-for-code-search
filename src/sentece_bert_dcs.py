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
from tensorflow.keras import backend as K
import pathlib
from dcs_data_generator import DataGeneratorDCS
import random
from help import *
from code_search_manager import CodeSearchManager
from bert.tokenization import FullTokenizer

class SBERT_DCS(CodeSearchManager):

    def __init__(self, data_path, data_chunk_id=0):

        self.tokenizer = None

        self.data_path = data_path
        self.max_len = 90
        # dataset info
        self.total_length = 18223872
        self.chunk_size = 100000   # 18223872  # 10000


        number_chunks = self.total_length / self.chunk_size - 1
        self.number_chunks = int(number_chunks + 1 if number_chunks > int(number_chunks) else number_chunks)

        self.data_chunk_id = min(data_chunk_id, int(self.number_chunks))
        print("### Loading SBERT model with DCS chunk number " + str(data_chunk_id) + " [0," + str(number_chunks)+"]")

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


    def generate_model(self, bert_layer):

        input_word_ids_desc = tf.keras.layers.Input(shape=(self.max_len,),
                                                    dtype=tf.int32,
                                                    name="input_word_ids_desc")
        input_mask_desc = tf.keras.layers.Input(shape=(self.max_len,),
                                                dtype=tf.int32,
                                                name="input_mask_desc")
        segment_ids_desc = tf.keras.layers.Input(shape=(self.max_len,),
                                                 dtype=tf.int32,
                                                 name="segment_ids_desc")

        bert_desc_output = bert_layer([input_word_ids_desc, input_mask_desc, segment_ids_desc])


        desc_output = tf.reduce_mean(bert_desc_output[1], 1)

        input_word_ids_code = tf.keras.layers.Input(shape=(self.max_len,),
                                                    dtype=tf.int32,
                                                    name="input_word_ids_code")
        input_mask_code = tf.keras.layers.Input(shape=(self.max_len,),
                                                dtype=tf.int32,
                                                name="input_mask_code")
        segment_ids_code = tf.keras.layers.Input(shape=(self.max_len,),
                                                 dtype=tf.int32,
                                                 name="segment_ids_code")

        bert_code_output = bert_layer([input_word_ids_code, input_mask_code, segment_ids_code])

        code_output = tf.reduce_mean(bert_code_output[1], 1)

        similarity = tf.keras.layers.Dot(axes=1, normalize=True)([desc_output, code_output])

        # Used in tests
        embedded_code = tf.keras.Input(shape=(code_output.shape[1],), name="embedded_code")
        embedded_desc = tf.keras.Input(shape=(desc_output.shape[1],), name="embedded_desc")

        dot = tf.keras.layers.Dot(axes=1, normalize=True)([embedded_code, embedded_desc])
        dot_model = tf.keras.Model(inputs=[embedded_code, embedded_desc], outputs=[dot],
                                   name='dot_model')

        # output = tf.keras.layers.Dense(1, activation="sigmoid")(dropout)

        cos_model = tf.keras.models.Model(
            inputs=[input_word_ids_desc, input_mask_desc, segment_ids_desc,
                    input_word_ids_code, input_mask_code, segment_ids_code],
            outputs=similarity
        )

        # cos_model.compile(loss='mse', optimizer='nadam', metrics=['mse'])

        embedding_desc_model = tf.keras.models.Model(
            inputs=[input_word_ids_desc, input_mask_desc, segment_ids_desc],
            outputs=desc_output
        )

        embedding_code_model = tf.keras.models.Model(
            inputs=[input_word_ids_code, input_mask_code, segment_ids_code],
            outputs=code_output
        )

        good_ids_desc = tf.keras.layers.Input(shape=(self.max_len,),
                                              dtype=tf.int32)
        good_mask_desc = tf.keras.layers.Input(shape=(self.max_len,),
                                               dtype=tf.int32)
        good_seg_desc = tf.keras.layers.Input(shape=(self.max_len,),
                                              dtype=tf.int32)

        good_ids_code = tf.keras.layers.Input(shape=(self.max_len,),
                                              dtype=tf.int32)
        good_mask_code = tf.keras.layers.Input(shape=(self.max_len,),
                                               dtype=tf.int32)
        good_seg_code = tf.keras.layers.Input(shape=(self.max_len,),
                                              dtype=tf.int32)

        bad_ids_code = tf.keras.layers.Input(shape=(self.max_len,),
                                             dtype=tf.int32)
        bad_mask_code = tf.keras.layers.Input(shape=(self.max_len,),
                                              dtype=tf.int32)
        bad_seg_code = tf.keras.layers.Input(shape=(self.max_len,),
                                             dtype=tf.int32)

        good_similarity = cos_model(
            [good_ids_desc, good_mask_desc, good_seg_desc, good_ids_code, good_mask_code, good_seg_code])

        bad_similarity = cos_model(
            [good_ids_desc, good_mask_desc, good_seg_desc, bad_ids_code, bad_mask_code, bad_seg_code])

        hinge_loss_margin = 0.2
        loss = tf.keras.layers.Lambda(lambda x: K.maximum(1e-6, hinge_loss_margin - x[0] + x[1]),
                                      output_shape=lambda x: x[0],
                                      name='loss')([good_similarity, bad_similarity])

        training_model = tf.keras.Model(inputs=[
            good_ids_desc, good_mask_desc, good_seg_desc,
            good_ids_code, good_mask_code, good_seg_code,

            bad_ids_code, bad_mask_code, bad_seg_code], outputs=[loss],
            name='training_model')

        training_model.compile(loss=lambda y_true, y_pred: y_pred + y_true - y_true, optimizer='adam')

        return training_model, embedding_code_model, embedding_desc_model, dot_model






# snn_dcs_weights

    def test(self, model_code, model_query, dot_model, results_path):
        test_tokens = load_hdf5(self.data_path + "test.tokens.h5" , 0, 100)
        test_desc = load_hdf5(self.data_path + "test.desc.h5" , 0, 100) # 10000

        code_test_vector = test_tokens
        desc_test_vector = test_desc

        embedded_tokens = []
        embedded_desc = []

        print("Embedding tokens and desc...")
        for idx, token in enumerate(code_test_vector):
            desc = (" ".join([vocab_desc[x] for x in desc_test_vector[idx]]))
            code = (" ".join([vocab_tokens[x] for x in code_test_vector[idx]]))

            desc_ = self.tokenize_sentences(desc, "")
            code_ = self.tokenize_sentences(code, "")

            embedded_tokens.append(model_code.predict([np.array(code_[0]).reshape((1, -1)),
                                                       np.array(code_[1]).reshape((1, -1)),
                                                       np.array(code_[2]).reshape((1, -1))

                                                       ])[0])

            embedded_desc.append(model_query.predict([np.array(desc_[0]).reshape((1, -1)),
                                                     np.array(desc_[1]).reshape((1, -1)),
                                                     np.array(desc_[2]).reshape((1, -1))

                                                     ])[0])

        self.test_embedded(dot_model, embedded_tokens, embedded_desc, results_path)



    def training_data_chunk(self, id, valid_perc):

        init_trainig = self.chunk_size * id
        init_valid = int(self.chunk_size * id + self.chunk_size * valid_perc)
        end_valid = int(self.chunk_size * id + self.chunk_size)

        return init_trainig, init_valid, end_valid

    def encode_sentence(self, s):
        if s == "":
            return []
        tokens = list(self.tokenizer.tokenize(s))
        tokens.append('[SEP]')
        return self.tokenizer.convert_tokens_to_ids(tokens)

    def tokenize_sentences(self, input1_str, input2_str):
        input1_encoded = self.encode_sentence(input1_str)
        input2_encoded = self.encode_sentence(input2_str)
        cls_ = self.tokenizer.convert_tokens_to_ids(['[CLS]'])
        concated = cls_ + input1_encoded + input2_encoded
        concated_ids = concated + [0] * ((self.max_len) - len(concated))

        masks = [1] * len(concated) + [0] * ((self.max_len) - len(concated))
        type_ids = [0] + [0] * len(input1_encoded) + [1] * len(input2_encoded) + [0] * (
                    (self.max_len) - (1 + len(input1_encoded) + len(input2_encoded)))

        return concated_ids, masks, type_ids

    def load_dataset(self, train_desc, train_tokens, vocab_desc, vocab_tokens):

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

        for idx, sentence in enumerate(train_desc):

            desc = (" ".join([vocab_desc[x] for x in train_desc[idx]]))
            code = (" ".join([vocab_tokens[x] for x in train_tokens[idx]]))

            random_code = train_tokens[random.randint(0, len(train_tokens) - 1)]
            neg_code = (" ".join([vocab_tokens[x] for x in random_code]))

            desc_ = self.tokenize_sentences(desc, "")
            code_ = self.tokenize_sentences(code, "")
            neg_ = self.tokenize_sentences(neg_code, "")

            if len(desc_[0]) != self.max_len or len(desc_[0]) != self.max_len or len(desc_[0]) != self.max_len:
                continue

            retokenized_desc.append(desc_[0])
            retokenized_mask_desc.append(desc_[1])
            retokenized_type_desc.append(desc_[2])

            retokenized_code.append(code_[0])
            retokenized_mask_code.append(code_[1])
            retokenized_type_code.append(code_[2])

            bad_retokenized_code.append(neg_[0])
            bad_retokenized_mask_code.append(neg_[1])
            bad_retokenized_type_code.append(neg_[2])

            # labels.append([0])

        labels = np.zeros((len(bad_retokenized_code), 1))

        return np.array(retokenized_desc), np.array(retokenized_mask_desc), np.array(retokenized_type_desc),\
               np.array(retokenized_code), np.array(retokenized_mask_code), np.array(retokenized_type_code),\
               np.array(bad_retokenized_code), np.array(bad_retokenized_mask_code), np.array(bad_retokenized_type_code),labels

    def train(self, trainig_model, training_set, weights_path, epochs=1):
        trainig_model.fit(x=[np.array(training_set[0]),
                     np.array(training_set[1]),
                     np.array(training_set[2]),

                     np.array(training_set[3]),
                     np.array(training_set[4]),
                     np.array(training_set[5]),

                     np.array(training_set[6]),
                     np.array(training_set[7]),
                     np.array(training_set[8]),
                     ],  # np.array(tokenized_code)
                  y=training_set[9], epochs=epochs, verbose=1, batch_size=32)

        trainig_model.save_weights(weights_path)
        print("Model saved!")

if __name__ == "__main__":

    args = sys.argv
    data_chunk_id = 0
    if len(args) > 1:
        data_chunk_id = int(args[1])

    script_path = str(pathlib.Path(__file__).parent)

    data_path = script_path + "/../data/deep-code-search/drive/"

    sbert_dcs = SBERT_DCS(data_path, data_chunk_id)

    BATCH_SIZE = 32 * 1

    bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1",
                                trainable=False)
    # Some verion incompatibility requires this line
    tf.gfile = tf.io.gfile

    # Get bert tokenizer
    bert_layer.resolved_object.vocab_file.asset_path.numpy()
    vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
    do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
    tokenizer = FullTokenizer(vocab_file, do_lower_case)

    sbert_dcs.tokenizer = tokenizer

    file_format = "h5"

    # 18223872 (len) #1000000
    train_tokens = load_hdf5(data_path + "train.tokens." + file_format, 0, 100000)  # 1000000
    train_desc = load_hdf5(data_path + "train.desc." + file_format, 0, 100000)

    vocabulary_tokens = load_pickle(data_path + "vocab.tokens.pkl")
    vocab_tokens = {y: x for x, y in vocabulary_tokens.items()}

    vocabulary_desc = load_pickle(data_path + "vocab.desc.pkl")
    vocab_desc = {y: x for x, y in vocabulary_desc.items()}

    dataset = sbert_dcs.load_dataset(train_desc, train_tokens, vocab_desc, vocab_tokens)

    multi_gpu = False

    print("Building model and loading weights")
    if multi_gpu:
        tf.debugging.set_log_device_placement(False)

        strategy = tf.distribute.MirroredStrategy()

        with strategy.scope():
            training_model, model_code, model_query, dot_model = sbert_dcs.generate_model(bert_layer)
            #sbert_dcs.load_weights(training_model, script_path+"/../weights/sbert_dcs_weights")
    else:
        training_model, model_code, model_query, dot_model = sbert_dcs.generate_model(bert_layer)
        #sbert_dcs.load_weights(training_model, script_path + "/../weights/sbert_dcs_weights")


    bert_layer.trainable = True
    sbert_dcs.train(training_model, dataset, script_path+"/../weights/sbert_dcs_weights", 1)

    sbert_dcs.test(model_code, model_query, dot_model, script_path+"/../results")

    sbert_dcs.train(training_model, dataset, script_path+"/../weights/sbert_dcs_weights", 1)

    sbert_dcs.test(model_code, model_query, dot_model, script_path+"/../results")

