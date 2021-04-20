import subprocess
import sys

subprocess.check_call([sys.executable, "-m", "pip", "install", "tables"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "tqdm"])

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

import tensorflow as tf
from tensorflow.keras import backend as K
import pathlib
from dcs_data_generator import DataGeneratorDCS
from help import *
from code_search_manager import CodeSearchManager

class SNN_DCS(CodeSearchManager):

    def __init__(self, data_path, data_chunk_id=0):
        self.data_path = data_path

        # dataset info
        self.total_length = 18223872
        self.chunk_size = 10000 // 2  # 18223872  # 10000


        number_chunks = self.total_length / self.chunk_size - 1
        self.number_chunks = int(number_chunks + 1 if number_chunks > int(number_chunks) else number_chunks)

        self.data_chunk_id = min(data_chunk_id, int(self.number_chunks))
        print("### Loading SNN model with DCS chunk number " + str(data_chunk_id) + " [0," + str(number_chunks)+"]")

    def get_dataset_meta_hardcoded(self):
        return 410, 13645

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


    def generate_model(self, embedding_size, number_tokens, sentence_length, hinge_loss_margin):
        input_layer = tf.keras.Input(shape=(sentence_length,), name="input")
        embedding_layer = tf.keras.layers.Embedding(number_tokens, embedding_size, name="embeding")(input_layer)

        attention_layer = tf.keras.layers.Attention(name="attention")([embedding_layer, embedding_layer])

        sum_layer = tf.keras.layers.Lambda(lambda x: K.sum(x, axis=1), name="sum")(attention_layer)
        # average_layer = tf.keras.layers.Lambda(lambda x: K.mean(x, axis=1), name="average")( attention_layer)

        embedding_model = tf.keras.Model(inputs=[input_layer], outputs=[sum_layer], name='siamese_model')

        input_code = tf.keras.Input(shape=(sentence_length,), name="code")
        input_desc = tf.keras.Input(shape=(sentence_length,), name="desc")
        input_bad_desc = tf.keras.Input(shape=(sentence_length,), name="bad_desc")

        output_code = embedding_model(input_code)
        output_desc = embedding_model(input_desc)
        output_bad_desc = embedding_model(input_bad_desc)

        cos_good_sim = tf.keras.layers.Dot(axes=1, normalize=True, name='cos_good_sim')([output_code, output_desc])

        cos_model = tf.keras.Model(inputs=[input_code, input_desc], outputs=[cos_good_sim],
                                        name='cos_model')


        # Used in tests
        embedded_code = tf.keras.Input(shape=(output_code.shape[1],), name="embedded_code")
        embedded_desc = tf.keras.Input(shape=(output_code.shape[1],), name="embedded_desc")

        dot = tf.keras.layers.Dot(axes=1, normalize=True)([embedded_code, embedded_desc])
        dot_model = tf.keras.Model(inputs=[embedded_code, embedded_desc], outputs=[dot],
                                        name='dot_model')

        cos_bad_sim = tf.keras.layers.Dot(axes=1, normalize=True, name='cos_bad_sim')([output_code, output_bad_desc])

        loss = tf.keras.layers.Lambda(lambda x: K.maximum(1e-6, hinge_loss_margin - x[0] + x[1]),
                                      output_shape=lambda x: x[0],
                                      name='loss')([cos_good_sim, cos_bad_sim])

        training_model = tf.keras.Model(inputs=[input_code, input_desc, input_bad_desc], outputs=[loss],
                                        name='training_model')

        training_model.compile(loss=lambda y_true, y_pred: y_pred + y_true - y_true, optimizer='adam')
        # y_true-y_true avoids warning

        return training_model, embedding_model, cos_model, dot_model

# snn_dcs_weights

    def test(self, embedding_model, dot_model, results_path, code_length, desc_length):
        test_tokens = load_hdf5(self.data_path + "test.tokens.h5" , 0, 100)
        test_desc = load_hdf5(self.data_path + "test.desc.h5" , 0, 100) # 10000

        test_tokens = pad(test_tokens, code_length)
        test_desc = pad(test_desc, desc_length)

        embedding_tokens = [None] * len(test_tokens)
        print("Embedding tokens...")
        for idx,token in enumerate(test_tokens):

            embedding_result = embedding_model(np.array(token).reshape(1,-1))
            embedding_tokens[idx] = embedding_result.numpy()[0]

        embedding_desc = [None] * len(test_desc)
        print("Embedding descs...")
        for idx,desc in enumerate(test_desc):

            embedding_result = embedding_model(np.array(desc).reshape(1,-1))
            embedding_desc[idx] = embedding_result.numpy()[0]

        self.test_embedded(dot_model, embedding_tokens, embedding_desc, results_path)



    def training_data_chunk(self, id, valid_perc):

        init_trainig = self.chunk_size * id
        init_valid = int(self.chunk_size * id + self.chunk_size * valid_perc)
        end_valid = int(self.chunk_size * id + self.chunk_size)

        return init_trainig, init_valid, end_valid


    def load_dataset(self, data_chunk_id, batch_size):

        init_trainig, init_valid, end_valid = self.training_data_chunk(data_chunk_id, 0.8)

        longer_sentence, number_tokens = snn_dcs.get_dataset_meta_hardcoded()

        training_set_generator = DataGeneratorDCS(self.data_path + "train.tokens.h5", self.data_path + "train.desc.h5",
                                                  batch_size, init_trainig, init_valid, longer_sentence, longer_sentence)
        return training_set_generator

if __name__ == "__main__":

    print("Running SNN Model")

    args = sys.argv
    data_chunk_id = 0
    if len(args) > 1:
        data_chunk_id = int(args[1])

    script_path = str(pathlib.Path(__file__).parent)

    data_path = script_path + "/../data/deep-code-search/processed/"

    snn_dcs = SNN_DCS(data_path, data_chunk_id)

    BATCH_SIZE = 32 * 1

    dataset = snn_dcs.load_dataset(0, BATCH_SIZE)

    longer_sentence, number_tokens = snn_dcs.get_dataset_meta_hardcoded()

    embedding_size = 2048

    multi_gpu = False

    print("Building model and loading weights")
    if multi_gpu:
        tf.debugging.set_log_device_placement(False)

        strategy = tf.distribute.MirroredStrategy()

        with strategy.scope():
            training_model, embedding_model, cos_model, dot_model = snn_dcs.generate_model(embedding_size, number_tokens, longer_sentence, 0.05)
            snn_dcs.load_weights(training_model, script_path+"/../weights/snn_dcs_weights")
    else:
        training_model, embedding_model, cos_model, dot_model = snn_dcs.generate_model(embedding_size, number_tokens,
                                                                                       longer_sentence, 0.05)
        snn_dcs.load_weights(training_model, script_path + "/../weights/snn_dcs_weights")

    #snn_dcs.train(training_model, dataset, script_path+"/../weights/snn_dcs_weights")

    snn_dcs.test(embedding_model, dot_model, script_path+"/../results", longer_sentence, longer_sentence)


