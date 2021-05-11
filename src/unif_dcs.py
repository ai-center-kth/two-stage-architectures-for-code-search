import subprocess
import sys

subprocess.check_call([sys.executable, "-m", "pip", "install", "tables"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "tqdm"])

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

import tensorflow as tf
from tensorflow.keras import backend as K
import pathlib
from data_generators.dcs_data_generator import DataGeneratorDCS
import help
from code_search_manager import CodeSearchManager
import numpy as np

class UNIF_DCS(CodeSearchManager):

    def __init__(self, data_path, data_chunk_id=0):
        self.data_path = data_path

        # dataset info
        self.total_length = 18223872
        self.chunk_size = 18223872   # 18223872  # 10000

        number_chunks = self.total_length / self.chunk_size - 1
        self.number_chunks = int(number_chunks + 1 if number_chunks > int(number_chunks) else number_chunks)

        self.data_chunk_id = min(data_chunk_id, int(self.number_chunks))
        print("### Loading UNIF model with DCS chunk number " + str(data_chunk_id) + " [0," + str(number_chunks)+"]")

    def get_dataset_meta_hardcoded(self):
        return 86, 410, 10001, 10001

    def get_dataset_meta(self):
        # 18223872 (len) #1000000
        code_vector = help.load_hdf5(self.data_path + "train.tokens.h5", 0, 18223872)
        desc_vector = help.load_hdf5(self.data_path + "train.desc.h5", 0, 18223872)
        vocabulary_merged = help.load_pickle(data_path + "vocab.merged.pkl")

        longer_code = max(len(t) for t in code_vector)
        print("longer_code", longer_code)
        longer_desc = max(len(t) for t in desc_vector)
        print("longer_desc", longer_desc)

        longer_sentence = max(longer_code, longer_desc)

        number_tokens = len(vocabulary_merged)

        return longer_sentence, number_tokens


    def generate_model(self, embedding_size, number_code_tokens, number_desc_tokens, code_length, desc_length, hinge_loss_margin):

        code_input = tf.keras.Input(shape=(code_length,), name="code_input")
        code_embeding = tf.keras.layers.Embedding(number_code_tokens, embedding_size, name="code_embeding")(code_input)

        attention_code = tf.keras.layers.Attention(name="attention_code")([code_embeding, code_embeding])

        query_input = tf.keras.Input(shape=(desc_length,), name="query_input")
        query_embeding = tf.keras.layers.Embedding(number_desc_tokens, embedding_size, name="query_embeding")(
            query_input)

        code_output = tf.keras.layers.Lambda(lambda x: K.sum(x, axis=1), name="sum")(attention_code)
        query_output = tf.keras.layers.Lambda(lambda x: K.mean(x, axis=1), name="average")(query_embeding)

        # This model generates code embedding
        model_code = tf.keras.Model(inputs=[code_input], outputs=[code_output], name='model_code')
        # This model generates description/query embedding
        model_query = tf.keras.Model(inputs=[query_input], outputs=[query_output], name='model_query')

        # Cosine similarity
        # If normalize set to True, then the output of the dot product is the cosine proximity between the two samples.
        cos_sim = tf.keras.layers.Dot(axes=1, normalize=True, name='cos_sim')([code_output, query_output])

        # This model calculates cosine similarity between code and query pairs
        cos_model = tf.keras.Model(inputs=[code_input, query_input], outputs=[cos_sim], name='sim_model')

        # Used in tests
        embedded_code = tf.keras.Input(shape=(code_output.shape[1],), name="embedded_code")
        embedded_desc = tf.keras.Input(shape=(query_output.shape[1],), name="embedded_desc")

        dot = tf.keras.layers.Dot(axes=1, normalize=True)([embedded_code, embedded_desc])
        dot_model = tf.keras.Model(inputs=[embedded_code, embedded_desc], outputs=[dot],
                                        name='dot_model')

        loss = tf.keras.layers.Flatten()(cos_sim)
        # training_model = tf.keras.Model(inputs=[ code_input, query_input], outputs=[cos_sim],name='training_model')

        model_code.compile(loss='cosine_proximity', optimizer='adam')
        model_query.compile(loss='cosine_proximity', optimizer='adam')

        cos_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=["accuracy"])  # extract similarity

        # Negative sampling
        good_desc_input = tf.keras.Input(shape=(desc_length,), name="good_desc_input")
        bad_desc_input = tf.keras.Input(shape=(desc_length,), name="bad_desc_input")

        good_desc_output = cos_model([code_input, good_desc_input])
        bad_desc_output = cos_model([code_input, bad_desc_input])

        loss = tf.keras.layers.Lambda(lambda x: K.maximum(1e-6, hinge_loss_margin - x[0] + x[1]),
                                      output_shape=lambda x: x[0],
                                      name='loss')([good_desc_output, bad_desc_output])

        training_model = tf.keras.Model(inputs=[code_input, good_desc_input, bad_desc_input], outputs=[loss],
                                        name='training_model')

        opt = tf.keras.optimizers.Adam(learning_rate=0.0001)

        training_model.compile(loss=lambda y_true, y_pred: y_pred + y_true - y_true, optimizer=opt)
        # y_true-y_true avoids warning

        return training_model, model_code, model_query, dot_model



    def test(self, model_code, model_query, dot_model, results_path, code_length, desc_length, number_of_elements=100):
        test_tokens = help.load_hdf5(self.data_path + "test.tokens.h5" , 0, number_of_elements)
        test_desc = help.load_hdf5(self.data_path + "test.desc.h5" , 0, number_of_elements) # 10000

        test_tokens = help.pad(test_tokens, code_length)
        test_desc = help.pad(test_desc, desc_length)

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


    def load_dataset(self, data_chunk_id, batch_size):

        init_trainig, init_valid, end_valid = self.training_data_chunk(data_chunk_id, 1)

        longer_code, longer_desc, number_code_tokens, number_desc_tokens= self.get_dataset_meta_hardcoded()

        training_set_generator = DataGeneratorDCS(self.data_path + "train.tokens.h5", self.data_path + "train.desc.h5",
                                                  batch_size, init_trainig, init_valid, longer_code, longer_desc)
        return training_set_generator

if __name__ == "__main__":

    print("Running UNIF Model")

    args = sys.argv
    data_chunk_id = 0
    if len(args) > 1:
        data_chunk_id = int(args[1])

    script_path = str(pathlib.Path(__file__).parent)

    data_path = script_path + "/../data/deep-code-search/drive/"

    unif_dcs = UNIF_DCS(data_path, data_chunk_id)

    BATCH_SIZE = 32 * 4 * 2

    dataset = unif_dcs.load_dataset(0, BATCH_SIZE)

    longer_code, longer_desc, number_code_tokens, number_desc_tokens= unif_dcs.get_dataset_meta_hardcoded()

    embedding_size = 2048

    multi_gpu = True

    print("Building model and loading weights")
    if multi_gpu:
        tf.debugging.set_log_device_placement(False)

        strategy = tf.distribute.MirroredStrategy()

        with strategy.scope():
            training_model, model_code, model_query, dot_model = unif_dcs.generate_model(embedding_size, number_code_tokens, number_desc_tokens, longer_code, longer_desc, 0.05)
            #unif_dcs.load_weights(training_model, script_path+"/../weights/unif_dcs_weights")
    else:
        training_model, model_code, model_query, dot_model = unif_dcs.generate_model(embedding_size, number_code_tokens,
                                                                                     number_desc_tokens, longer_code,
                                                                                     longer_desc, 0.5)
        #unif_dcs.load_weights(training_model, script_path + "/../weights/unif_dcs_weights")


    print("Not trained results")
    unif_dcs.test(model_code, model_query, dot_model, script_path+"/../results/unif-dcs", longer_code, longer_desc, 100)

    print("First epoch")
    unif_dcs.train(training_model, dataset, script_path+"/../weights/unif_dcs_weights", 1)

    print("Trained results with 100")
    unif_dcs.test(model_code, model_query, dot_model, script_path+"/../results/sunif-dcs", longer_code, longer_desc, 100)

    print("Trained results with 200")
    unif_dcs.test(model_code, model_query, dot_model, script_path+"/../results/unif-dcs", longer_code, longer_desc, 200)

