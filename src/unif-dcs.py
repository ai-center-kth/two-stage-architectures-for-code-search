import subprocess
import sys

subprocess.check_call([sys.executable, "-m", "pip", "install", "tables"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "tqdm"])
#subprocess.check_call([sys.executable, "-m", "pip", "install", "pickle5"])


# import pickle5 as pickle
import tensorflow as tf
from tensorflow.keras import backend as K
import pickle
import sys
import tables
from tqdm import tqdm

import os.path
import time
import pathlib

from dcs_data_generator import DataGeneratorDCS
from help import *


def get_dataset_meta():
    # 18223872 (len) #1000000
    code_vector = load_hdf5(data_path + "train.tokens.h5", 0, 18223872)
    desc_vector = load_hdf5(data_path + "train.desc.h5", 0, 18223872)
    vocabulary_desc = load_pickle(data_path + "vocab.desc.pkl")
    vocabulary_tokens = load_pickle(data_path + "vocab.tokens.pkl")

    longer_code = max(len(t) for t in code_vector)
    longer_desc = max(len(t) for t in desc_vector)

    number_code_tokens = len(vocabulary_desc)
    number_desc_tokens = len(vocabulary_tokens)

    return longer_code, longer_desc, number_code_tokens, number_desc_tokens


def get_dataset_meta_hardcoded():
    return 86, 410, 10001, 10001


def generate_model(embedding_size, number_code_tokens, number_desc_tokens, code_length, desc_length, hinge_loss_margin):

    code_input = tf.keras.Input(shape=(code_length,), name="code_input")
    code_embeding = tf.keras.layers.Embedding(number_code_tokens, embedding_size, name="code_embeding")(code_input)

    attention_code = tf.keras.layers.Attention(name="attention_code")([code_embeding, code_embeding])

    query_input = tf.keras.Input(shape=(desc_length,), name="query_input")
    query_embeding = tf.keras.layers.Embedding(number_desc_tokens, embedding_size, name="query_embeding")(query_input)

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

    margin = 0.5
    loss = tf.keras.layers.Lambda(lambda x: K.maximum(1e-6, hinge_loss_margin - x[0] + x[1]), output_shape=lambda x: x[0],
                                  name='loss')([good_desc_output, bad_desc_output])

    training_model = tf.keras.Model(inputs=[code_input, good_desc_input, bad_desc_input], outputs=[loss],
                                    name='training_model')

    training_model.compile(loss=lambda y_true, y_pred: y_pred + y_true - y_true, optimizer='adam')
    # y_true-y_true avoids warning

    return training_model, model_code, model_query


def load_weights(model, path):
    if os.path.isfile(path+'/unif_dcs_weights.index'):
        model.load_weights(path+'/unif_dcs_weights')
        print("Weights loaded!")
    else:
        print("Warning!!  Weights not loaded")

# n >= 1
def get_top_n(n, results):
    count = 0
    for r in results:
        if results[r] < n:
            count+= 1
    return count / len(results)


def train(trainig_model, training_set_generator, valid_set_generator, weights_path, batch_size):
    trainig_model.fit(training_set_generator, epochs=1, validation_data=valid_set_generator, batch_size=batch_size)
    trainig_model.save_weights(weights_path)
    print("Model saved!")


def test(data_path, code_embedding_model, desc_embedding_model, results_path, code_length, desc_length, batch_id):

    # 10000
    test_tokens = load_hdf5(data_path + "test.tokens.h5" , 0, 10000)
    test_desc = load_hdf5(data_path + "test.desc.h5" , 0, 10000)

    test_tokens = pad(test_tokens, code_length)
    test_desc = pad(test_desc, desc_length)

    code_embeddings = []
    for idx, code_test in enumerate(test_tokens):

        code_rep = code_embedding_model.predict(code_test.reshape((1, -1)))

        code_embeddings.append(code_rep)

    desc_embeddings = []
    for idx, desc_test in enumerate(test_desc):

        desc_rep = desc_embedding_model.predict(desc_test.reshape((1, -1)))

        desc_embeddings.append(desc_rep)

    results = {}
    pbar = tqdm(total=len(desc_embeddings))

    for rowid, testvalue in enumerate(desc_embeddings):

        expected_best_result = \
            tf.keras.layers.Dot(axes=1, normalize=True)([code_embeddings[rowid], desc_embeddings[rowid]]).numpy()[0][0]

        count = 0

        # here we count the number of results with greater similarity with the expected best result
        for codeidx, codevalue in enumerate(code_embeddings):

            if not rowid == codeidx:

                new_result = \
                    tf.keras.layers.Dot(axes=1, normalize=True)(
                        [code_embeddings[codeidx], desc_embeddings[rowid]]).numpy()[
                        0][0]

                if new_result > expected_best_result:
                    count += 1

            # This break speeds up the process. Change the number if you want bigger "TopN results"
            if count > 5:
                break

        pbar.update(1)

        results[rowid] = count

    pbar.close()

    top_1 = get_top_n(1, results)
    top_3 = get_top_n(3, results)
    top_5 = get_top_n(5, results)

    print(top_1)
    print(top_3)
    print(top_5)

    name = results_path+"/results-unif-dcs-" + time.strftime("%Y%m%d-%H%M%S") + ".csv"

    f = open(name, "a")

    f.write("batch,top1,top3,top5\n")
    f.write(str(batch_id)+","+str(top_1) + "," + str(top_3) + "," + str(top_5) + "\n")
    f.close()


def training_data_chunk(id, valid_perc, chunk_size):

    init_trainig = chunk_size * id
    init_valid = int(chunk_size * id + chunk_size * valid_perc)
    end_valid = int(chunk_size * id + chunk_size)

    return init_trainig, init_valid, end_valid


if __name__ == "__main__":
    script_path = str(pathlib.Path(__file__).parent)

    print("UNIF Model")

    # dataset info
    total_length = 18223872
    chunk_size = 18223872 #1000000

    number_chunks = total_length/chunk_size - 1
    number_chunks = int(number_chunks + 1 if number_chunks > int(number_chunks) else number_chunks)

    args = sys.argv
    data_chunk_id = 0
    if len(args) > 1:
        data_chunk_id = int(args[1])

    data_chunk_id = min(data_chunk_id, int(number_chunks))

    data_path = script_path+"/../data/deep-code-search/drive/"

    #longer_code, longer_desc, number_code_tokens, number_desc_tokens = get_dataset_meta()
    longer_code, longer_desc, number_code_tokens, number_desc_tokens = get_dataset_meta_hardcoded()
    embedding_size = 2048

    print("Building model and loading weights")
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        training_model, model_code, model_query = generate_model(embedding_size, number_code_tokens, number_desc_tokens, longer_code, longer_desc, 0.05)

    load_weights(training_model, script_path+"/../weights")

    init_trainig, init_valid, end_valid = training_data_chunk(data_chunk_id, 0.8, chunk_size)

    print("Training model with chunk number ", data_chunk_id, " of ", number_chunks)

    batch_size = 64 * 2
    training_set_generator = DataGeneratorDCS(data_path + "train.tokens.h5", data_path + "train.desc.h5", batch_size, init_trainig, init_valid, longer_code, longer_desc)
    valid_set_generator = DataGeneratorDCS(data_path + "train.tokens.h5", data_path + "train.desc.h5", batch_size, init_valid, end_valid, longer_code, longer_desc)

    train(training_model, training_set_generator, valid_set_generator, script_path+"/../weights/unif_dcs_weights", batch_size)

    #test(data_path, model_code, model_query, script_path+"/../results", longer_code, longer_desc, data_chunk_id)

