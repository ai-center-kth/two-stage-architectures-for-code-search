import subprocess
import sys

subprocess.check_call([sys.executable, "-m", "pip", "install", "tables"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "tqdm"])
#subprocess.check_call([sys.executable, "-m", "pip", "install", "pickle5"])


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"

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

    code = tf.keras.Input(shape=(code_length,), dtype='int32', name='code')
    query = tf.keras.Input(shape=(desc_length,), dtype='int32', name='query')


    # add embedding layers
    code_embedding = tf.keras.layers.Embedding(number_code_tokens, embedding_size,
                            name='answer_embedding')(code)
    query_embedding = tf.keras.layers.Embedding(number_desc_tokens, embedding_size,
                            name='question_embedding')(query)


    # cnn
    filters = 400
    kernel_size = 2

    question_cnn = None
    answer_cnn = None

    k = kernel_size
    query_cnn = tf.keras.layers.Conv1D(kernel_size=k,
                   filters=filters,
                   activation='relu',
                   padding='same',
                   name=f'question_conv1d_{k}')(query_embedding)
    # question_cnn = merge([cnn(question_embedding) for cnn in cnns], mode='concat')

    # answer_cnn = merge([cnn(answer_embedding) for cnn in cnns], mode='concat')
    code_cnn = tf.keras.layers.Conv1D(kernel_size=k,
                   filters=filters,
                   activation='relu',
                   padding='same',
                   name=f'answer_conv1d_{k}')(code_embedding)

    # maxpooling
    maxpool = tf.keras.layers.Lambda(lambda x: K.max(x, axis=1, keepdims=False), output_shape=lambda x: (x[0], x[2]),
                     name='max')
    maxpool.supports_masking = True
    # enc = Dense(100, activation='tanh')
    # question_pool = enc(maxpool(question_cnn))
    # answer_pool = enc(maxpool(answer_cnn))
    query_pool = maxpool(query_cnn)
    code_pool = maxpool(code_cnn)

    # This model generates code embedding
    model_code = tf.keras.Model(inputs=[code], outputs=[code_pool], name='model_code')
    # This model generates description/query embedding
    model_query = tf.keras.Model(inputs=[query], outputs=[query_pool], name='model_query')

    cos_similarity = tf.keras.layers.Dot(axes=1, normalize=True, name='cos_good_sim')([code_pool, query_pool])


    cos_model = tf.keras.Model(inputs=[code, query], outputs=cos_similarity, name='cos_model')


    # Used in tests
    embedded_code = tf.keras.Input(shape=(query_pool.shape[1],), name="embedded_code")
    embedded_desc = tf.keras.Input(shape=(query_pool.shape[1],), name="embedded_desc")

    #dot = tf.keras.layers.Dot(axes=1, normalize=True)([embedded_code, embedded_desc])
    dot_model = None #tf.keras.Model(inputs=[embedded_code, embedded_desc], outputs=[dot], name='dot_model')



    # Negative sampling
    good_desc_input = tf.keras.Input(shape=(desc_length,), name="good_desc_input")
    bad_desc_input = tf.keras.Input(shape=(desc_length,), name="bad_desc_input")

    good_desc_output = cos_model([code, good_desc_input])
    bad_desc_output = cos_model([code, bad_desc_input])

    margin = 0.5
    loss = tf.keras.layers.Lambda(lambda x: K.maximum(1e-6, hinge_loss_margin - x[0] + x[1]), output_shape=lambda x: x[0],
                                  name='loss')([good_desc_output, bad_desc_output])

    training_model = tf.keras.Model(inputs=[code, good_desc_input, bad_desc_input], outputs=[loss],
                                    name='training_model')

    training_model.compile(loss=lambda y_true, y_pred: y_pred + y_true - y_true, optimizer='adam')

    return training_model, model_code, model_query, cos_model, dot_model


def load_weights(model, path):
    if os.path.isfile(path+'/concra_dcs_weights.index'):
        model.load_weights(path+'/concra_dcs_weights')
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


def test(data_path, model_code, model_query, cos_model, dot_model, results_path, code_length, desc_length, batch_id):
    test_tokens = load_hdf5(data_path + "test.tokens.h5" , 0, 50)
    test_desc = load_hdf5(data_path + "test.desc.h5" , 0, 50) #10000

    test_tokens = pad(test_tokens, code_length)
    test_desc = pad(test_desc, desc_length)



    print(test_tokens.shape)
    print(test_desc.shape)
    print(model_code.predict(test_tokens))
    exit(0)



    results = {}
    pbar = tqdm(total=len(test_desc))

    for rowid, desc in enumerate(test_desc):
        expected_best_result = cos_model.predict([test_tokens[rowid].reshape((1, -1)), test_desc[rowid].reshape((1, -1))])[0][0]

        deleted_tokens = np.delete(test_tokens, rowid, 0)

        tiled_desc = np.tile(desc, (deleted_tokens.shape[0], 1))

        ress = cos_model.predict([deleted_tokens, tiled_desc])

        results[rowid] = len(ress[ress > expected_best_result])

        pbar.update(1)
    pbar.close()

    top_1 = get_top_n(1, results)
    top_3 = get_top_n(3, results)
    top_5 = get_top_n(5, results)

    print(top_1)
    print(top_3)
    print(top_5)

    name = results_path+"/results-concra-dcs-" + time.strftime("%Y%m%d-%H%M%S") + ".csv"

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

    print("Concra Model")

    # dataset info
    total_length = 18223872
    chunk_size = 18223872 #18223872 #1000000

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
    #with strategy.scope():
    training_model, model_code, model_query, cos_model, dot_model = generate_model(embedding_size, number_code_tokens, number_desc_tokens, longer_code, longer_desc, 0.05)

    load_weights(training_model, script_path+"/../weights")

    init_trainig, init_valid, end_valid = training_data_chunk(data_chunk_id, 0.8, chunk_size)

    print("Training model with chunk number ", data_chunk_id, " of ", number_chunks)

    batch_size = 64 * 3
    training_set_generator = DataGeneratorDCS(data_path + "train.tokens.h5", data_path + "train.desc.h5", batch_size, init_trainig, init_valid, longer_code, longer_desc)
    valid_set_generator = DataGeneratorDCS(data_path + "train.tokens.h5", data_path + "train.desc.h5", batch_size, init_valid, end_valid, longer_code, longer_desc)

    train(training_model, training_set_generator, valid_set_generator, script_path+"/../weights/concra_dcs_weights", batch_size)

    #test(data_path, model_code, model_query, cos_model, dot_model, script_path+"/../results", longer_code, longer_desc, data_chunk_id)
    #test(data_path, cos_model, script_path+"/../results", longer_code, longer_desc, data_chunk_id)

