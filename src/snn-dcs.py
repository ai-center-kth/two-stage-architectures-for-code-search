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
import numpy as np
import os.path
import time
import pathlib
from dcs_data_generator import DataGeneratorDCS
import logging

logger = logging.getLogger("SNN-DCS")

def load_hdf5(vecfile, start_offset, chunk_size):
    """reads training sentences(list of int array) from a hdf5 file"""
    table = tables.open_file(vecfile)
    data = table.get_node('/phrases')[:].astype(np.int)
    index = table.get_node('/indices')[:]
    data_len = index.shape[0]
    if chunk_size == -1:  # if chunk_size is set to -1, then, load all data
        chunk_size = data_len
    start_offset = start_offset % data_len
    sents = []
    for offset in tqdm(range(start_offset, start_offset + chunk_size)):
        offset = offset % data_len
        len, pos = index[offset]['length'], index[offset]['pos']
        sents.append(data[pos:pos + len])
    table.close()
    return sents


def pad(data, len=None):
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    return pad_sequences(data, maxlen=len, padding='post', truncating='post', value=0)


def load_pickle(filename):
    return pickle.load(open(filename, 'rb'))


def main(argv):
    print(argv)


def get_dataset_meta():
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


def get_dataset_meta_hardcoded():
    return 410, 13645


def generate_model(embedding_size, number_tokens, sentence_length, hinge_loss_margin):
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

    cos_bad_sim = tf.keras.layers.Dot(axes=1, normalize=True, name='cos_bad_sim')([output_code, output_bad_desc])

    loss = tf.keras.layers.Lambda(lambda x: K.maximum(1e-6, hinge_loss_margin - x[0] + x[1]),
                                  output_shape=lambda x: x[0],
                                  name='loss')([cos_good_sim, cos_bad_sim])

    training_model = tf.keras.Model(inputs=[input_code, input_desc, input_bad_desc], outputs=[loss],
                                    name='training_model')

    training_model.compile(loss=lambda y_true, y_pred: y_pred + y_true - y_true, optimizer='adam')
    # y_true-y_true avoids warning

    return training_model, embedding_model


def load_weights(model, path):
    if os.path.isfile(path+'/snn_dcs_weights.index'):
        model.load_weights(path+'/snn_dcs_weights')
        logger.info("Weights loaded!")
    else:
        logger.warning("Warning! No weights loaded!")

# n >= 1
def get_top_n(n, results):
    count = 0
    for r in results:
        if results[r] < n:
            count+= 1
    return count / len(results)


def train(trainig_model, training_set_generator, valid_set_generator, weights_path, batch_size=32):
    trainig_model.fit(training_set_generator, epochs=1, validation_data=valid_set_generator, batch_size=batch_size)
    trainig_model.save_weights(weights_path)
    logger.info("Model saved!")


def test(data_path, code_embedding_model, desc_embedding_model, results_path, longer_sentence, data_batch):

    logger.info("Starting tests!")
    
    test_tokens = load_hdf5(data_path + "test.tokens.h5" , 0, 10000)
    test_desc = load_hdf5(data_path + "test.desc.h5" , 0, 10000)

    test_tokens = pad(test_tokens, longer_sentence)
    test_desc = pad(test_desc, longer_sentence)

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

    name = results_path+"/results-snn-dcs-" + time.strftime("%Y%m%d-%H%M%S") + ".csv"

    f = open(name, "a")

    f.write("batch,top1,top3,top5\n")
    f.write(str(data_batch)+","+str(top_1) + "," + str(top_3) + "," + str(top_5) + "\n")
    f.close()


def training_data_chunk(id, valid_perc, chunk_size):

    init_trainig = chunk_size * id
    init_valid = int(chunk_size * id + chunk_size * valid_perc)
    end_valid = int(chunk_size * id + chunk_size)

    return init_trainig, init_valid, end_valid


if __name__ == "__main__":
    script_path = str(pathlib.Path(__file__).parent)

    logger.info("Running SNN Model")

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

    data_path = script_path+"/../data/deep-code-search/processed/"

    longer_sentence, number_tokens = get_dataset_meta_hardcoded()
    embedding_size = 2048

    #tf.debugging.set_log_device_placement(True)

    #strategy = tf.distribute.MirroredStrategy()
    #with strategy.scope():

    training_model, embedding_model = generate_model(embedding_size, number_tokens, longer_sentence, 0.05)

    load_weights(training_model, script_path+"/../weights")

    init_trainig, init_valid, end_valid = training_data_chunk(data_chunk_id, 0.8, chunk_size)

    logger.info("Training model with chunk number " + str(data_chunk_id) + " of " + str(number_chunks))

    batch_size = 64 * 2
    training_set_generator = DataGeneratorDCS(data_path + "train.tokens.h5", data_path + "train.desc.h5", batch_size, init_trainig, init_valid, longer_sentence, longer_sentence)
    valid_set_generator = DataGeneratorDCS(data_path + "train.tokens.h5", data_path + "train.desc.h5", batch_size, init_valid, end_valid, longer_sentence, longer_sentence)

    #train(training_model, training_set_generator, valid_set_generator, script_path+"/../weights/snn_dcs_weights", batch_size)

    #test(data_path, embedding_model, embedding_model, script_path+"/../results", longer_sentence, data_chunk_id)

