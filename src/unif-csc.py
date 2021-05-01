
import subprocess
import sys

subprocess.check_call([sys.executable, "-m", "pip", "install", "tables"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "tqdm"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "transformers"])

import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
import os.path
import time
import pathlib
from tfrecord_parser import TFRecordParser
from pathlib import Path
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras import backend as K
import numpy as np


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


    embedded_code = tf.keras.Input(shape=(code_output.shape[1],), name="embedded_code")
    embedded_desc = tf.keras.Input(shape=(query_output.shape[1],), name="embedded_desc")

    dot = tf.keras.layers.Dot(axes=1, normalize=True)([embedded_code, embedded_desc])
    dot_model = tf.keras.Model(inputs=[embedded_code, embedded_desc], outputs=[dot],
                                    name='dot_model')

    # Cosine similarity
    # If normalize set to True, then the output of the dot product is the cosine proximity between the two samples.
    cos_sim = dot_model([code_output, query_output])

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

    loss = tf.keras.layers.Lambda(lambda x: K.maximum(1e-6, hinge_loss_margin - x[0] + x[1]), output_shape=lambda x: x[0],
                                  name='loss')([good_desc_output, bad_desc_output])

    training_model = tf.keras.Model(inputs=[code_input, good_desc_input, bad_desc_input], outputs=[loss],
                                    name='training_model')

    training_model.compile(loss=lambda y_true, y_pred: y_pred + y_true - y_true, optimizer='adam')
    # y_true-y_true avoids warning

    return training_model, model_code, model_query, cos_model, dot_model

def load_weights(model, path):
    if os.path.isfile(path+'/unif_csc_weights.index'):
        model.load_weights(path+'/unif_csc_weights')
        print("Weights loaded!")
    else:
        print("Warning!!  Weights not loaded")

def train(trainig_model, training_set_generator, weights_path, steps_per_epoch ):
    print("Training model...")
    trainig_model.fit(training_set_generator, epochs=1, steps_per_epoch=steps_per_epoch)
    trainig_model.save_weights(weights_path)
    print("Model saved!")


def test(dataset, code_model, desc_model, dot_model, results_path):

    print("Testing model...")
    print(code_model)
    # Hardcoded
    dataset_size = 100 # 22176

    code_test_ds_np = iter(dataset.batch(dataset_size)).get_next()[0][0].numpy()
    code_test_ds_np = code_test_ds_np.reshape((dataset_size,-1))

    desc_test_ds_np = iter(dataset.batch(dataset_size)).get_next()[0][1].numpy()
    desc_test_ds_np = desc_test_ds_np.reshape((dataset_size,-1))

    code_embeddings = []
    desc_embeddings = []

    print("Embedding code and descriptions...")
    pbar = tqdm(total=dataset_size)
    for i in range(0,dataset_size):

        code_embeddings.append(code_model.predict(code_test_ds_np[i].reshape(1, -1))[0])
        desc_embeddings.append(desc_model.predict(desc_test_ds_np[i].reshape(1, -1))[0])

        pbar.update(1)
    pbar.close()

    print("Testing...")
    results = {}
    pbar = tqdm(total=len(desc_embeddings))
    for rowid, desc in enumerate(desc_embeddings):

        expected_best_result = dot_model.predict([code_embeddings[rowid].reshape((1, -1)), desc_embeddings[rowid].reshape((1, -1))])[0][0]

        deleted_tokens = np.delete(desc_embeddings, rowid, 0)

        tiled_desc = np.tile(desc, (deleted_tokens.shape[0], 1))

        prediction = dot_model.predict([deleted_tokens, tiled_desc]) # , batch_size=32*4

        results[rowid] = len(prediction[prediction > expected_best_result])

        pbar.update(1)
    pbar.close()

    top_1 = get_top_n(1, results)
    top_3 = get_top_n(3, results)
    top_5 = get_top_n(5, results)
    print(results)

    print(top_1)
    print(top_3)
    print(top_5)

    name = results_path+"/results-unif-csc-dcs-" + time.strftime("%Y%m%d-%H%M%S") + ".csv"

    f = open(name, "a")

    f.write("top1,top3,top5\n")
    f.write(str(top_1) + "," + str(top_3) + "," + str(top_5) + "\n")
    f.close()

def get_top_n(n, results):
    count = 0
    for r in results:
        if results[r] < n:
            count += 1
    return count / len(results)

if __name__ == "__main__":

    print("UNIF with SearchCodeChallenge dataset")

    script_path = str(pathlib.Path(__file__).parent)+"/"

    target_path = script_path+"../data/codesearchnet/tfrecord/"

    tfr_files = sorted(Path(target_path + 'python/train/').glob('**/*.tfrecordtest'))

    tfr_files = [x.__str__() for x in tfr_files]

    BATCH_SIZE = 1 #64
    dataset = TFRecordParser.generate_dataset(tfr_files, BATCH_SIZE)

    number_code_tokens = 30522
    number_desc_tokens = 30522

    longer_code = 45
    longer_desc = 45
    embedding_size = 2048 # 16384 #

    print("Building model and loading weights")
    strategy = tf.distribute.MirroredStrategy()

    multi_gpu = False


    training_model, model_code, model_query, cos_model, dot_model = generate_model(embedding_size,
                                                                                   number_code_tokens,
                                                                                   number_desc_tokens, longer_code,
                                                                                   longer_desc, 0.05)
    load_weights(training_model, script_path + "/../weights")

    num_elements = 420000
    steps_per_epoch = num_elements // BATCH_SIZE

    #train(training_model, dataset, script_path + "/../weights/unif_csc_weights", steps_per_epoch)

    test_files = sorted(Path(target_path + 'python/test/').glob('**/*.tfrecordtest'))
    test_files = [x.__str__() for x in test_files]
    test_dataset = TFRecordParser.generate_dataset(tfr_files, 1)

    test(dataset, model_code, model_query, dot_model, script_path+"/../results")
