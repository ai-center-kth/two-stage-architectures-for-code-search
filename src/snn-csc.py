
import subprocess
import sys

subprocess.check_call([sys.executable, "-m", "pip", "install", "tables"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "tqdm"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "transformers"])

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
import os.path
import time
import pathlib
from tfrecord_parser import TFRecordParser
from pathlib import Path
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras import backend as K
import numpy as np


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

    return training_model, embedding_model, embedding_model, cos_model, dot_model

def load_weights(model, path):
    if os.path.isfile(path+'/snn_csc_weights.index'):
        model.load_weights(path+'/snn_csc_weights')
        print("Weights loaded!")
    else:
        print("Warning!!  Weights not loaded")

def train(trainig_model, training_set_generator, weights_path, steps_per_epoch ):
    print("Training model...")
    trainig_model.fit(training_set_generator, epochs=20, steps_per_epoch=steps_per_epoch)
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

    name = results_path+"/results-snn-csc-dcs-" + time.strftime("%Y%m%d-%H%M%S") + ".csv"

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

def calc_num_elements(dataset):
    num_elements = 0
    for element in dataset:
        num_elements += 1
    return num_elements


if __name__ == "__main__":

    print("SNN with SearchCodeChallenge dataset")

    script_path = str(pathlib.Path(__file__).parent)+"/"

    target_path = script_path+"../data/codesearchnet/tfrecord/"

    tfr_files = sorted(Path(target_path + 'python/train/').glob('**/*.tfrecordtest'))

    tfr_files = [x.__str__() for x in tfr_files]

    BATCH_SIZE = 64 * 2
    dataset = TFRecordParser.generate_dataset(tfr_files, BATCH_SIZE)

    number_code_tokens = 30522
    number_desc_tokens = 30522

    longer_code = 45
    longer_desc = 45
    embedding_size = 2048 # 16384 #

    print("Building model and loading weights")
    strategy = tf.distribute.MirroredStrategy()

    multi_gpu = False

    if multi_gpu:
        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
            training_model, model_code, model_query, cos_model, dot_model = generate_model(embedding_size,
                                                                                           number_code_tokens,
                                                                                        longer_code,0.1)
            load_weights(training_model, script_path + "/../weights")
    else:
        training_model, model_code, model_query, cos_model, dot_model = generate_model(embedding_size,
                                                                                       number_code_tokens,
                                                                                       longer_code, 0.1)
        load_weights(training_model, script_path + "/../weights")


    num_elements = calc_num_elements(dataset)

    steps_per_epoch = num_elements // BATCH_SIZE

    train(training_model, dataset, script_path + "/../weights/snn_csc_weights", steps_per_epoch)

    test_files = sorted(Path(target_path + 'python/test/').glob('**/*.tfrecordtest'))
    test_files = [x.__str__() for x in test_files]
    test_dataset = TFRecordParser.generate_dataset(test_files, 1)

    test(test_dataset, model_code, model_query, dot_model, script_path+"/../results")
