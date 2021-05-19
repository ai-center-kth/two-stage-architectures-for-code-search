
import os
import os.path
import time
import pathlib
from .tfrecord_parser import TFRecordParser
from pathlib import Path
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras import backend as K
import numpy as np
from transformers import RobertaConfig
from transformers import TFRobertaModel

def generate_model(embedding_size, number_tokens, sentence_length, hinge_loss_margin):

    config = RobertaConfig()
    config.num_hidden_layers = 2
    config.attention_probs_dropout_prob = 0.3
    config.vocab_size = number_tokens

    roberta = TFRobertaModel(config)

    toks1 = tf.keras.layers.Input(shape=(sentence_length,), dtype='int64')
    atts1 = tf.keras.layers.Input(shape=(sentence_length,), dtype="int64")
    out1 = roberta(toks1)

    toks2 = tf.keras.layers.Input(shape=(sentence_length,), dtype='int64')
    atts2 = tf.keras.layers.Input(shape=(sentence_length,), dtype="int64")
    out2 = roberta(toks2)

    toks3 = tf.keras.layers.Input(shape=(sentence_length,), dtype='int64')
    atts3 = tf.keras.layers.Input(shape=(sentence_length,), dtype="int64")
    out3 = roberta(toks3)

    mean1 = tf.reduce_mean(out1[0], 1)
    mean2 = tf.reduce_mean(out2[0], 1)
    mean3 = tf.reduce_mean(out3[0], 1)

    code_embedding = tf.keras.Model(inputs=[toks1], outputs=[mean1],
                                  name='code_embedding')

    desc_embedding = tf.keras.Model(inputs=[toks2], outputs=[mean2],
                                  name='desc_embedding')

    #########Comment this block if objective is cosine similarity calculation
    good_similarity = tf.keras.layers.Dot(axes=1, normalize=True)([mean1, mean2])

    cos_good_sim = tf.keras.Model(inputs=[toks1, toks2], outputs=[good_similarity],
                               name='cos_model')


    bad_similarity = tf.keras.layers.Dot(axes=1, normalize=True)([mean1, mean3])

    cos_bad_sim = tf.keras.Model(inputs=[toks1, toks3], outputs=[bad_similarity],
                               name='cos_model')



    embedded_code = tf.keras.Input(shape=(mean1.shape[1],), name="embedded_code")
    embedded_desc = tf.keras.Input(shape=(mean2.shape[1],), name="embedded_desc")

    dot = tf.keras.layers.Dot(axes=1, normalize=True)([embedded_code, embedded_desc])
    dot_model = tf.keras.Model(inputs=[embedded_code, embedded_desc], outputs=[dot],
                                    name='dot_model')




    loss = tf.keras.layers.Lambda(lambda x: K.maximum(1e-6, hinge_loss_margin - x[0] + x[1]),
                                  output_shape=lambda x: x[0],
                                  name='loss')([good_similarity, bad_similarity])

    training_model = tf.keras.Model(inputs=[toks1, toks2, toks3], outputs=[loss],
                                    name='training_model')

    training_model.compile(loss=lambda y_true, y_pred: y_pred + y_true - y_true, optimizer='adam')
    # y_true-y_true avoids warning

    return training_model, code_embedding, desc_embedding, cos_good_sim, dot_model

def load_weights(model, path):
    if os.path.isfile(path+'/sbert_csc_weights.index'):
        model.load_weights(path+'/sbert_csc_weights')
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

    name = results_path+"/results-sbert-csc-dcs-" + time.strftime("%Y%m%d-%H%M%S") + ".csv"

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

    print("S-BERT with SearchCodeChallenge dataset")

    script_path = str(pathlib.Path(__file__).parent)+"/"

    target_path = script_path+"../data/codesearchnet/tfrecord/"

    tfr_files = sorted(Path(target_path + 'python/train/').glob('**/*.tfrecord'))

    tfr_files = [x.__str__() for x in tfr_files]

    print(str(len(tfr_files))+" files loaded for the dataset")

    BATCH_SIZE = 32
    dataset = TFRecordParser.generate_dataset(tfr_files, BATCH_SIZE)

    number_code_tokens = 30522
    number_desc_tokens = 30522

    longer_code = 90
    longer_desc = 90
    embedding_size = 16384

    print("Building model and loading weights")

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

    train(training_model, dataset, script_path + "/../weights/sbert_csc_weights", steps_per_epoch)

    test_files = sorted(Path(target_path + 'python/test/').glob('**/*.tfrecord'))
    test_files = [x.__str__() for x in test_files]
    test_dataset = TFRecordParser.generate_dataset(test_files, 1)

    test(test_dataset, model_code, model_query, dot_model, script_path+"/../results")
