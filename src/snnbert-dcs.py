import subprocess
import sys

subprocess.check_call([sys.executable, "-m", "pip", "install", "tables"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "tqdm"])
#subprocess.check_call([sys.executable, "-m", "pip", "install", "pickle5"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "transformers"])


#import pickle5 as pickle
import pickle
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"

import tensorflow as tf
from tensorflow.keras import backend as K
import sys
import tables
from tqdm import tqdm
import numpy as np
import os.path
import time
import pathlib
from dcs_bert_data_generator import DataGeneratorDCSBERT
from help import *
from transformers import BertTokenizer, TFBertModel, BertConfig
import transformers

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
    longer_sentence = 64
    number_tokens = None
    return longer_sentence, number_tokens


def generate_model(embedding_size, number_tokens, sentence_length, hinge_loss_margin):

    encoder = TFBertModel.from_pretrained("bert-base-uncased")
    encoder.trainable = False

    input_ids = tf.keras.Input(shape=(sentence_length,), dtype=tf.int32)
    #token_type_ids = tf.keras.Input(shape=(sentence_length,), dtype=tf.int32)
    #attention_mask = tf.keras.Input(shape=(sentence_length,), dtype=tf.int32)

    embedding_layer = encoder(
        input_ids)[0]

    attention_layer = tf.keras.layers.Attention(name="attention")([embedding_layer, embedding_layer])

    sum_layer = tf.keras.layers.Lambda(lambda x: K.sum(x, axis=1), name="sum")(embedding_layer)
    # average_layer = tf.keras.layers.Lambda(lambda x: K.mean(x, axis=1), name="average")( attention_layer)

    embedding_model = tf.keras.Model(inputs=[input_ids], outputs=[sum_layer], name='siamese_model')

    ids_code = tf.keras.Input(shape=(sentence_length,), name="ids_code")
    #type_code = tf.keras.Input(shape=(sentence_length,), name="type_code")
    #mask_code = tf.keras.Input(shape=(sentence_length,), name="mask_code")

    ids_desc = tf.keras.Input(shape=(sentence_length,), name="ids_desc")
    #type_desc = tf.keras.Input(shape=(sentence_length,), name="type_desc")
    #mask_desc = tf.keras.Input(shape=(sentence_length,), name="mask_desc")

    ids_bad = tf.keras.Input(shape=(sentence_length,), name="ids_bad")
    #type_bad = tf.keras.Input(shape=(sentence_length,), name="type_bad")
    #mask_bad = tf.keras.Input(shape=(sentence_length,), name="mask_bad")

    output_code = embedding_model([ids_code])
    output_desc = embedding_model([ids_desc])
    output_bad_desc = embedding_model([ids_bad])

    cos_good_sim = tf.keras.layers.Dot(axes=1, normalize=True, name='cos_good_sim')([output_code, output_desc])

    cos_model = tf.keras.Model(inputs=[ids_code, ids_desc], outputs=[cos_good_sim],
                                    name='cos_model')

    cos_bad_sim = tf.keras.layers.Dot(axes=1, normalize=True, name='cos_bad_sim')([output_code, output_bad_desc])

    loss = tf.keras.layers.Lambda(lambda x: K.maximum(1e-6, hinge_loss_margin - x[0] + x[1]),
                                  output_shape=lambda x: x[0],
                                  name='loss')([cos_good_sim, cos_bad_sim])

    training_model = tf.keras.Model(inputs=[ids_code, ids_desc, ids_bad], outputs=[loss],
                                    name='training_model')

    training_model.compile(loss=lambda y_true, y_pred: y_pred + y_true - y_true, optimizer='adam')
    # y_true-y_true avoids warning

    return training_model, embedding_model, cos_model


def load_weights(model, path):
    if os.path.isfile(path+'/snnbert_dcs_weights.index'):
        model.load_weights(path+'/snnbert_dcs_weights')
        print("Weights loaded!")
    else:
        print("Warning! No weights loaded!")

# n >= 1
def get_top_n(n, results):
    count = 0
    for r in results:
        if results[r] < n:
            count+= 1
    return count / len(results)


def train(trainig_model, training_set_generator, weights_path, batch_size=32):
    trainig_model.fit(training_set_generator, epochs=1, batch_size=batch_size)
    trainig_model.save_weights(weights_path)
    print("Model saved!")




def test(data_path, cos_model, results_path, code_length, desc_length, batch_id, vocab, tokenizer):
    test_tokens = load_hdf5(data_path + "test.tokens.h5" , 0, 10) #10000
    test_desc = load_hdf5(data_path + "test.desc.h5" , 0, 10)

    for idx,token in enumerate(test_tokens):
        encoded_code = tokenizer.batch_encode_plus(
            [" ".join([vocab[x] for x in token])],
            add_special_tokens=True,
            max_length=code_length,
            # return_attention_mask=True,
            # return_token_type_ids=True,
            pad_to_max_length=code_length,
            return_tensors="tf",
        )["input_ids"]

        test_tokens[idx] = encoded_code.numpy()[0]

    for idx,desc in enumerate(test_desc):
        encoded_desc = tokenizer.batch_encode_plus(
            [" ".join([vocab[x] for x in desc])],
            add_special_tokens=True,
            max_length=code_length,
            # return_attention_mask=True,
            # return_token_type_ids=True,
            pad_to_max_length=code_length,
            return_tensors="tf",
        )["input_ids"]

        test_desc[idx] = encoded_desc.numpy()[0]

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

    name = results_path+"/results-snnbert-dcs-" + time.strftime("%Y%m%d-%H%M%S") + ".csv"

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

    print("Running SNN Bert Model")

    # dataset info
    total_length = 18223872
    chunk_size = 9111936 #1000000

    number_chunks = total_length/chunk_size - 1
    number_chunks = int(number_chunks + 1 if number_chunks > int(number_chunks) else number_chunks)

    args = sys.argv
    data_chunk_id = 0
    if len(args) > 1:
        data_chunk_id = int(args[1])

    data_chunk_id = min(data_chunk_id, int(number_chunks))

    data_path = script_path+"/../data/deep-code-search/processed/"

    longer_sentence, number_tokens = get_dataset_meta_hardcoded()
    embedding_size = None

    #tf.debugging.set_log_device_placement(True)

    #strategy = tf.distribute.MirroredStrategy()
    #with strategy.scope():

    print("Building model and loading weights")
    training_model, embedding_model, cos_model = generate_model(embedding_size, number_tokens, longer_sentence, 0.05)
        #load_weights(training_model, script_path+"/../weights")

    init_trainig, init_valid, end_valid = training_data_chunk(data_chunk_id, 1.0, chunk_size)

    print("Training model with chunk number " + str(data_chunk_id) + " of " + str(number_chunks))

    batch_size = 32 * 2

    merged = load_pickle(data_path+"vocab.merged.pkl")
    vocab = {y: x for x, y in merged.items()}

    tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-uncased") #RobertaTokenizer.from_pretrained("microsoft/codebert-base")

    training_set_generator = DataGeneratorDCSBERT(data_path + "train.tokens.h5", data_path + "train.desc.h5", batch_size, init_trainig, init_valid, longer_sentence, longer_sentence, tokenizer, vocab)

    #valid_set_generator = DataGeneratorDCS(data_path + "train.tokens.h5", data_path + "train.desc.h5", batch_size, init_valid, end_valid, longer_sentence, longer_sentence)

    #train(training_model, training_set_generator, script_path+"/../weights/snnbert_dcs_weights", batch_size)

    test(data_path, cos_model, script_path+"/../results", longer_sentence, longer_sentence, data_chunk_id, vocab, tokenizer)

