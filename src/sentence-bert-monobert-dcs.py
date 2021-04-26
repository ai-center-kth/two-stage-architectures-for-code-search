
import subprocess
import sys

subprocess.check_call([sys.executable, "-m", "pip", "install", "tables"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "tqdm"])

subprocess.check_call([sys.executable, "-m", "pip", "install", "bert-tensorflow==1.0.1"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "tf-hub-nightly"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "transformers"])


import pathlib
import tensorflow as tf
import tensorflow_hub as hub
from bert.tokenization import FullTokenizer

from code_search_manager import CodeSearchManager
import numpy as np
from help import *
import random

import transformers

from mono_bert import MONOBERT_DCS
from sentece_bert_dcs import SBERT_DCS

class SBERT_MONOBERT_DCS(CodeSearchManager):

    def __init__(self, data_path, data_chunk_id=0):
        self.data_path = data_path
        self.tokenizer = None
        self.max_len = 90


def get_monobert_score_candidates(my_id, topN_ids):
    desc = (" ".join([vocab_desc[x] for x in test_desc[my_id]]))
    code = (" ".join([vocab_tokens[x] for x in test_tokens[my_id]]))

    # expected_best_result = dot_model.predict([embedded_tokens[rowid].reshape((1, -1)), embedded_desc[rowid].reshape((1, -1))])[0][0]
    input_ids, attention_mask, token_type_ids, = sentece_bert.tokenize_sentences(desc, code)

    prediction = bert_model.predict(x=[np.array(input_ids).reshape((1, -1)),
                                  np.array(attention_mask).reshape((1, -1)),
                                  np.array(token_type_ids).reshape((1, -1))])[0]

    ground_truth_similarity = prediction[0]

    count = 0
    for tokenidx in topN_ids:

        if my_id == tokenidx:
            continue

        candidate_code = (" ".join([vocab_tokens[x] for x in test_tokens[tokenidx]]))

        candidate_input_ids, candidate_attention_mask, candidate_token_type_ids, = sentece_bert.tokenize_sentences(desc,
                                                                                                      candidate_code)

        candidate_prediction = bert_model.predict(x=[np.array(candidate_input_ids).reshape((1, -1)),
                                                np.array(candidate_attention_mask).reshape((1, -1)),
                                                np.array(candidate_token_type_ids).reshape((1, -1))])[0]

        # this means negative relation

        if candidate_prediction[0] > ground_truth_similarity:
            count = count + 1

    return count

if __name__ == "__main__":

    args = sys.argv
    data_chunk_id = 0
    if len(args) > 1:
        data_chunk_id = int(args[1])

    script_path = str(pathlib.Path(__file__).parent)

    data_path = script_path + "/../data/deep-code-search/drive/"

    monobert = MONOBERT_DCS(data_path)
    sentece_bert = SBERT_DCS(data_path)


    mono_bert_layer = transformers.TFBertModel.from_pretrained("bert-base-uncased")

    sentence_bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1",
                                trainable=False)

    # Some verion incompatibility requires this line
    tf.gfile = tf.io.gfile

    ## Sentece bert tokenizer
    vocab_file = sentence_bert_layer.resolved_object.vocab_file.asset_path.numpy()
    do_lower_case = sentence_bert_layer.resolved_object.do_lower_case.numpy()
    tokenizer = FullTokenizer(vocab_file, do_lower_case)

    sentece_bert.tokenizer = tokenizer

    multi_gpu = False

    print("Loading monobert and weights")
    if multi_gpu:
        tf.debugging.set_log_device_placement(False)

        strategy = tf.distribute.MirroredStrategy()

        with strategy.scope():
            bert_model = monobert.generate_model(mono_bert_layer)
            monobert.load_weights(bert_model, script_path+"/../weights/monobert_weights")

            training_model, model_code, model_query, dot_model = sentece_bert.generate_model(sentence_bert_layer)
            sentece_bert.load_weights(training_model, script_path+"/../weights/sbert_dcs_weights")
    else:
        bert_model = monobert.generate_model(mono_bert_layer)
        monobert.load_weights(bert_model, script_path + "/../weights/monobert_weights")

        training_model, model_code, model_query, dot_model  = sentece_bert.generate_model(sentence_bert_layer)
        sentece_bert.load_weights(training_model, script_path + "/../weights/sbert_dcs_weights")


    test_tokens = load_hdf5(data_path + "test.tokens.h5", 0, 100)
    test_desc = load_hdf5(data_path + "test.desc.h5", 0, 100)  # 10000

    code_test_vector = test_tokens
    desc_test_vector = test_desc

    embedded_tokens = []
    embedded_desc = []

    vocabulary_tokens = load_pickle(data_path + "vocab.tokens.pkl")
    vocab_tokens = {y: x for x, y in vocabulary_tokens.items()}

    vocabulary_desc = load_pickle(data_path + "vocab.desc.pkl")
    vocab_desc = {y: x for x, y in vocabulary_desc.items()}

    print("Embedding tokens and desc for SBERT")
    for idx, token in enumerate(code_test_vector):
        desc = (" ".join([vocab_desc[x] for x in desc_test_vector[idx]]))
        code = (" ".join([vocab_tokens[x] for x in code_test_vector[idx]]))

        desc_ = sentece_bert.tokenize_sentences(desc, "")
        code_ = sentece_bert.tokenize_sentences(code, "")

        embedded_tokens.append(model_code.predict([np.array(code_[0]).reshape((1, -1)),
                                                   np.array(code_[1]).reshape((1, -1)),
                                                   np.array(code_[2]).reshape((1, -1))

                                                   ])[0])

        embedded_desc.append(model_query.predict([np.array(desc_[0]).reshape((1, -1)),
                                                  np.array(desc_[1]).reshape((1, -1)),
                                                  np.array(desc_[2]).reshape((1, -1))

                                                  ])[0])

    embedded_tokens = np.array(embedded_tokens)
    embedded_desc = np.array(embedded_desc)

    results = {}
    results_extended = {}

    pbar = tqdm(total=len(embedded_desc))

    for rowid, desc in enumerate(embedded_desc):

        # SBERT
        expected_best_result = \
        dot_model.predict([embedded_tokens[rowid].reshape((1, -1)), embedded_desc[rowid].reshape((1, -1))])[0][0]

        deleted_tokens = np.delete(embedded_tokens, rowid, 0)

        tiled_desc = np.tile(desc, (deleted_tokens.shape[0], 1))

        prediction = dot_model.predict([deleted_tokens, tiled_desc], batch_size=32 * 4)

        results[rowid] = len(prediction[prediction > expected_best_result])


        # SBERT+BERT
        tiled_desc = np.tile(desc, (embedded_tokens.shape[0], 1))

        prediction = dot_model.predict([embedded_tokens, tiled_desc], batch_size=32 * 4)

        tiled_desc = np.tile(desc, (len(embedded_tokens), 1))
        prediction = dot_model.predict([np.array(embedded_tokens), tiled_desc], batch_size=32 * 4)
        prediction = prediction.reshape((-1))

        N = 15
        # get ids sorted by prediction value
        predictions_ordered = prediction.argsort()[::-1]
        topN = predictions_ordered[-N:]

        if not rowid in topN:
            position = np.where(predictions_ordered == rowid)[0][0]
            results_extended[rowid] = position
        else:
            owo = get_monobert_score_candidates(rowid, topN)
            results_extended[rowid] = owo


        pbar.update(1)
    pbar.close()


    def get_top_n(n, results):
        count = 0
        for r in results:
            if results[r] < n:
                count += 1
        return count / len(results)


    print("Sentence Bert")
    top_1 = get_top_n(1, results)
    top_3 = get_top_n(3, results)
    top_5 = get_top_n(5, results)
    top_15 = get_top_n(15, results)

    print(top_1)
    print(top_3)
    print(top_5)
    print(top_15)

    print("Sentence+Mono Bert")
    top_1 = get_top_n(1, results_extended)
    top_3 = get_top_n(3, results_extended)
    top_5 = get_top_n(5, results_extended)
    top_15 = get_top_n(15, results_extended)

    print(top_1)
    print(top_3)
    print(top_5)
    print(top_15)