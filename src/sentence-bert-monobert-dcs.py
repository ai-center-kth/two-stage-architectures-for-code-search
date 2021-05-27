
import sys
import pathlib
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import pandas as pd
import random
from tqdm import tqdm
import transformers

from . import help
from .code_search_manager import CodeSearchManager
from .mono_bert_dcs import MONOBERT_DCS
from .sentence_roberta_dcs import SBERT_DCS


class SBERT_MONOBERT_DCS(CodeSearchManager):

    def __init__(self, data_path, data_chunk_id=0):
        self.data_path = data_path
        self.tokenizer = None
        self.max_len = 90


def get_monobert_score_candidates(my_id, topN_ids, test_desc_str, test_code_str):
    desc = test_desc_str[my_id]

    # Generate a list with the target description with len "topN_ids"
    tiled_desc = np.tile(desc, (len(topN_ids),))

    # Get the topN codes
    topN_codes = test_code_str[topN_ids]

    # Tokenize the target description with the topN codes
    input_ids, attention_mask, token_type_ids, = monobert.tokenize_sentences(tiled_desc, topN_codes)

    # Get the similarities
    prediction = bert_model.predict(x=[input_ids,
                                       attention_mask,
                                       token_type_ids])

    # Sort the topN index with the result of the prediction
    sorted_pred, sorted_topN_ids = zip(*sorted(zip(prediction.reshape((-1,)), topN_ids), reverse=True))

    # Return the position of the target description in the sorted topN ids
    return list(sorted_topN_ids).index(my_id)

def get_top_n(n, results):
    count = 0
    for r in results:
        if results[r] < n:
            count += 1
    return count / len(results)

def topN_test(embedded_desc, embedded_tokens):

    results = {}
    results_extended = {}

    pbar = tqdm(total=len(embedded_desc))

    for rowid in range(len(embedded_tokens)):
        desc = embedded_desc[rowid]
        # array with same length of code, but only with this description
        tiled_desc = np.tile(desc, (len(embedded_tokens), 1))

        # for this description, get the similarity with all the code snippets
        prediction = dot_model.predict([embedded_tokens, tiled_desc], batch_size=32 * 4)

        prediction = prediction.reshape((-1))

        N = 15
        # get ids sorted by prediction value
        predictions_ordered = prediction.argsort()[::-1]
        topN = predictions_ordered[:N]

        # Standalone S-bert
        results[rowid] = np.where(predictions_ordered == rowid)[0][0]

        # Sbert + monobert
        if not rowid in topN:
            results_extended[rowid] = np.where(predictions_ordered == rowid)[0][0]
        else:
            results_extended[rowid] = get_monobert_score_candidates(rowid, topN, test_desc_str, test_code_str)
        pbar.update(1)
    pbar.close()

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


def get_code_candidates(rowid, embedded_tokens, embedded_desc):
    desc = embedded_desc[rowid]

    # array with same length of code, but only with this description
    tiled_desc = np.tile(desc, (len(embedded_tokens), 1))

    # for this description, get the similarity with all the code snippets
    prediction = dot_model.predict([embedded_tokens, tiled_desc], batch_size=32 * 4)

    prediction = prediction.reshape((-1))

    # get ids sorted by prediction value
    predictions_ordered = prediction.argsort()[::-1]

    return predictions_ordered

def rephrasing_test(rephrased_descriptions_df, embedded_tokens, embedded_desc):

    rephrased_ranking = {}
    new_ranking = {}
    for i, row in enumerate(rephrased_descriptions_df.iterrows()):
        idx = row[1].values[0]

        original_desc = row[1].values[1]

        embedded_tokens_copy = embedded_tokens.copy()
        embedded_desc_copy = embedded_desc.copy()

        # Sentence-BERT candidates
        candidates = get_code_candidates(idx, embedded_tokens_copy, embedded_desc_copy)

        N = 15
        topN = candidates[:N]
        if not idx in topN:
            original_rank = np.where(candidates == idx)[0][0]
        else:
            original_rank = get_monobert_score_candidates(idx, topN, test_desc_str, test_code_str)

        # Rephrashed description
        desc = row[1].values[2]

        desc_ = sentence_bert.tokenize(desc)

        embedded_desc_copy[idx] = (sentence_bert.desc_model.predict([np.array(desc_[0]).reshape((1, -1)),
                                                        np.array(desc_[1]).reshape((1, -1)),
                                                        np.array(desc_[2]).reshape((1, -1))

                                                        ])[0])
        test_desc_str[idx]=desc

        # Sentence-BERT candidates
        candidates = get_code_candidates(idx, embedded_tokens_copy, embedded_desc_copy)

        N = 15
        topN = candidates[:N]
        if not idx in topN:
            new_rank = np.where(candidates == idx)[0][0]
        else:
            new_rank = get_monobert_score_candidates(idx, topN, test_desc_str, test_code_str)

        rephrased_ranking[idx] = original_rank
        new_ranking[idx] = new_rank

    print("Number of queries: ",str(len(rephrased_descriptions_df.index)))
    print("Selected topN:")
    print(get_top_n(1, rephrased_ranking))
    print(get_top_n(3, rephrased_ranking))
    print(get_top_n(5, rephrased_ranking))

    print("Rephrased topN:")
    print(get_top_n(1, new_ranking))
    print(get_top_n(3, new_ranking))
    print(get_top_n(5, new_ranking))
    return rephrased_ranking, new_ranking





if __name__ == "__main__":

    args = sys.argv
    data_chunk_id = 0
    if len(args) > 1:
        data_chunk_id = int(args[1])

    script_path = str(pathlib.Path(__file__).parent)

    data_path = script_path + "/../data/deep-code-search/drive/"

    monobert = MONOBERT_DCS(data_path)
    sentence_bert = SBERT_DCS(data_path)

    monobert.generate_bert_layer()
    monobert.generate_model()
    monobert.generate_tokenizer()
    monobert.load_weights(script_path + "/../final_weights/monobert_dcs_weights")

    sentence_bert.generate_bert_layer()
    sentence_bert.generate_model()
    sentence_bert.generate_tokenizer()
    sentence_bert.load_weights(script_path + "/../final_weights/sroberta_dcs_weights")

    number_of_elements = 200
    test_tokens = help.load_hdf5(data_path + "test.tokens.h5", 0, number_of_elements)
    test_desc = help.load_hdf5(data_path + "test.desc.h5", 0, number_of_elements)  # 10000

    code_test_vector = test_tokens
    desc_test_vector = test_desc

    embedded_tokens = []
    embedded_desc = []

    vocabulary_tokens = help.load_pickle(data_path + "vocab.tokens.pkl")
    vocab_tokens = {y: x for x, y in vocabulary_tokens.items()}

    vocabulary_desc = help.load_pickle(data_path + "vocab.desc.pkl")
    vocab_desc = {y: x for x, y in vocabulary_desc.items()}

    test_desc_str = []
    test_code_str = []
    for i in range(len(test_desc)):
        test_desc_str.append(" ".join([vocab_desc[x] for x in test_desc[i]]))
        test_code_str.append(" ".join([vocab_tokens[x] for x in test_tokens[i]]))

    test_desc_str = np.array(test_desc_str)
    test_code_str = np.array(test_code_str)

    model_code = sentence_bert.code_model
    model_query = sentence_bert.desc_model

    dot_model = sentence_bert.dot_model

    bert_model = monobert.training_model

    desc_ = sentence_bert.tokenize(test_desc_str)
    code_ = sentence_bert.tokenize(test_code_str)

    ## Code and desc embedding
    embedded_tokens = model_code.predict([np.array(code_[0]),
                                          np.array(code_[1]),
                                           np.array(code_[2])

                                          ])

    embedded_desc = model_query.predict([np.array(desc_[0]),
                                         np.array(desc_[1]),
                                         np.array(desc_[2])
                                         ])

    topN_test(embedded_desc, embedded_tokens, test_desc_str, test_code_str)

    df = pd.read_csv(data_path + "descriptions.csv", header=0)
    df = df.dropna()
    df = df[df["rowid"] < number_of_elements]

    #rephrasing_test(df, embedded_tokens, embedded_desc)