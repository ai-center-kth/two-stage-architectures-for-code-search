
import subprocess
import sys

subprocess.check_call([sys.executable, "-m", "pip", "install", "tables"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "tqdm"])

subprocess.check_call([sys.executable, "-m", "pip", "install", "bert-tensorflow==1.0.1"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "tf-hub-nightly"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "transformers"])


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

import pathlib

import tensorflow as tf
import tensorflow_hub as hub

from code_search_manager import CodeSearchManager
import numpy as np
from help import *
import random
import transformers

class MONOBERT_DCS(CodeSearchManager):

    def __init__(self, data_path, data_chunk_id=0):
        self.data_path = data_path
        self.tokenizer = None
        self.max_len = 90
        print("Loading monoBERT model")

    def generate_model(self, bert_layer):
        # The model

        input_word_ids = tf.keras.layers.Input(shape=(self.max_len,),
                                               dtype=tf.int32,
                                               name="input_word_ids")
        input_mask = tf.keras.layers.Input(shape=(self.max_len,),
                                           dtype=tf.int32,
                                           name="input_mask")
        segment_ids = tf.keras.layers.Input(shape=(self.max_len,),
                                            dtype=tf.int32,
                                            name="segment_ids")

        bert_output = bert_layer([input_word_ids, input_mask, segment_ids])

        output = tf.keras.layers.Dense(1, activation="sigmoid")(bert_output[1])

        model = tf.keras.models.Model(
            inputs=[input_word_ids, input_mask, segment_ids], outputs=output
        )

        model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss="binary_crossentropy",
            metrics=["acc"],
        )

        return model

def get_top_n(n, results):
    count = 0
    for r in results:
        if results[r] < n:
            count += 1
    return count / len(results)


def test(data_path):

    file_format = "h5"

    test_tokens = load_hdf5(data_path + "test.tokens." + file_format, 0, 100)  # 1000000
    test_desc = load_hdf5(data_path + "test.desc." + file_format, 0, 100)

    # In[ ]:

    results = {}
    pbar = tqdm(total=len(test_desc))

    for rowid, desc in enumerate(test_desc):

        desc = (" ".join([vocab_desc[x] for x in test_desc[rowid]]))
        code = (" ".join([vocab_tokens[x] for x in test_tokens[rowid]]))

        # expected_best_result = dot_model.predict([embedded_tokens[rowid].reshape((1, -1)), embedded_desc[rowid].reshape((1, -1))])[0][0]
        input_ids, attention_mask, token_type_ids, = tokenize_sentences(desc, code)

        if len(input_ids) != 90:
            pbar.update(1)
            continue

        prediction = model.predict(x=[np.array(input_ids).reshape((1, -1)),
                                      np.array(attention_mask).reshape((1, -1)),
                                      np.array(token_type_ids).reshape((1, -1))])[0]

        best_result = prediction[0]

        count = 0
        for tokenidx, tokens in enumerate(test_tokens):

            if rowid == tokenidx:
                continue

            candidate_code = (" ".join([vocab_tokens[x] for x in test_tokens[tokenidx]]))

            candidate_input_ids, candidate_attention_mask, candidate_token_type_ids, = tokenize_sentences(desc,
                                                                                                          candidate_code)

            if len(candidate_input_ids) != 90:
                continue

            candidate_prediction = model.predict(x=[np.array(candidate_input_ids).reshape((1, -1)),
                                                    np.array(candidate_attention_mask).reshape((1, -1)),
                                                    np.array(candidate_token_type_ids).reshape((1, -1))])[0]

            # this means negative relation

            if candidate_prediction[0] > best_result:
                count = count + 1

        results[rowid] = count
        pbar.update(1)

    top_1 = get_top_n(1, results)
    top_3 = get_top_n(3, results)
    top_5 = get_top_n(5, results)
    top_15 = get_top_n(15, results)

    print(top_1)
    print(top_3)
    print(top_5)
    print(top_15)




def tokenize_sentences(input1_str, input2_str):
    # return concated_ids, masks, type_ids
    tokenized = tokenizer.batch_encode_plus(
        [[input1_str, input2_str]],
        add_special_tokens=True,
        max_length=90,
        return_attention_mask=True,
        return_token_type_ids=True,
        pad_to_max_length=True,
        return_tensors="np",
    )

    return tokenized["input_ids"][0], tokenized["attention_mask"][0], tokenized["token_type_ids"][0]

def generate_dataset():

    retokenized_desc = []
    retokenized_mask = []
    retokenizedtype = []
    labels = []

    for idx, sentence in enumerate(train_desc):

        desc = (" ".join([vocab_desc[x] for x in train_desc[idx]]))
        code = (" ".join([vocab_tokens[x] for x in train_tokens[idx]]))

        random_code = train_tokens[random.randint(0, len(train_tokens) - 1)]
        neg_code = (" ".join([vocab_tokens[x] for x in random_code]))

        input_ids, attention_mask, token_type_ids, = tokenize_sentences(desc, code)
        if len(input_ids) != MAX_LEN:
            continue
        retokenized_desc.append(input_ids)
        retokenized_mask.append(attention_mask)
        retokenizedtype.append(token_type_ids)

        labels.append([1])

        input_ids_neg, attention_mask_neg, token_type_ids_neg, = tokenize_sentences(desc, neg_code)
        if len(input_ids_neg) != MAX_LEN:
            continue
        retokenized_desc.append(input_ids_neg)
        retokenized_mask.append(attention_mask_neg)
        retokenizedtype.append(token_type_ids_neg)

        labels.append([0])
    return retokenized_desc, retokenized_mask, retokenizedtype, labels

if __name__ == "__main__":

    args = sys.argv
    data_chunk_id = 0
    if len(args) > 1:
        data_chunk_id = int(args[1])

    script_path = str(pathlib.Path(__file__).parent)

    data_path = script_path + "/../data/deep-code-search/drive/"

    monobert = MONOBERT_DCS(data_path)

    bert_layer = transformers.TFBertModel.from_pretrained("bert-base-uncased")

    monobert.tokenizer = transformers.BertTokenizer.from_pretrained(
        "bert-base-uncased", do_lower_case=True
    )


    file_format = "h5"

    # 18223872 (len) #1000000
    train_tokens = load_hdf5( data_path+"train.tokens."+file_format, 0, 1000) # 1000000
    train_desc = load_hdf5( data_path+"train.desc."+file_format, 0, 1000)
    # Negative sampling
    train_bad_desc = load_hdf5( data_path+"train.desc."+file_format, 0, 1000)
    random.shuffle(train_bad_desc)

    vocabulary_tokens = load_pickle(data_path+"vocab.tokens.pkl")
    vocab_tokens = {y: x for x, y in vocabulary_tokens.items()}

    vocabulary_desc = load_pickle(data_path+"vocab.desc.pkl")
    vocab_desc = {y: x for x, y in vocabulary_desc.items()}

    code_vector = train_tokens
    desc_vector = train_desc
    bad_desc_vector = train_bad_desc

    longer_code = max(len(t) for t in code_vector)
    longer_desc = max(len(t) for t in desc_vector)

    longer_desc = 90

    longer_sentence = max(longer_code, longer_desc)

    code_vector = pad(code_vector, longer_code)
    desc_vector = pad(desc_vector, longer_desc)
    bad_desc_vector = pad(bad_desc_vector, longer_desc)

    number_desc_tokens = len(vocabulary_desc)
    number_code_tokens = len(vocabulary_tokens)

    MAX_LEN = 90


    bert_layer = transformers.TFBertModel.from_pretrained("bert-base-uncased")
    # Freeze the BERT model to reuse the pretrained features without modifying them.
    bert_layer.trainable = False
    # bert_layer = bert_model(input_ids, attention_mask=attention_masks, token_type_ids=token_type_ids)
    model = generate_model(bert_layer)


    tokenizer = transformers.BertTokenizer.from_pretrained(
                "bert-base-uncased", do_lower_case=True
            )

    retokenized_desc, retokenized_mask, retokenizedtype, labels = generate_dataset()

    labels = np.array(labels)

    model.fit(x=[np.array(retokenized_desc),
                 np.array(retokenized_mask),
                 np.array(retokenizedtype)],
              y=labels, epochs=4, verbose=1, batch_size=15)

    bert_layer.trainable = True

    model.fit(x=[np.array(retokenized_desc),
                 np.array(retokenized_mask),
                 np.array(retokenizedtype)],
              y=labels, epochs=2, verbose=1, batch_size=15)

    model.save_weights(script_path+"/../weights/monobert_dcs_weights")

    test(data_path)