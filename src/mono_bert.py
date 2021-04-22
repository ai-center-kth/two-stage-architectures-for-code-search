
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

class MONOBERT_DCS(CodeSearchManager):

    def __init__(self, data_path, data_chunk_id=0):
        self.data_path = data_path
        self.tokenizer = None
        self.max_len = 90
        self.tokenizer = None
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


    def tokenize_sentences(self, input1_str, input2_str):

        tokenized = self.tokenizer.batch_encode_plus(
            [[input1_str, input2_str]],
            add_special_tokens=True,
            max_length=90,
            return_attention_mask=True,
            return_token_type_ids=True,
            pad_to_max_length=True,
            return_tensors="np",
        )

        return tokenized["input_ids"][0], tokenized["attention_mask"][0], tokenized["token_type_ids"][0]

    def load_dataset(self, train_desc, train_tokens, vocab_desc, vocab_tokens):
        retokenized_desc = []
        retokenized_mask = []
        retokenizedtype = []
        labels = []

        for idx, sentence in enumerate(train_desc):

            desc = (" ".join([vocab_desc[x] for x in train_desc[idx]]))
            code = (" ".join([vocab_tokens[x] for x in train_tokens[idx]]))

            random_code = train_tokens[random.randint(0, len(train_tokens) - 1)]
            neg_code = (" ".join([vocab_tokens[x] for x in random_code]))

            input_ids, attention_mask, token_type_ids, = self.tokenize_sentences(desc, code)
            if len(input_ids) != self.max_len:
                continue
            retokenized_desc.append(input_ids)
            retokenized_mask.append(attention_mask)
            retokenizedtype.append(token_type_ids)

            labels.append([1])

            input_ids_neg, attention_mask_neg, token_type_ids_neg, = self.tokenize_sentences(desc, neg_code)
            if len(input_ids_neg) != self.max_len:
                continue
            retokenized_desc.append(input_ids_neg)
            retokenized_mask.append(attention_mask_neg)
            retokenizedtype.append(token_type_ids_neg)

            labels.append([0])

        labels = np.array(labels)

        return retokenized_desc, retokenized_mask, retokenizedtype, labels


    def train(self, trainig_model, training_set, weights_path, epochs=1):
        trainig_model.fit(x=[np.array(training_set[0]),
                     np.array(training_set[1]),
                     np.array(training_set[2])
                     ],  # np.array(tokenized_code)
                  y=training_set[3], epochs=epochs, verbose=1, batch_size=32, validation_split=0.9)

        trainig_model.save_weights(weights_path)
        print("Model saved!")


    def test(self, model, results_path):

        file_format = "h5"

        test_tokens = load_hdf5(self.data_path + "test.tokens." + file_format, 0, 20)  # 1000000
        test_desc = load_hdf5(self.data_path + "test.desc." + file_format, 0, 20)

        results = {}
        pbar = tqdm(total=len(test_desc))

        for rowid, desc in enumerate(test_desc):

            desc = (" ".join([vocab_desc[x] for x in test_desc[rowid]]))
            code = (" ".join([vocab_tokens[x] for x in test_tokens[rowid]]))

            # expected_best_result = dot_model.predict([embedded_tokens[rowid].reshape((1, -1)), embedded_desc[rowid].reshape((1, -1))])[0][0]
            input_ids, attention_mask, token_type_ids, = self.tokenize_sentences(desc, code)
            prediction = model.predict(x=[np.array(input_ids).reshape((1, -1)),
                                          np.array(attention_mask).reshape((1, -1)),
                                          np.array(token_type_ids).reshape((1, -1))])[0]

            ground_truth_similarity = prediction[0]

            count = 0
            for tokenidx, tokens in enumerate(test_tokens):

                if rowid == tokenidx:
                    continue

                candidate_code = (" ".join([vocab_tokens[x] for x in test_tokens[tokenidx]]))

                candidate_input_ids, candidate_attention_mask, candidate_token_type_ids, = self.tokenize_sentences(desc,
                                                                                                              candidate_code)

                candidate_prediction = model.predict(x=[np.array(candidate_input_ids).reshape((1, -1)),
                                                        np.array(candidate_attention_mask).reshape((1, -1)),
                                                        np.array(candidate_token_type_ids).reshape((1, -1))])[0]

                # this means negative relation

                if candidate_prediction[0] > ground_truth_similarity:
                    count = count + 1

            results[rowid] = count
            pbar.update(1)

        def get_top_n(n, results):
            count = 0
            for r in results:
                if results[r] < n:
                    count += 1
            return count / len(results)

        top_1 = get_top_n(1, results)
        top_3 = get_top_n(3, results)
        top_5 = get_top_n(5, results)

        print(top_1)
        print(top_3)
        print(top_5)


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

    vocabulary_tokens = load_pickle(data_path + "vocab.tokens.pkl")
    vocab_tokens = {y: x for x, y in vocabulary_tokens.items()}

    vocabulary_desc = load_pickle(data_path + "vocab.desc.pkl")
    vocab_desc = {y: x for x, y in vocabulary_desc.items()}

    file_format = "h5"

    train_tokens = load_hdf5(data_path + "train.tokens." + file_format, 0, 5000)  # 100000
    train_desc = load_hdf5(data_path + "train.desc." + file_format, 0, 5000)

    dataset = monobert.load_dataset(train_desc, train_tokens, vocab_desc, vocab_tokens)

    multi_gpu = False

    print("Building model and loading weights")
    if multi_gpu:
        tf.debugging.set_log_device_placement(False)

        strategy = tf.distribute.MirroredStrategy()

        with strategy.scope():
            training_model = monobert.generate_model(bert_layer)
            #sbert_dcs.load_weights(training_model, script_path+"/../weights/sbert_dcs_weights")
    else:
        training_model = monobert.generate_model(bert_layer)
        #sbert_dcs.load_weights(training_model, script_path + "/../weights/sbert_dcs_weights")

    monobert.train(training_model, dataset, script_path+"/../weights/monobert_weights", 4)

    bert_layer.trainable = True

    monobert.train(training_model, dataset, script_path+"/../weights/monobert_weights", 2)

    monobert.test(training_model, script_path + "/../results")