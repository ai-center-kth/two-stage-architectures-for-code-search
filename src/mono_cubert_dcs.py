
import os
import sys
import pathlib
import time
import random

from tqdm import tqdm
import numpy as np
import pandas as pd
import tensorflow as tf
import transformers

from transformers.models.bert import convert_bert_original_tf_checkpoint_to_pytorch
from transformers import BertConfig, TFBertModel

from .cubert.cubert_hug_tokenizer import CuBertHugTokenizer
from . import CodeSearchManager, help
from .data_generators.monobert_dcs_data_generator import DataGeneratorDCSMonoBERT


class MONO_CUBERT_DCS(CodeSearchManager):

    def __init__(self, data_path, data_chunk_id=0):
        self.data_path = data_path
        self.tokenizer = None
        self.max_len = 90
        self.bert_layer = None
        self.vocab_desc = None
        self.vocab_code = None
        self.inverse_vocab_tokens = None
        self.inverse_vocab_desc = None
        self.training_model = None
        print("Loading monoBERT model")

    def get_vocabularies(self):
        self.inverse_vocab_tokens = help.load_pickle(self.data_path + "vocab.tokens.pkl")
        self.vocab_code = {y: x for x, y in self.inverse_vocab_tokens.items()}

        self.inverse_vocab_desc = help.load_pickle(self.data_path + "vocab.desc.pkl")
        self.vocab_desc = {y: x for x, y in self.inverse_vocab_desc.items()}

        return self.vocab_code, self.vocab_desc

    def get_bert_layer(self, path):
        self.bert_layer = self.get_cubert_layer(path)


    def get_cubert_layer(self, path):
        MODEL_PATH = path+'/../cuBERTconfig/20200621_Python_function_docstring__epochs_20__pre_trained_epochs_1_model.ckpt-6072.index'
        MODEL_CONFIG = path+'/../cuBERTconfig/cubert_config.json'
        MODEL_PATH_TORCH = path+'/../cuBERTconfig/20200621_Python_function_docstring__epochs_20__pre_trained_epochs_1_model.ckpt-6072.bin'
        if not os.path.isfile(MODEL_PATH_TORCH):
            convert_bert_original_tf_checkpoint_to_pytorch.convert_tf_checkpoint_to_pytorch(MODEL_PATH, MODEL_CONFIG, MODEL_PATH_TORCH)
        model_config = BertConfig.from_json_file(MODEL_CONFIG)
        return TFBertModel.from_pretrained(pretrained_model_name_or_path=MODEL_PATH_TORCH, from_pt=True,
                                                   config=model_config)
    def get_cubert_tokenizer(self, path):
        MODEL_VOCAB = path + '/../cuBERTconfig/vocab.txt'
        return CuBertHugTokenizer(MODEL_VOCAB)

    def generate_tokenizer(self, path):
        self.tokenizer = self.get_cubert_tokenizer(path)


    def generate_model(self):
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

        bert_output = self.bert_layer([input_word_ids, input_mask, segment_ids])

        output = tf.keras.layers.Dense(1, activation="sigmoid")(bert_output[1])

        model = tf.keras.models.Model(
            inputs=[input_word_ids, input_mask, segment_ids], outputs=output
        )

        opt = tf.keras.optimizers.Adam(learning_rate=0.000001)

        model.compile(
            optimizer=opt,
            loss="binary_crossentropy",
            metrics=["acc"],
        )

        self.training_model = model
        return model


    def get_id_rank(self, rowid, desc_string, test_tokens_str):
        desc = desc_string

        # Create array same length as the rest of code embeddings only containing this description
        tiled_desc = [desc] * len(test_tokens_str)

        input_ids, attention_mask, token_type_ids = self.tokenize_sentences(tiled_desc, test_tokens_str)

        candidate_prediction = self.training_model.predict([input_ids,
                                                attention_mask,
                                                token_type_ids])

        candidate_prediction = candidate_prediction.reshape((-1))

        # get ids sorted by prediction value
        candidate_prediction = candidate_prediction.argsort()[::-1]

        return np.where(candidate_prediction == rowid)[0][0]



    def test_embedded(self, results_path, number_of_elements=100 ):

        test_tokens = help.load_hdf5(self.data_path + "test.tokens.h5", 0, number_of_elements)
        test_desc = help.load_hdf5(self.data_path + "test.desc.h5" , 0, number_of_elements)

        test_code_str = []
        for token in test_tokens:
            code = (" ".join([self.vocab_code[x] for x in token]))
            test_code_str.append(code)


        results = {}
        pbar = tqdm(total=len(test_desc))

        for rowid, desc in enumerate(test_desc):

            desc = (" ".join([self.vocab_desc[x] for x in test_desc[rowid]]))
            results[rowid] = self.get_id_rank(rowid, desc, test_code_str)
            pbar.update(1)

        pbar.close()

        top_1 = self.get_top_n(1, results)
        top_3 = self.get_top_n(3, results)
        top_5 = self.get_top_n(5, results)
        top_15 = self.get_top_n(15, results)

        print(top_1)
        print(top_3)
        print(top_5)
        print(top_15)
        name = results_path + time.strftime("%Y%m%d-%H%M%S") + ".csv"

        f = open(name, "a")

        f.write("top1,top3,top5\n")
        f.write( str(top_1) + "," + str(top_3) + "," + str(top_5) + "\n")
        f.close()

        help.save_pickle(results_path + time.strftime("%Y%m%d-%H%M%S")+ "-rankings" + ".pkl", results)

    def test(self, results_path, number_of_elements=100 ):
        self.test_embedded(results_path, number_of_elements )

        df = pd.read_csv(self.data_path + "descriptions.csv", header=0)
        df = df.dropna()
        df = df[df["rowid"] < number_of_elements]

        self.rephrasing_test(df, number_of_elements)

    def generate_similarity_examples(self, results_path, number_of_elements=100):

        f = open(results_path + time.strftime("%Y%m%d-%H%M%S")+ "-similarities" + ".csv", "a")
        f.write("match,similarity\n")

        test_desc = help.load_hdf5(self.data_path + "test.desc.h5" , 0, number_of_elements)
        test_tokens = help.load_hdf5(self.data_path + "test.tokens.h5", 0, number_of_elements)

        test_code_str = []
        test_desc_str = []

        for token in test_tokens:
            test_code_str.append(" ".join([self.vocab_code[x] for x in token]))

        for desc in test_desc:
            test_desc_str.append(" ".join([self.vocab_code[x] for x in desc]))


        for idx,desc in enumerate(test_desc_str):

            input_ids, attention_mask, token_type_ids = self.tokenize_sentences(test_desc_str[idx], test_code_str[idx])

            prediction = self.training_model.predict([input_ids,
                                                                attention_mask,
                                                                token_type_ids])[0][0]

            f.write(str(1) + "," + str(prediction) + "\n")

            random_code = random.randint(0, len(test_code_str) - 1)
            random_desc = random.randint(0, len(test_desc_str) - 1)

            input_ids, attention_mask, token_type_ids = self.tokenize_sentences(test_desc_str[random_desc], test_code_str[random_code])

            prediction = self.training_model.predict([input_ids,
                                                                attention_mask,
                                                                token_type_ids])[0][0]

            f.write(str(0) + "," + str(prediction) + "\n")

        f.close()

    def rephrasing_test(self, rephrased_descriptions_df,  number_of_elements=100 ):

        test_tokens = help.load_hdf5(self.data_path + "test.tokens.h5", 0, number_of_elements)
        test_code_str = []
        for token in test_tokens:
            code = (" ".join([self.vocab_code[x] for x in token]))
            test_code_str.append(code)

        rephrased_ranking = {}
        new_ranking = {}
        pbar = tqdm(total=number_of_elements)
        for i, row in enumerate(rephrased_descriptions_df.iterrows()):
            idx = row[1].values[0]
            original_desc = row[1].values[1]
            new_desc = row[1].values[2]

            rephrased_ranking[idx] = self.get_id_rank(idx, original_desc, test_code_str)
            new_ranking[idx] = self.get_id_rank(idx, new_desc, test_code_str)
            pbar.update(1)
        pbar.close()


        print("Number of queries: ",str(len(rephrased_descriptions_df.index)))
        print("Selected topN:")
        print(self.get_top_n(1, rephrased_ranking))
        print(self.get_top_n(3, rephrased_ranking))
        print(self.get_top_n(5, rephrased_ranking))

        print("Rephrased topN:")
        print(self.get_top_n(1, new_ranking))
        print(self.get_top_n(3, new_ranking))
        print(self.get_top_n(5, new_ranking))
        return rephrased_ranking, new_ranking

    def tokenize_sentences(self, input_str1, input_str2):
        #return DataGeneratorDCSMonoBERT.tokenize_sentences(self.tokenizer, 90, input_str1, input_str2)
        if isinstance(input_str1, str):
            input_str1 = [input_str1]
            input_str2 = [input_str2]
        tokenizer_input = list(zip(input_str1, input_str2))

        tokenized = self.tokenizer.batch_encode_plus(
            tokenizer_input,
            add_special_tokens=True,
            max_length=self.max_len,
            return_attention_mask=True,
            return_token_type_ids=True,
            return_tensors="np",
            padding='max_length',
            truncation=True
        )

        return tokenized["input_ids"], tokenized["attention_mask"], tokenized["token_type_ids"]



if __name__ == "__main__":

    args = sys.argv
    data_chunk_id = 0
    if len(args) > 1:
        data_chunk_id = int(args[1])

    script_path = str(pathlib.Path(__file__).parent)

    data_path = script_path + "/../data/deep-code-search/drive/"

    monobert = MONO_CUBERT_DCS(data_path)

    vocabulary_tokens, vocabulary_desc = monobert.get_vocabularies()

    longer_desc = 90

    number_desc_tokens = len(vocabulary_desc)
    number_code_tokens = len(vocabulary_tokens)

    MAX_LEN = 90

    bert_layer = monobert.get_bert_layer(script_path)

    monobert.bert_layer.trainable = False

    model = monobert.generate_model()

    monobert.generate_tokenizer(script_path)

    #retokenized_desc, retokenized_mask, retokenizedtype, labels = generate_dataset()

    dataset = DataGeneratorDCSMonoBERT(data_path+"train.tokens.h5", data_path+"train.desc.h5",
                                       32, 0, 600000, 90, monobert.tokenizer, monobert.vocab_code, monobert.vocab_desc)

    #monobert.load_weights(script_path + "/../final_weights/monobert_dcs_weights")

    monobert.train(dataset, script_path+"/../weights/mono_cubert_dcs_weights", epochs=10, batch_size=None)

    #model.save_weights(script_path+"/../weights/monobert_dcs_weights")

    monobert.test("results/mono_cubert", 100)