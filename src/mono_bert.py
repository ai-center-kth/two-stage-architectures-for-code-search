
import sys
import pathlib
import tensorflow as tf
from tqdm import tqdm
import numpy as np
import transformers

from . import CodeSearchManager, help
from .data_generators.monobert_dcs_data_generator import DataGeneratorDCSMonoBERT


class MONOBERT_DCS(CodeSearchManager):

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

    def generate_tokenizer(self):
        self.tokenizer = transformers.RobertaTokenizer.from_pretrained('roberta-base', do_lower_case=True)
        return self.tokenizer

    def generate_bert_layer(self):
        self.bert_layer = transformers.TFRobertaModel.from_pretrained('roberta-base')
        return self.bert_layer

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

        output = tf.keras.layers.Dense(1, activation="sigmoid")(bert_output[0])

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

    def test(self):

        file_format = "h5"

        test_tokens = help.load_hdf5(self.data_path + "test.tokens." + file_format, 0, 100)  # 1000000
        test_desc = help.load_hdf5(self.data_path + "test.desc." + file_format, 0, 100)

        # In[ ]:

        results = {}
        pbar = tqdm(total=len(test_desc))

        for rowid, desc in enumerate(test_desc):

            desc = (" ".join([self.vocab_desc[x] for x in test_desc[rowid]]))
            code = (" ".join([self.vocab_code[x] for x in test_tokens[rowid]]))

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

                candidate_code = (" ".join([self.vocab_code[x] for x in test_tokens[tokenidx]]))

                candidate_input_ids, candidate_attention_mask, candidate_token_type_ids, = tokenize_sentences(desc,
                                                                                                              candidate_code)

                if len(candidate_input_ids) != 90:
                    continue

                candidate_prediction = model.predict(x=[np.array(candidate_input_ids).reshape((1, -1)),
                                                        np.array(candidate_attention_mask).reshape((1, -1)),
                                                        np.array(candidate_token_type_ids).reshape((1, -1))])[0]

                # this means negative relation

                if candidate_prediction[0] >= best_result:
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


def get_top_n(n, results):
    count = 0
    for r in results:
        if results[r] < n:
            count += 1
    return count / len(results)





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


if __name__ == "__main__":

    args = sys.argv
    data_chunk_id = 0
    if len(args) > 1:
        data_chunk_id = int(args[1])

    script_path = str(pathlib.Path(__file__).parent)

    data_path = script_path + "/../data/deep-code-search/drive/"

    monobert = MONOBERT_DCS(data_path)

    vocabulary_tokens, vocabulary_desc = monobert.get_vocabularies()

    longer_desc = 90

    number_desc_tokens = len(vocabulary_desc)
    number_code_tokens = len(vocabulary_tokens)

    MAX_LEN = 90

    bert_layer = monobert.generate_bert_layer()

    monobert.bert_layer.trainable = True

    model = monobert.generate_model()

    monobert.generate_tokenizer()

    #retokenized_desc, retokenized_mask, retokenizedtype, labels = generate_dataset()

    dataset = DataGeneratorDCSMonoBERT(data_path+"train.tokens.h5", data_path+"train.desc.h5",
                                       16, 0, 600000, 90, monobert.tokenizer, monobert.vocab_code, monobert.vocab_desc)

    #monobert.load_weights(script_path + "/../weights/monobert_dcs_weights")

    monobert.train(dataset, script_path+"/../weights/sroberta_dcs_weights", epochs=1, batch_size=None)

    #model.save_weights(script_path+"/../weights/monobert_dcs_weights")

    #monobert.test()