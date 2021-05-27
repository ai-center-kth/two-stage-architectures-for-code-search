
import sys
import tensorflow as tf
from tensorflow.keras import backend as K
import pathlib
import pandas as pd
import transformers
import numpy as np
from tqdm import tqdm

from . import help
from .data_generators.sentence_bert_dcs_generator import DataGeneratorDCSBERT
from .code_search_manager import CodeSearchManager
from .data_generators import data_generator

class SBERT_DCS(CodeSearchManager):

    def __init__(self, data_path, data_chunk_id=0):

        self.tokenizer = None

        self.data_path = data_path
        self.max_len = 90
        # dataset info
        self.total_length = 18223872
        self.chunk_size = 600000   # 18223872  # 10000

        self.vocab_desc = None
        self.vocab_code = None
        self.inverse_vocab_tokens = None
        self.inverse_vocab_desc = None

        number_chunks = self.total_length / self.chunk_size - 1
        self.number_chunks = int(number_chunks + 1 if number_chunks > int(number_chunks) else number_chunks)

        self.data_chunk_id = min(data_chunk_id, int(self.number_chunks))
        print("### Loading SRoBERTa model with DCS chunk number " + str(data_chunk_id) + " [0," + str(number_chunks)+"]")

        self.training_model, self.code_model, self.desc_model, self.dot_model = None, None, None, None
        self.bert_layer = None

    def get_dataset_meta_hardcoded(self):
        return 86, 410, 10001, 10001

    def get_dataset_meta(self):
        # 18223872 (len) #1000000
        code_vector = help.load_hdf5(self.data_path + "train.tokens.h5", 0, 18223872)
        desc_vector = help.load_hdf5(self.data_path + "train.desc.h5", 0, 18223872)
        vocabulary_merged = help.load_pickle(data_path + "vocab.merged.pkl")

        longer_code = max(len(t) for t in code_vector)
        print("longer_code", longer_code)
        longer_desc = max(len(t) for t in desc_vector)
        print("longer_desc", longer_desc)

        longer_sentence = max(longer_code, longer_desc)

        number_tokens = len(vocabulary_merged)

        return longer_sentence, number_tokens

    def get_vocabularies(self):
        self.inverse_vocab_tokens = help.load_pickle(self.data_path + "vocab.tokens.pkl")
        self.vocab_tokens = {y: x for x, y in self.inverse_vocab_tokens.items()}

        self.inverse_vocab_desc = help.load_pickle(self.data_path + "vocab.desc.pkl")
        self.vocab_desc = {y: x for x, y in self.inverse_vocab_desc.items()}

        return self.vocab_tokens, self.vocab_desc

    def generate_tokenizer(self):
        self.tokenizer = transformers.RobertaTokenizer.from_pretrained('roberta-base', do_lower_case=True)
        return self.tokenizer

    def generate_bert_layer(self):
        self.bert_layer = transformers.TFRobertaModel.from_pretrained('roberta-base')
        return self.bert_layer

    def generate_model(self):

        def mean_pooling(model_output, attention_mask):
            token_embeddings = model_output[0]  # First element of model_output contains all token embeddings

            input_mask_expanded = tf.repeat(tf.expand_dims(attention_mask, -1), token_embeddings.shape[-1], axis=-1)
            input_mask_expanded = tf.dtypes.cast(input_mask_expanded, tf.float32)
            sum_embeddings = tf.math.reduce_sum(token_embeddings * input_mask_expanded, 1)

            sum_mask = tf.keras.backend.clip(tf.math.reduce_sum(input_mask_expanded, 1), min_value=0, max_value=1000000)

            return sum_embeddings / sum_mask

        input_word_ids_desc = tf.keras.layers.Input(shape=(self.max_len,),
                                                    dtype=tf.int32,
                                                    name="input_word_ids_desc")
        input_mask_desc = tf.keras.layers.Input(shape=(self.max_len,),
                                                dtype=tf.int32,
                                                name="input_mask_desc")
        segment_ids_desc = tf.keras.layers.Input(shape=(self.max_len,),
                                                 dtype=tf.int32,
                                                 name="segment_ids_desc")

        bert_desc_output = self.bert_layer([input_word_ids_desc, input_mask_desc, segment_ids_desc])

        desc_output = tf.keras.layers.Lambda(lambda x: mean_pooling(x[0], x[1]), name="desc_pooling")([bert_desc_output, input_mask_desc])
        #desc_output = tf.reduce_mean(bert_desc_output, 1, name="desc_pooling")

        input_word_ids_code = tf.keras.layers.Input(shape=(self.max_len,),
                                                    dtype=tf.int32,
                                                    name="input_word_ids_code")
        input_mask_code = tf.keras.layers.Input(shape=(self.max_len,),
                                                dtype=tf.int32,
                                                name="input_mask_code")
        segment_ids_code = tf.keras.layers.Input(shape=(self.max_len,),
                                                 dtype=tf.int32,
                                                 name="segment_ids_code")

        bert_code_output = self.bert_layer([input_word_ids_code, input_mask_code, segment_ids_code])

        code_output = tf.keras.layers.Lambda(lambda x: mean_pooling(x[0], x[1]), name="code_pooling")(
            [bert_code_output, input_mask_code])

        similarity = tf.keras.layers.Dot(axes=1, normalize=True)([desc_output, code_output])

        # Used in tests
        embedded_code = tf.keras.Input(shape=(code_output.shape[1],), name="embedded_code")
        embedded_desc = tf.keras.Input(shape=(desc_output.shape[1],), name="embedded_desc")

        dot = tf.keras.layers.Dot(axes=1, normalize=True)([embedded_code, embedded_desc])
        dot_model = tf.keras.Model(inputs=[embedded_code, embedded_desc], outputs=[dot],
                                   name='dot_model')

        cos_model = tf.keras.models.Model(
            inputs=[input_word_ids_desc, input_mask_desc, segment_ids_desc,
                    input_word_ids_code, input_mask_code, segment_ids_code],
            outputs=similarity
        )

        embedding_desc_model = tf.keras.models.Model(
            inputs=[input_word_ids_desc, input_mask_desc, segment_ids_desc],
            outputs=desc_output
        )

        embedding_code_model = tf.keras.models.Model(
            inputs=[input_word_ids_code, input_mask_code, segment_ids_code],
            outputs=code_output
        )

        good_ids_desc = tf.keras.layers.Input(shape=(self.max_len,),
                                              dtype=tf.int32)
        good_mask_desc = tf.keras.layers.Input(shape=(self.max_len,),
                                               dtype=tf.int32)
        good_seg_desc = tf.keras.layers.Input(shape=(self.max_len,),
                                              dtype=tf.int32)

        good_ids_code = tf.keras.layers.Input(shape=(self.max_len,),
                                              dtype=tf.int32)
        good_mask_code = tf.keras.layers.Input(shape=(self.max_len,),
                                               dtype=tf.int32)
        good_seg_code = tf.keras.layers.Input(shape=(self.max_len,),
                                              dtype=tf.int32)

        bad_ids_code = tf.keras.layers.Input(shape=(self.max_len,),
                                             dtype=tf.int32)
        bad_mask_code = tf.keras.layers.Input(shape=(self.max_len,),
                                              dtype=tf.int32)
        bad_seg_code = tf.keras.layers.Input(shape=(self.max_len,),
                                             dtype=tf.int32)

        good_similarity = cos_model(
            [good_ids_desc, good_mask_desc, good_seg_desc, good_ids_code, good_mask_code, good_seg_code])

        bad_similarity = cos_model(
            [good_ids_desc, good_mask_desc, good_seg_desc, bad_ids_code, bad_mask_code, bad_seg_code])

        hinge_loss_margin = 0.6
        loss = tf.keras.layers.Lambda(lambda x: K.maximum(1e-6, hinge_loss_margin - x[0] + x[1]),
                                      output_shape=lambda x: x[0],
                                      name='loss')([good_similarity, bad_similarity])

        training_model = tf.keras.Model(inputs=[
            good_ids_desc, good_mask_desc, good_seg_desc,
            good_ids_code, good_mask_code, good_seg_code,

            bad_ids_code, bad_mask_code, bad_seg_code], outputs=[loss],
            name='training_model')

        opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
        training_model.compile(loss=lambda y_true, y_pred: y_pred + y_true - y_true, optimizer=opt)

        self.training_model, self.code_model, self.desc_model, self.dot_model = training_model, embedding_code_model, embedding_desc_model, dot_model
        return training_model, embedding_code_model, embedding_desc_model, dot_model



    def generate_embeddings(self, number_of_elements=100):
        test_tokens = help.load_hdf5(self.data_path + "test.tokens.h5" , 0, number_of_elements)
        test_desc = help.load_hdf5(self.data_path + "test.desc.h5" , 0, number_of_elements)

        embedded_tokens = []
        embedded_desc = []

        pbar = tqdm(total=len(test_tokens))
        print("Embedding tokens and desc...")
        for idx, token in enumerate(test_tokens):
            desc = (" ".join([self.vocab_desc[x] for x in test_desc[idx]]))
            code = (" ".join([self.vocab_tokens[x] for x in test_tokens[idx]]))

            desc_ = self.tokenize(desc)
            code_ = self.tokenize(code)

            result = self.code_model.predict([np.array(code_[0]).reshape((1, -1)),
                                     np.array(code_[1]).reshape((1, -1)),
                                     np.array(code_[2]).reshape((1, -1))

                                     ])


            embedded_tokens.append(self.code_model.predict([np.array(code_[0]).reshape((1, -1)),
                                                       np.array(code_[1]).reshape((1, -1)),
                                                       np.array(code_[2]).reshape((1, -1))

                                                       ])[0])

            embedded_desc.append(self.desc_model.predict([np.array(desc_[0]).reshape((1, -1)),
                                                     np.array(desc_[1]).reshape((1, -1)),
                                                     np.array(desc_[2]).reshape((1, -1))

                                                     ])[0])
            pbar.update(1)
        pbar.close()
        
        return embedded_tokens, embedded_desc

    def test(self, results_path, number_of_elements=100):

        embedded_tokens, embedded_desc = self.generate_embeddings(number_of_elements)
        #self.test_embedded(embedded_tokens, embedded_desc, results_path)

        df = pd.read_csv(self.data_path + "descriptions.csv", header=0)
        df = df.dropna()
        df = df[df["rowid"] < number_of_elements]

        self.rephrasing_test(df, embedded_tokens, embedded_desc)

    def tokenize(self, input_str):
        return DataGeneratorDCSBERT.tokenize_sentences(self.tokenizer, 90, input_str)

    ''''
    def load_dataset(self, batch_size=32):
        init_trainig, init_valid, end_valid = self.training_data_chunk(data_chunk_id)
        return DataGeneratorDCSBERT(self.data_path + "train.tokens.h5", self.data_path + "train.desc.h5",
                             batch_size, init_trainig, init_valid, 90, self.tokenizer, self.vocab_tokens, self.vocab_desc)
    '''



    def load_dataset(self, batch_size=32):

        # ds output is (desc, code, neg_code) strings
        ds = data_generator.get_dcs_dataset(self.data_path + "train.desc.h5", self.data_path + "train.tokens.h5",
                                            self.vocab_desc, self.vocab_tokens, max_len=self.chunk_size)

        # Tokenize the dataset
        ds = ds.map(data_generator.sentece_bert_tokenizer_map(self.tokenize, self.max_len))

        #ds = ds.map(self.mapeo)

        ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        ds = ds.batch(batch_size, drop_remainder=True)

        return ds


if __name__ == "__main__":

    args = sys.argv
    data_chunk_id = 0
    if len(args) > 1:
        data_chunk_id = int(args[1])

    script_path = str(pathlib.Path(__file__).parent)

    data_path = script_path + "/../data/deep-code-search/drive/"

    sbert_dcs = SBERT_DCS(data_path, data_chunk_id)

    multi_gpu = False

    print("Building model and loading weights")
    if multi_gpu:
        tf.debugging.set_log_device_placement(False)

        strategy = tf.distribute.MirroredStrategy()

        with strategy.scope():
            sbert_dcs.generate_bert_layer()

            training_model, model_code, desc_model, dot_model = sbert_dcs.generate_model()
            #sbert_dcs.load_weights(script_path+"/../final_weights/sroberta_dcs_weights")
    else:
        sbert_dcs.generate_bert_layer()

        training_model, model_code, desc_model, dot_model = sbert_dcs.generate_model()
        #sbert_dcs.load_weights(script_path+"/../final_weights/sroberta_dcs_weights")

    sbert_dcs.generate_tokenizer()

    sbert_dcs.get_vocabularies()

    BATCH_SIZE = 16
    dataset = sbert_dcs.load_dataset(BATCH_SIZE)

    print("Not trained results")

    sbert_dcs.bert_layer.trainable = True

    steps_per_epoch = sbert_dcs.chunk_size // BATCH_SIZE
    sbert_dcs.train(dataset, script_path+"/../weights/sroberta_600k_0001_dcs_weights",  epochs=1, steps_per_epoch=steps_per_epoch)

    print("Trained results with 100")
    sbert_dcs.test(script_path+"/../results/sentence-roberta", 500)

    #print("Trained results with 1000")
    #sbert_dcs.test(model_code, desc_model, dot_model, script_path+"/../results/sentence-roberta", 200)