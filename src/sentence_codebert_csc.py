import sys
import tensorflow as tf
from tensorflow.keras import backend as K
import pathlib
import pandas as pd
import transformers
import numpy as np
from tqdm import tqdm
from pathlib import Path
import time

from . import help
from .data_generators.sentence_bert_dcs_generator import DataGeneratorDCSBERT
from .code_search_manager import CodeSearchManager
from .tfrecord_parser import TFRecordParser


class SCODEBERT_CSC(CodeSearchManager):

    def __init__(self, data_path, data_chunk_id=0):
        self.tokenizer = None

        self.data_path = data_path
        self.max_len = 90
        # dataset info
        self.total_length = 18223872
        self.chunk_size = 300000  # 18223872  # 10000

        self.vocab_desc = None
        self.vocab_code = None
        self.inverse_vocab_tokens = None
        self.inverse_vocab_desc = None

        number_chunks = self.total_length / self.chunk_size - 1
        self.number_chunks = int(number_chunks + 1 if number_chunks > int(number_chunks) else number_chunks)

        self.data_chunk_id = min(data_chunk_id, int(self.number_chunks))
        print("### Loading SRoBERTa model with DCS chunk number " + str(data_chunk_id) + " [0," + str(
            number_chunks) + "]")

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
        self.tokenizer = transformers.RobertaTokenizer.from_pretrained('microsoft/codebert-base', do_lower_case=True)
        return self.tokenizer

    def generate_bert_layer(self):
        self.bert_layer = transformers.TFRobertaModel.from_pretrained('microsoft/codebert-base')
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

        desc_output = tf.keras.layers.Lambda(lambda x: mean_pooling(x[0], x[1]), name="desc_pooling")(
            [bert_desc_output[0], input_mask_desc])
        # desc_output = tf.reduce_mean(bert_desc_output[0], 1, name="desc_pooling")

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
            [bert_code_output[0], input_mask_code])
        # code_output = tf.reduce_mean(bert_code_output[0], 1, name="code_pooling")

        similarity = tf.keras.layers.Dot(axes=1, normalize=True)([desc_output, code_output])

        # Used in tests
        embedded_code = tf.keras.Input(shape=(code_output.shape[1],), name="embedded_code")
        embedded_desc = tf.keras.Input(shape=(desc_output.shape[1],), name="embedded_desc")

        dot = tf.keras.layers.Dot(axes=1, normalize=True)([embedded_code, embedded_desc])
        dot_model = tf.keras.Model(inputs=[embedded_code, embedded_desc], outputs=[dot],
                                   name='dot_model')

        # output = tf.keras.layers.Dense(1, activation="sigmoid")(dropout)

        cos_model = tf.keras.models.Model(
            inputs=[input_word_ids_desc, input_mask_desc, segment_ids_desc,
                    input_word_ids_code, input_mask_code, segment_ids_code],
            outputs=similarity
        )

        # cos_model.compile(loss='mse', optimizer='nadam', metrics=['mse'])

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

        opt = tf.keras.optimizers.Adam(learning_rate=0.000001)
        training_model.compile(loss=lambda y_true, y_pred: y_pred + y_true - y_true, optimizer=opt)

        self.training_model, self.code_model, self.desc_model, self.dot_model = training_model, embedding_code_model, embedding_desc_model, dot_model
        return training_model, embedding_code_model, embedding_desc_model, dot_model

    def generate_embeddings(self, number_of_elements=100):
        test_tokens = help.load_hdf5(self.data_path + "test.tokens.h5", 0, number_of_elements)
        test_desc = help.load_hdf5(self.data_path + "test.desc.h5", 0, number_of_elements)

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
        self.test_embedded(embedded_tokens, embedded_desc, results_path)

        df = pd.read_csv(self.data_path + "descriptions.csv", header=0)
        df = df.dropna()
        df = df[df["rowid"] < number_of_elements]

        self.rephrasing_test(df, embedded_tokens, embedded_desc)

    #def tokenize(self, input_str):
    #    return DataGeneratorDCSBERT.tokenize_sentences(self.tokenizer, 90, input_str)

    def tokenize(self, string):
        encoded = self.tokenizer.batch_encode_plus(
            [string],
            add_special_tokens=True,
            max_length=self.max_len,
            return_attention_mask=True,
            return_token_type_ids=True,
            padding='max_length',
            truncation=True
            # return_tensors="tf",
        )
        return encoded["input_ids"][0], encoded["attention_mask"][0], encoded["token_type_ids"][0]

    def tokenize_map(self, inputs, outputs):
        desc_ = tf.reshape(inputs[0], (1,))
        desc_tokenized = tf.py_function(lambda x: tf.constant(self.tokenize(x[0].numpy().decode('utf-8'))), [desc_],
                                        tf.int32)

        desc_ids = tf.squeeze(tf.slice(desc_tokenized, [0, 0], [1, self.max_len]), [0])
        desc_attention = tf.squeeze(tf.slice(desc_tokenized, [1, 0], [1, self.max_len]), [0])
        desc_type = tf.squeeze(tf.slice(desc_tokenized, [2, 0], [1, self.max_len]), [0])

        code_ = tf.reshape(inputs[1], (1,))
        code_tokenized = tf.py_function(lambda x: tf.constant(self.tokenize(x[0].numpy().decode('utf-8'))), [code_],
                                        tf.int32)

        code_ids = tf.squeeze(tf.slice(code_tokenized, [0, 0], [1, self.max_len]), [0])
        code_attention = tf.squeeze(tf.slice(code_tokenized, [1, 0], [1, self.max_len]), [0])
        code_type = tf.squeeze(tf.slice(code_tokenized, [2, 0], [1, self.max_len]), [0])

        neg_ = tf.reshape(inputs[2], (1,))
        neg_tokenized = tf.py_function(lambda x: tf.constant(self.tokenize(x[0].numpy().decode('utf-8'))), [neg_], tf.int32)

        neg_ids = tf.squeeze(tf.slice(neg_tokenized, [0, 0], [1, self.max_len]), [0])
        neg_attention = tf.squeeze(tf.slice(neg_tokenized, [1, 0], [1, self.max_len]), [0])
        neg_type = tf.squeeze(tf.slice(neg_tokenized, [2, 0], [1, self.max_len]), [0])

        return (desc_ids, desc_attention, desc_type,
                code_ids, code_attention, code_type,
                neg_ids, neg_attention, neg_type
                ), outputs

    def load_dataset(self):

        tfr_files = sorted(Path(data_path + 'python/train/').glob('**/*.tfrecord'))

        tfr_files = [x.__str__() for x in tfr_files]

        BATCH_SIZE = 8 * 1

        dataset = TFRecordParser.generate_dataset(tfr_files, BATCH_SIZE)

        dataset = dataset.map(self.tokenize_map)

        dataset = dataset.shuffle(2048)
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        dataset = dataset.batch(BATCH_SIZE)

        return dataset

    def train(self, training_set_generator, weights_path, steps_per_epoch ):
        print("Training model...")
        self.training_model.fit(training_set_generator, epochs=1, steps_per_epoch=steps_per_epoch)
        self.training_model.save_weights(weights_path)
        print("Model saved!")



def test(dataset, code_model, desc_model, dot_model, results_path):

    print("Testing model...")
    print(code_model)
    # Hardcoded
    dataset_size = 100 # 22176
    input_element = iter(dataset.batch(dataset_size)).get_next()[0]

    desc_id   = input_element[0].numpy()
    desc_att  = input_element[1].numpy()
    desc_type = input_element[2].numpy()

    code_id   = input_element[3].numpy()
    code_att  = input_element[4].numpy()
    code_type = input_element[5].numpy()


    code_embeddings = []
    desc_embeddings = []

    print("Embedding code and descriptions...")
    pbar = tqdm(total=dataset_size)
    for i in range(0,dataset_size):

        desc_embeddings.append(desc_model.predict([(desc_id[i]).reshape((1, -1)),
                                                   (desc_att[i]).reshape((1, -1)),
                                                   (desc_type[i]).reshape((1, -1))

                                                        ])[0])

        code_embeddings.append(code_model.predict([(code_id[i]).reshape((1, -1)),
                                                 (code_att[i]).reshape((1, -1)),
                                                 (code_type[i]).reshape((1, -1))

                                                      ])[0])


        pbar.update(1)
    pbar.close()

    print("Testing...")
    results = {}
    pbar = tqdm(total=len(desc_embeddings))
    for rowid, desc in enumerate(desc_embeddings):

        expected_best_result = dot_model.predict([code_embeddings[rowid].reshape((1, -1)), desc_embeddings[rowid].reshape((1, -1))])[0][0]

        print(expected_best_result)

        exit()
        deleted_tokens = np.delete(desc_embeddings, rowid, 0)

        tiled_desc = np.tile(desc, (deleted_tokens.shape[0], 1))

        prediction = dot_model.predict([deleted_tokens, tiled_desc]) # , batch_size=32*4

        results[rowid] = len(prediction[prediction >= expected_best_result])

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


if __name__ == "__main__":

    args = sys.argv
    data_chunk_id = 0
    if len(args) > 1:
        data_chunk_id = int(args[1])

    script_path = str(pathlib.Path(__file__).parent)

    data_path = script_path + "/../data/codesearchnet/tfrecord/"

    BATCH_SIZE = 32 * 1

    sbert_dcs = SCODEBERT_CSC(data_path, data_chunk_id)

    sbert_dcs.generate_tokenizer()

    dataset = sbert_dcs.load_dataset()


    multi_gpu = False

    print("Building model and loading weights")
    if multi_gpu:
        tf.debugging.set_log_device_placement(False)

        strategy = tf.distribute.MirroredStrategy()

        with strategy.scope():
            sbert_dcs.generate_bert_layer()

            training_model, model_code, desc_model, dot_model = sbert_dcs.generate_model()
            #sbert_dcs.load_weights(script_path + "/../final_weights/scodebert_dcs_weights")
    else:
        sbert_dcs.generate_bert_layer()

        training_model, model_code, desc_model, dot_model = sbert_dcs.generate_model()
        #sbert_dcs.load_weights(script_path + "/../final_weights/scodebert_dcs_weights")

    sbert_dcs.generate_tokenizer()

    print("Not trained results")
    # sbert_dcs.test(script_path+"/../results/sentence-codebert", 100)

    sbert_dcs.bert_layer.trainable = True

    steps_per_epoch = 412178 // BATCH_SIZE

    #sbert_dcs.train(dataset, script_path+"/../weights/scodebert_dcs_weights", steps_per_epoch)

    print("Trained results with 100")

    test_files = sorted(Path(data_path + 'python/test/').glob('**/*.tfrecord'))
    test_files = [x.__str__() for x in test_files]
    test_dataset = TFRecordParser.generate_dataset(test_files, 1)

    test_dataset = test_dataset.map(sbert_dcs.tokenize_map)

    test(test_dataset, model_code, desc_model, dot_model, script_path + "/../results")

    #sbert_dcs.test(script_path + "/../results/sentence-codebert", 100)

    print("Trained results with 200")
    # sbert_dcs.test(script_path+"/../results/sentence-codebert", 200)