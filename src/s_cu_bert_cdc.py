
import sys
import pathlib
import os

import pandas as pd
import numpy as np
import transformers
import tensorflow as tf
from tensorflow.keras import backend as K
from transformers.models.bert import convert_bert_original_tf_checkpoint_to_pytorch
from transformers import BertConfig, TFBertModel

from .cubert.cubert_hug_tokenizer import CuBertHugTokenizer

class scu_bert_cdc():

    def __init__(self, data_path, data_chunk_id=0):

        self.data_path = data_path
        self.triplet_loss_margin = 0.6
        self.max_len = 128
        self.batch_size = 32

        self.bert_layer = None
        self.cubert_layer = None
        self.bert_tokenizer = None
        self.cubert_tokenizer = None

        self.training_model, self.code_model, self.desc_model, self.dot_model = None, None, None, None

    def get_bert_layers(self, path):
        self.bert_layer = transformers.TFBertModel.from_pretrained('bert-large-uncased')
        self.cubert_layer = self.get_cubert_layer(path)


    def get_cubert_layer(self, path):
        MODEL_PATH = path+'/../cuBERTconfig/20200621_Python_function_docstring__epochs_20__pre_trained_epochs_1_model.ckpt-6072.index'
        MODEL_CONFIG = path+'/../cuBERTconfig/cubert_config.json'
        MODEL_VOCAB = path+'/../cuBERTconfig/vocab.txt'
        MAX_SEQUENCE_LENGTH = 512
        MODEL_PATH_TORCH = path+'/../cuBERTconfig/20200621_Python_function_docstring__epochs_20__pre_trained_epochs_1_model.ckpt-6072.bin'
        if not os.path.isfile(MODEL_PATH_TORCH):
            convert_bert_original_tf_checkpoint_to_pytorch.convert_tf_checkpoint_to_pytorch(MODEL_PATH, MODEL_CONFIG, MODEL_PATH_TORCH)
        model_config = BertConfig.from_json_file(MODEL_CONFIG)
        return TFBertModel.from_pretrained(pretrained_model_name_or_path=MODEL_PATH_TORCH, from_pt=True,
                                                   config=model_config)
    def get_cubert_tokenizer(self, path):
        MODEL_VOCAB = path + '/../cuBERTconfig/vocab.txt'
        return CuBertHugTokenizer(MODEL_VOCAB)

    def generate_tokenizers(self, path):
        self.bert_tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        self.cubert_tokenizer = self.get_cubert_tokenizer(path)


    def generate_model(self):

        def mean_pooling(model_output, attention_mask):
            token_embeddings = model_output[0]  # First element of model_output contains all token embeddings

            input_mask_expanded = tf.repeat(tf.expand_dims(attention_mask, -1), token_embeddings.shape[-1], axis=-1)
            input_mask_expanded = tf.dtypes.cast(input_mask_expanded, tf.float32)
            sum_embeddings = tf.math.reduce_sum(token_embeddings * input_mask_expanded, 1)

            sum_mask = tf.keras.backend.clip(tf.math.reduce_sum(input_mask_expanded, 1), min_value=0, max_value=1000000)

            return sum_embeddings / sum_mask

        # Desc embedding
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

        desc_output = tf.keras.layers.Lambda(lambda x: mean_pooling(x[0], x[1]), name="desc_pooling")([bert_desc_output[0], input_mask_desc])

        # Code embedding
        input_word_ids_code = tf.keras.layers.Input(shape=(self.max_len,),
                                                    dtype=tf.int32,
                                                    name="input_word_ids_code")
        input_mask_code = tf.keras.layers.Input(shape=(self.max_len,),
                                                dtype=tf.int32,
                                                name="input_mask_code")
        segment_ids_code = tf.keras.layers.Input(shape=(self.max_len,),
                                                 dtype=tf.int32,
                                                 name="segment_ids_code")

        bert_code_output = self.cubert_layer([input_word_ids_code, input_mask_code, segment_ids_code])

        code_output = tf.keras.layers.Lambda(lambda x: mean_pooling(x[0], x[1]), name="code_pooling")(
            [bert_code_output[0], input_mask_code])

        similarity = tf.keras.layers.Dot(axes=1, normalize=True)([desc_output, code_output])

        # Used in tests
        embedded_code = tf.keras.Input(shape=(code_output.shape[1],), name="embedded_code")
        embedded_desc = tf.keras.Input(shape=(desc_output.shape[1],), name="embedded_desc")

        dot = tf.keras.layers.Dot(axes=1, normalize=True)([embedded_code, embedded_desc])
        dot_model = tf.keras.Model(inputs=[embedded_code, embedded_desc], outputs=[dot],
                                   name='dot_model')

        # Cosine model
        cos_model = tf.keras.models.Model(
            inputs=[input_word_ids_desc, input_mask_desc, segment_ids_desc,
                    input_word_ids_code, input_mask_code, segment_ids_code],
            outputs=similarity
        )

        # Embedding models
        embedding_desc_model = tf.keras.models.Model(
            inputs=[input_word_ids_desc, input_mask_desc, segment_ids_desc],
            outputs=desc_output
        )

        embedding_code_model = tf.keras.models.Model(
            inputs=[input_word_ids_code, input_mask_code, segment_ids_code],
            outputs=code_output
        )

        # Training model
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

        loss = tf.keras.layers.Lambda(lambda x: K.maximum(1e-6, self.triplet_loss_margin - x[0] + x[1]),
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


    def load_dataset(self):
        code_df = pd.read_csv(self.data_path+'data_ps.bodies.train.gz', compression='gzip', header=None,
                              sep='\n', quotechar='"', error_bad_lines=False)
        desc_df = pd.read_csv(self.data_path+'data_ps.descriptions.train', header=None, sep='\n',
                              quotechar='"', error_bad_lines=False)

        # pandas shuffle
        neg_code_df = code_df.sample(frac=1).reset_index(drop=True)

        dataset = tf.data.Dataset.from_tensor_slices((desc_df.values, code_df.values, neg_code_df.values))

        dataset = dataset.map(self.tokenize_map)

        #dataset = dataset.shuffle(2048)
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        dataset = dataset.batch(self.batch_size)

        return dataset

    def tokenize_desc(self, input):
        _input = input
        if isinstance(input, str):
            _input = [input]

        encoded = self.bert_tokenizer.batch_encode_plus(
            _input,
            add_special_tokens=True,
            max_length=self.max_len,
            return_attention_mask=True,
            return_token_type_ids=True,
            padding='max_length',
            truncation=True,
            return_tensors="np"
        )
        if isinstance(input, str):
            return encoded["input_ids"][0], encoded["attention_mask"][0], encoded["token_type_ids"][0]
        return encoded["input_ids"], encoded["attention_mask"], encoded["token_type_ids"]


    def tokenize_code(self, input):
        _input = input
        if isinstance(input, str):
            _input = [input]

        encoded = self.cubert_tokenizer.batch_encode_plus(
            _input,
            add_special_tokens=True,
            max_length=self.max_len,
            return_attention_mask=True,
            return_token_type_ids=True,
            padding='max_length',
            truncation=True,
            return_tensors="np"
        )

        if isinstance(input, str):
            return encoded["input_ids"][0], encoded["attention_mask"][0], encoded["token_type_ids"][0]
        return encoded["input_ids"], encoded["attention_mask"], encoded["token_type_ids"]


    def tokenize_map(self, desc,code,neg):
        desc_ = tf.reshape(desc, (1,))
        desc_tokenized = tf.py_function(lambda x: tf.constant(self.tokenize_desc(x[0].numpy().decode('utf-8'))), [desc_],
                                        tf.int32)

        desc_ids = tf.squeeze(tf.slice(desc_tokenized, [0, 0], [1, self.max_len]), [0])
        desc_attention = tf.squeeze(tf.slice(desc_tokenized, [1, 0], [1, self.max_len]), [0])
        desc_type = tf.squeeze(tf.slice(desc_tokenized, [2, 0], [1, self.max_len]), [0])

        code_ = tf.reshape(code, (1,))
        code_tokenized = tf.py_function(lambda x: tf.constant(self.tokenize_code(x[0].numpy().decode('utf-8'))), [code_],
                                        tf.int32)

        code_ids = tf.squeeze(tf.slice(code_tokenized, [0, 0], [1, self.max_len]), [0])
        code_attention = tf.squeeze(tf.slice(code_tokenized, [1, 0], [1, self.max_len]), [0])
        code_type = tf.squeeze(tf.slice(code_tokenized, [2, 0], [1, self.max_len]), [0])

        neg_ = tf.reshape(neg, (1,))
        neg_tokenized = tf.py_function(lambda x: tf.constant(self.tokenize_code(x[0].numpy().decode('utf-8'))), [neg_], tf.int32)

        neg_ids = tf.squeeze(tf.slice(neg_tokenized, [0, 0], [1, self.max_len]), [0])
        neg_attention = tf.squeeze(tf.slice(neg_tokenized, [1, 0], [1, self.max_len]), [0])
        neg_type = tf.squeeze(tf.slice(neg_tokenized, [2, 0], [1, self.max_len]), [0])

        return (desc_ids, desc_attention, desc_type,
                code_ids, code_attention, code_type,
                neg_ids, neg_attention, neg_type
                ), tf.constant(0.)


    def train(self, dataset, weights_path, steps_per_epoch=None ):
        print("Training model...")
        self.training_model.fit(dataset, epochs=2, steps_per_epoch=steps_per_epoch)
        self.training_model.save_weights(weights_path)
        print("Model saved!")

    def load_weights(self, path):
        if os.path.isfile(path + '.index'):
            self.training_model.load_weights(path)
            print("Weights loaded!")
        else:
            print("Warning! No weights loaded!")


if __name__ == "__main__":

    args = sys.argv

    script_path = str(pathlib.Path(__file__).parent)

    data_path = script_path + "/../data/code-docstring-corpus/"

    code_search = scu_bert_cdc(data_path)

    code_search.generate_tokenizers( script_path)

    code_search.batch_size = 64*2

    dataset = code_search.load_dataset()


    hardcoded_number_items = 109108
    steps_per_epoch = hardcoded_number_items // code_search.batch_size

    code_search.get_bert_layers(script_path)

    code_search.bert_layer.trainable = True
    code_search.cubert_layer.trainable = True

    code_search.generate_model()

    #code_search.load_weights(script_path + "/../final_weights/s_cu_bert_cdc_weights")

    code_search.train(dataset, "../weights/s_cu_bert_cdc_weights", steps_per_epoch)



