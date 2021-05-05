#subprocess.check_call([sys.executable, "-m", "pip", "install", "tables"])
#subprocess.check_call([sys.executable, "-m", "pip", "install", "tqdm"])

#subprocess.check_call([sys.executable, "-m", "pip", "install", "bert-tensorflow==1.0.1"])
#subprocess.check_call([sys.executable, "-m", "pip", "install", "tf-hub-nightly"])

#subprocess.check_call([sys.executable, "-m", "pip", "install", "transformers"])

import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

import tensorflow as tf
from tensorflow.keras import backend as K
import pathlib
from help import *
from data_generators.sentence_bert_dcs_generator import DataGeneratorDCSBERT
from code_search_manager import CodeSearchManager
from transformers.models.bert import convert_bert_original_tf_checkpoint_to_pytorch
from transformers import BertConfig, TFBertModel
from cubert.cubert_hug_tokenizer import CuBertHugTokenizer

class ScuBERT_DCS(CodeSearchManager):

    def __init__(self, data_path, data_chunk_id=0):

        self.tokenizer = None

        self.data_path = data_path
        self.max_len = 90
        # dataset info
        self.total_length = 18223872
        self.chunk_size = 100000   # 18223872  # 10000


        number_chunks = self.total_length / self.chunk_size - 1
        self.number_chunks = int(number_chunks + 1 if number_chunks > int(number_chunks) else number_chunks)

        self.data_chunk_id = min(data_chunk_id, int(self.number_chunks))
        print("### Loading SRoBERTa model with DCS chunk number " + str(data_chunk_id) + " [0," + str(number_chunks)+"]")

    def get_dataset_meta_hardcoded(self):
        return 86, 410, 10001, 10001

    def get_dataset_meta(self):
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


    def generate_model(self, bert_layer):

        input_word_ids_desc = tf.keras.layers.Input(shape=(self.max_len,),
                                                    dtype=tf.int32,
                                                    name="input_word_ids_desc")
        input_mask_desc = tf.keras.layers.Input(shape=(self.max_len,),
                                                dtype=tf.int32,
                                                name="input_mask_desc")
        segment_ids_desc = tf.keras.layers.Input(shape=(self.max_len,),
                                                 dtype=tf.int32,
                                                 name="segment_ids_desc")

        bert_desc_output = bert_layer([input_word_ids_desc, input_mask_desc, segment_ids_desc])


        desc_output = tf.reduce_mean(bert_desc_output[0], 1)

        input_word_ids_code = tf.keras.layers.Input(shape=(self.max_len,),
                                                    dtype=tf.int32,
                                                    name="input_word_ids_code")
        input_mask_code = tf.keras.layers.Input(shape=(self.max_len,),
                                                dtype=tf.int32,
                                                name="input_mask_code")
        segment_ids_code = tf.keras.layers.Input(shape=(self.max_len,),
                                                 dtype=tf.int32,
                                                 name="segment_ids_code")

        bert_code_output = bert_layer([input_word_ids_code, input_mask_code, segment_ids_code])

        code_output = tf.reduce_mean(bert_code_output[0], 1)

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

        hinge_loss_margin = 0.2
        loss = tf.keras.layers.Lambda(lambda x: K.maximum(1e-6, hinge_loss_margin - x[0] + x[1]),
                                      output_shape=lambda x: x[0],
                                      name='loss')([good_similarity, bad_similarity])

        training_model = tf.keras.Model(inputs=[
            good_ids_desc, good_mask_desc, good_seg_desc,
            good_ids_code, good_mask_code, good_seg_code,

            bad_ids_code, bad_mask_code, bad_seg_code], outputs=[loss],
            name='training_model')

        training_model.compile(loss=lambda y_true, y_pred: y_pred + y_true - y_true, optimizer='adam')

        return training_model, embedding_code_model, embedding_desc_model, dot_model


    def tokenize_sentences(self, input1_str, input2_str):
        tokenized = self.tokenizer.batch_encode_plus(
            [[input1_str, input2_str]],
            add_special_tokens=True,
            max_length=self.max_len,
            return_attention_mask=True,
            return_token_type_ids=True,
            pad_to_max_length=True,
            return_tensors="np",
        )

        return tokenized["input_ids"][0], tokenized["attention_mask"][0], tokenized["token_type_ids"][0]

    def test(self, model_code, model_query, dot_model, results_path, number_of_elements=100):
        test_tokens = load_hdf5(self.data_path + "test.tokens.h5" , 0, number_of_elements)
        test_desc = load_hdf5(self.data_path + "test.desc.h5" , 0, number_of_elements) # 10000

        code_test_vector = test_tokens
        desc_test_vector = test_desc

        embedded_tokens = []
        embedded_desc = []

        print("Embedding tokens and desc...")
        for idx, token in enumerate(code_test_vector):
            desc = (" ".join([vocab_desc[x] for x in desc_test_vector[idx]]))
            code = (" ".join([vocab_tokens[x] for x in code_test_vector[idx]]))

            desc_ = self.tokenize_sentences(desc, "")
            code_ = self.tokenize_sentences(code, "")

            embedded_tokens.append(model_code.predict([np.array(code_[0]).reshape((1, -1)),
                                                       np.array(code_[1]).reshape((1, -1)),
                                                       np.array(code_[2]).reshape((1, -1))

                                                       ])[0])

            embedded_desc.append(model_query.predict([np.array(desc_[0]).reshape((1, -1)),
                                                     np.array(desc_[1]).reshape((1, -1)),
                                                     np.array(desc_[2]).reshape((1, -1))

                                                     ])[0])

        self.test_embedded(dot_model, embedded_tokens, embedded_desc, results_path)



    def training_data_chunk(self, id, valid_perc):

        init_trainig = self.chunk_size * id
        init_valid = int(self.chunk_size * id + self.chunk_size * valid_perc)
        end_valid = int(self.chunk_size * id + self.chunk_size)

        return init_trainig, init_valid, end_valid

    def train(self, trainig_model, training_set, weights_path, epochs=1, batch_size=None):
        trainig_model.fit(training_set, epochs=epochs, verbose=1, batch_size=batch_size)
        trainig_model.save_weights(weights_path)
        print("Model saved!")



    def get_cubert_layer(self, path):
        MODEL_PATH = path+'/../cuBERTconfig/20200621_Python_function_docstring__epochs_20__pre_trained_epochs_1_model.ckpt-6072.index'
        MODEL_CONFIG = path+'/../cuBERTconfig/cubert_config.json'
        MODEL_VOCAB = path+'/../cuBERTconfig/vocab.txt'
        MAX_SEQUENCE_LENGTH = 512
        MODEL_PATH_TORCH = path+'/../cuBERTconfig/20200621_Python_function_docstring__epochs_20__pre_trained_epochs_1_model.ckpt-6072.bin'
        if not os.path.isfile(MODEL_PATH_TORCH):
            convert_bert_original_tf_checkpoint_to_pytorch.convert_tf_checkpoint_to_pytorch(MODEL_PATH, MODEL_CONFIG, MODEL_PATH_TORCH)
        model_config = BertConfig.from_json_file(MODEL_CONFIG)
        cubert_layer = TFBertModel.from_pretrained(pretrained_model_name_or_path=MODEL_PATH_TORCH, from_pt=True,
                                                   config=model_config)

        return cubert_layer

if __name__ == "__main__":

    args = sys.argv
    data_chunk_id = 0
    if len(args) > 1:
        data_chunk_id = int(args[1])

    script_path = str(pathlib.Path(__file__).parent)

    data_path = script_path + "/../data/deep-code-search/drive/"

    scubert_dcs = ScuBERT_DCS(data_path, data_chunk_id)

    BATCH_SIZE = 32 * 1

    #bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1",
    #                            trainable=False)
    # Some verion incompatibility requires this line
    #tf.gfile = tf.io.gfile

    # Get bert tokenizer
    #bert_layer.resolved_object.vocab_file.asset_path.numpy()
    #vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
    #do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
    #tokenizer = FullTokenizer(vocab_file, do_lower_case)
    #sbert_dcs.tokenizer = tokenizer

    multi_gpu = False

    print("Building model and loading weights")
    if multi_gpu:
        tf.debugging.set_log_device_placement(False)

        strategy = tf.distribute.MirroredStrategy()

        with strategy.scope():
            #bert_layer = transformers.TFBertModel.from_pretrained("bert-base-uncased")
            bert_layer = scubert_dcs.get_cubert_layer(script_path)

            training_model, model_code, model_query, dot_model = scubert_dcs.sbert_dcs.generate_model(bert_layer)
            #scubert_dcs.load_weights(training_model, script_path+"/../weights/scubert_dcs_weights")
    else:
        #bert_layer = transformers.TFBertModel.from_pretrained("bert-base-uncased")
        bert_layer = scubert_dcs.get_cubert_layer(script_path)
        training_model, model_code, model_query, dot_model = scubert_dcs.generate_model(bert_layer)
        #scubert_dcs.sbert_dcs.load_weights(training_model, script_path+"/../weights/scubert_dcs_weights")
        #scubert_dcs.load_weights(training_model, script_path + "/../weights/scubert_dcs_weights")

    MODEL_VOCAB = script_path + '/../cuBERTconfig/vocab.txt'
    tokenizer = CuBertHugTokenizer(MODEL_VOCAB)
    #sbert_dcs.tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
    scubert_dcs.tokenizer = tokenizer


    file_format = "h5"

    # 18223872 (len) #1000000
    #train_tokens = load_hdf5(data_path + "train.tokens." + file_format, 0, 18223872)  # 1000000
    #train_desc = load_hdf5(data_path + "train.desc." + file_format, 0, 18223872)

    vocabulary_tokens = load_pickle(data_path + "vocab.tokens.pkl")
    vocab_tokens = {y: x for x, y in vocabulary_tokens.items()}

    vocabulary_desc = load_pickle(data_path + "vocab.desc.pkl")
    vocab_desc = {y: x for x, y in vocabulary_desc.items()}

    dataset = DataGeneratorDCSBERT(data_path + "train.tokens." + file_format, data_path + "train.desc." + file_format,
                                   8, 0, 100000, 90, tokenizer, vocab_tokens, vocab_desc)



    print("Not trained results")
    scubert_dcs.test(model_code, model_query, dot_model, script_path+"/../results/sentence-cubert", 100)

    bert_layer.trainable = True

    scubert_dcs.train(training_model, dataset, script_path+"/../weights/scubert_dcs_weights", 1)

    print("Trained results with 100")
    scubert_dcs.test(model_code, model_query, dot_model, script_path+"/../results/sentence-cubert", 100)

    print("Trained results with 1000")
    scubert_dcs.test(model_code, model_query, dot_model, script_path+"/../results/sentence-cubert", 1000)
