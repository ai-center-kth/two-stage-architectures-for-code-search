
import tensorflow as tf
from tensorflow.keras import backend as K

def unif_model(embedding_size, number_code_tokens, number_desc_tokens, code_length, desc_length, hinge_loss_margin):

    code_input = tf.keras.Input(shape=(code_length,), name="code_input")
    code_embeding = tf.keras.layers.Embedding(number_code_tokens, embedding_size, name="code_embeding")(code_input)

    attention_code = tf.keras.layers.Attention(name="attention_code")([code_embeding, code_embeding])

    query_input = tf.keras.Input(shape=(desc_length,), name="query_input")
    query_embeding = tf.keras.layers.Embedding(number_desc_tokens, embedding_size, name="query_embeding")(
        query_input)

    code_output = tf.keras.layers.Lambda(lambda x: K.sum(x, axis=1), name="sum")(attention_code)
    query_output = tf.keras.layers.Lambda(lambda x: K.mean(x, axis=1), name="average")(query_embeding)

    # This model generates code embedding
    model_code = tf.keras.Model(inputs=[code_input], outputs=[code_output], name='model_code')
    # This model generates description/query embedding
    model_query = tf.keras.Model(inputs=[query_input], outputs=[query_output], name='model_query')

    # Cosine similarity
    # If normalize set to True, then the output of the dot product is the cosine proximity between the two samples.
    cos_sim = tf.keras.layers.Dot(axes=1, normalize=True, name='cos_sim')([code_output, query_output])

    # This model calculates cosine similarity between code and query pairs
    cos_model = tf.keras.Model(inputs=[code_input, query_input], outputs=[cos_sim], name='sim_model')

    # Used in tests
    embedded_code = tf.keras.Input(shape=(code_output.shape[1],), name="embedded_code")
    embedded_desc = tf.keras.Input(shape=(query_output.shape[1],), name="embedded_desc")

    dot = tf.keras.layers.Dot(axes=1, normalize=True)([embedded_code, embedded_desc])
    dot_model = tf.keras.Model(inputs=[embedded_code, embedded_desc], outputs=[dot],
                               name='dot_model')

    loss = tf.keras.layers.Flatten()(cos_sim)
    # training_model = tf.keras.Model(inputs=[ code_input, query_input], outputs=[cos_sim],name='training_model')

    model_code.compile(loss='cosine_proximity', optimizer='adam')
    model_query.compile(loss='cosine_proximity', optimizer='adam')

    cos_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=["accuracy"])  # extract similarity

    # Negative sampling
    good_desc_input = tf.keras.Input(shape=(desc_length,), name="good_desc_input")
    bad_desc_input = tf.keras.Input(shape=(desc_length,), name="bad_desc_input")

    good_desc_output = cos_model([code_input, good_desc_input])
    bad_desc_output = cos_model([code_input, bad_desc_input])

    loss = tf.keras.layers.Lambda(lambda x: K.maximum(1e-6, hinge_loss_margin - x[0] + x[1]),
                                  output_shape=lambda x: x[0],
                                  name='loss')([good_desc_output, bad_desc_output])

    training_model = tf.keras.Model(inputs=[code_input, good_desc_input, bad_desc_input], outputs=[loss],
                                    name='training_model')

    opt = tf.keras.optimizers.Adam(learning_rate=0.001)

    training_model.compile(loss=lambda y_true, y_pred: y_pred + y_true - y_true, optimizer=opt)
    # y_true-y_true avoids warning

    return training_model, model_code, model_query, dot_model


def unif_snn_model(embedding_size, number_tokens, sentence_length, hinge_loss_margin):
    input_layer = tf.keras.Input(shape=(sentence_length,), name="input")
    embedding_layer = tf.keras.layers.Embedding(number_tokens, embedding_size, name="embeding")(input_layer)

    attention_layer = tf.keras.layers.Attention(name="attention")([embedding_layer, embedding_layer])

    sum_layer = tf.keras.layers.Lambda(lambda x: K.sum(x, axis=1), name="sum")(attention_layer)
    # average_layer = tf.keras.layers.Lambda(lambda x: K.mean(x, axis=1), name="average")( attention_layer)

    embedding_model = tf.keras.Model(inputs=[input_layer], outputs=[sum_layer], name='siamese_model')

    input_code = tf.keras.Input(shape=(sentence_length,), name="code")
    input_desc = tf.keras.Input(shape=(sentence_length,), name="desc")
    input_bad_desc = tf.keras.Input(shape=(sentence_length,), name="bad_desc")

    output_code = embedding_model(input_code)
    output_desc = embedding_model(input_desc)
    output_bad_desc = embedding_model(input_bad_desc)

    cos_good_sim = tf.keras.layers.Dot(axes=1, normalize=True, name='cos_good_sim')([output_code, output_desc])

    cos_model = tf.keras.Model(inputs=[input_code, input_desc], outputs=[cos_good_sim],
                                    name='cos_model')

    # Used in tests
    embedded_code = tf.keras.Input(shape=(output_code.shape[1],), name="embedded_code")
    embedded_desc = tf.keras.Input(shape=(output_code.shape[1],), name="embedded_desc")

    dot = tf.keras.layers.Dot(axes=1, normalize=True)([embedded_code, embedded_desc])
    dot_model = tf.keras.Model(inputs=[embedded_code, embedded_desc], outputs=[dot],
                                    name='dot_model')

    cos_bad_sim = tf.keras.layers.Dot(axes=1, normalize=True, name='cos_bad_sim')([output_code, output_bad_desc])

    loss = tf.keras.layers.Lambda(lambda x: K.maximum(1e-6, hinge_loss_margin - x[0] + x[1]),
                                  output_shape=lambda x: x[0],
                                  name='loss')([cos_good_sim, cos_bad_sim])

    training_model = tf.keras.Model(inputs=[input_code, input_desc, input_bad_desc], outputs=[loss],
                                    name='training_model')

    opt = tf.keras.optimizers.Adam(learning_rate=0.001)

    training_model.compile(loss=lambda y_true, y_pred: y_pred + y_true - y_true, optimizer=opt)
    # y_true-y_true avoids warning

    return training_model, embedding_model, embedding_model, dot_model

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings

    input_mask_expanded = tf.repeat(tf.expand_dims(attention_mask, -1), token_embeddings.shape[-1], axis=-1)
    input_mask_expanded = tf.dtypes.cast(input_mask_expanded, tf.float32)
    sum_embeddings = tf.math.reduce_sum(token_embeddings * input_mask_expanded, 1)

    sum_mask = tf.keras.backend.clip(tf.math.reduce_sum(input_mask_expanded, 1), min_value=0, max_value=1000000)

    return sum_embeddings / sum_mask

def sentence_bert_model(bert_layer, max_len):


    input_word_ids_desc = tf.keras.layers.Input(shape=(max_len,),
                                                dtype=tf.int32,
                                                name="input_word_ids_desc")
    input_mask_desc = tf.keras.layers.Input(shape=(max_len,),
                                            dtype=tf.int32,
                                            name="input_mask_desc")
    segment_ids_desc = tf.keras.layers.Input(shape=(max_len,),
                                             dtype=tf.int32,
                                             name="segment_ids_desc")

    bert_desc_output = bert_layer([input_word_ids_desc, input_mask_desc, segment_ids_desc])

    desc_output = tf.keras.layers.Lambda(lambda x: mean_pooling(x[0], x[1]))([bert_desc_output, input_mask_desc])

    input_word_ids_code = tf.keras.layers.Input(shape=(max_len,),
                                                dtype=tf.int32,
                                                name="input_word_ids_code")
    input_mask_code = tf.keras.layers.Input(shape=(max_len,),
                                            dtype=tf.int32,
                                            name="input_mask_code")
    segment_ids_code = tf.keras.layers.Input(shape=(max_len,),
                                             dtype=tf.int32,
                                             name="segment_ids_code")

    bert_code_output = bert_layer([input_word_ids_code, input_mask_code, segment_ids_code])

    code_output = tf.keras.layers.Lambda(lambda x: mean_pooling(x[0], x[1]))([bert_code_output, input_mask_code])

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

    good_ids_desc = tf.keras.layers.Input(shape=(max_len,),
                                          dtype=tf.int32)
    good_mask_desc = tf.keras.layers.Input(shape=(max_len,),
                                           dtype=tf.int32)
    good_seg_desc = tf.keras.layers.Input(shape=(max_len,),
                                          dtype=tf.int32)

    good_ids_code = tf.keras.layers.Input(shape=(max_len,),
                                          dtype=tf.int32)
    good_mask_code = tf.keras.layers.Input(shape=(max_len,),
                                           dtype=tf.int32)
    good_seg_code = tf.keras.layers.Input(shape=(max_len,),
                                          dtype=tf.int32)

    bad_ids_code = tf.keras.layers.Input(shape=(max_len,),
                                         dtype=tf.int32)
    bad_mask_code = tf.keras.layers.Input(shape=(max_len,),
                                          dtype=tf.int32)
    bad_seg_code = tf.keras.layers.Input(shape=(max_len,),
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

    return training_model, embedding_code_model, embedding_desc_model, dot_model



def mono_bert_model(bert_layer, max_len):
    # The model

    input_word_ids = tf.keras.layers.Input(shape=(max_len,),
                                           dtype=tf.int32,
                                           name="input_word_ids")
    input_mask = tf.keras.layers.Input(shape=(max_len,),
                                       dtype=tf.int32,
                                       name="input_mask")
    segment_ids = tf.keras.layers.Input(shape=(max_len,),
                                        dtype=tf.int32,
                                        name="segment_ids")

    bert_output = bert_layer([input_word_ids, input_mask, segment_ids])

    output = tf.keras.layers.Dense(1, activation="sigmoid")(bert_output[0][:,0,:])

    model = tf.keras.models.Model(
        inputs=[input_word_ids, input_mask, segment_ids], outputs=output
    )

    opt = tf.keras.optimizers.Adam(learning_rate=1e-5)

    model.compile(
        optimizer=opt,
        loss="binary_crossentropy",
        metrics=["acc"],
    )

    return model