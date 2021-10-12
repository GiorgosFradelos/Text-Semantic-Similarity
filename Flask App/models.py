import tensorflow as tf
import tensorflow_hub as hub

from keras.utils.vis_utils import plot_model
from keras_radam import RAdam
from keras import backend as K
from keras.models import load_model

import transformers

import numpy as np
import config


class BertSemanticDataGenerator(tf.keras.utils.Sequence):
    """Generates batches of data.
    Args:
        sentence_pairs: Array of premise and hypothesis input sentences.
        labels: Array of labels.
        batch_size: Integer batch size.
        shuffle: boolean, whether to shuffle the data.
        include_targets: boolean, whether to incude the labels.
    Returns:
        Tuples `([input_ids, attention_mask, `token_type_ids], labels)`
        (or just `[input_ids, attention_mask, `token_type_ids]`
         if `include_targets=False`)
    """

    def __init__(
        self,
        sentence_pairs,
        labels,
        batch_size,
        shuffle=True,
        include_targets=True,
    ):
        self.sentence_pairs = sentence_pairs
        self.labels = labels
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.include_targets = include_targets
        # Load our BERT Tokenizer to encode the text.
        # We will use base-base-uncased pretrained model.
        self.tokenizer = transformers.BertTokenizer.from_pretrained(
            "bert-base-uncased", do_lower_case=True
        )
        self.indexes = np.arange(len(self.sentence_pairs))
        self.on_epoch_end()

    def __len__(self):
        # Denotes the number of batches per epoch.
        return len(self.sentence_pairs) // self.batch_size

    def __getitem__(self, idx):
        # Retrieves the batch of index.
        indexes = self.indexes[idx * self.batch_size : (idx + 1) * self.batch_size]
        sentence_pairs = self.sentence_pairs[indexes]

        # With BERT tokenizer's batch_encode_plus batch of both the sentences are
        # encoded together and separated by [SEP] token.
        encoded = self.tokenizer.batch_encode_plus(
            sentence_pairs.tolist(),
            add_special_tokens=True,
            max_length=config.MAX_LEN,
            return_attention_mask=True,
            return_token_type_ids=True,
            pad_to_max_length=True,
            return_tensors="tf",
        )

        # Convert batch of encoded features to numpy array.
        input_ids = np.array(encoded["input_ids"], dtype="int32")
        attention_masks = np.array(encoded["attention_mask"], dtype="int32")
        token_type_ids = np.array(encoded["token_type_ids"], dtype="int32")

    #    print(input_ids, attention_masks, token_type_ids)

        # Set to true if data generator is used for training/validation.
        if self.include_targets:
            labels = np.array(self.labels[indexes], dtype="int32")
            return [input_ids, attention_masks, token_type_ids], labels
        else:
            return [input_ids, attention_masks, token_type_ids]

    def on_epoch_end(self):
        # Shuffle indexes after each epoch if shuffle is set to True.
        if self.shuffle:
            np.random.RandomState(42).shuffle(self.indexes)


"""
## Create a custom data generator
"""
class SiameseBertSemanticDataGenerator(tf.keras.utils.Sequence):
    """Generates batches of data.
    Args:
        sentence_pairs: Array of premise and hypothesis input sentences.
        labels: Array of labels.
        batch_size: Integer batch size.
        shuffle: boolean, whether to shuffle the data.
        include_targets: boolean, whether to incude the labels.
    Returns:
        Tuples `([input_ids_1, input_ids_2, attention_masks_1, attention_masks_2, token_type_ids_1, token_type_ids_2], labels)`
        (or just `[input_ids_1, input_ids_2, attention_masks_1, attention_masks_2, token_type_ids_1, token_type_ids_2]`
         if `include_targets=False`)
    """

    def __init__(
        self,
        sentence_1,
        sentence_2,
        labels,
        batch_size,
        shuffle=True,
        include_targets=True,
    ):
        self.sentence_1 = sentence_1
        self.sentence_2 = sentence_2
        self.labels = labels
        self.shuffle = shuffle
        self.batch_size=batch_size
        self.include_targets = include_targets
        # Load our BERT Tokenizer to encode the text.
        # We will use base-base-uncased pretrained model.

        self.tokenizer = transformers.BertTokenizer.from_pretrained(
            "bert-base-uncased", do_lower_case=True
        )

        self.indexes = np.arange(len(self.sentence_2))
        self.on_epoch_end()

    def __len__(self):
        # Denotes the number of batches per epoch.
        return len(self.sentence_1) // self.batch_size

    def __getitem__(self, idx):
        # Retrieves the batch of index.
        indexes = self.indexes[idx * self.batch_size : (idx + 1) * self.batch_size]
        sentence_1 = self.sentence_1[indexes]
        sentence_2 = self.sentence_2[indexes]

        # With BERT tokenizer's batch_encode_plus batch of both the sentences are
        # encoded together and separated by [SEP] token.
        encoded_1 = self.tokenizer.batch_encode_plus(
            sentence_1.tolist(),
            add_special_tokens=True,
            max_length=config.MAX_LEN,
            return_attention_mask=True,
            return_token_type_ids=True,
            pad_to_max_length=True,
            return_tensors="tf",
        )
        encoded_2 = self.tokenizer.batch_encode_plus(
            sentence_2.tolist(),
            add_special_tokens=True,
            max_length=config.MAX_LEN,
            return_attention_mask=True,
            return_token_type_ids=True,
            pad_to_max_length=True,
            return_tensors="tf",
        )

        # Convert batch of encoded features to numpy array.
        input_ids_1 = np.array(encoded_1["input_ids"], dtype="int32")
        attention_masks_1 = np.array(encoded_1["attention_mask"], dtype="int32")
        token_type_ids_1 = np.array(encoded_1["token_type_ids"], dtype="int32")

        input_ids_2 = np.array(encoded_2["input_ids"], dtype="int32")
        attention_masks_2 = np.array(encoded_2["attention_mask"], dtype="int32")
        token_type_ids_2 = np.array(encoded_2["token_type_ids"], dtype="int32")

    #    print('\nSentence1\n', sentence_1[0])
    #    print(input_ids_1[0])
    #    print('\n', attention_masks_2[0])
    #    print('\n', token_type_ids_1[0])

    #    print('\nSentence2\n', sentence_2[0])
    #    print(input_ids_2[0])
    #    print('\n', attention_masks_2[0])
    #    print('\n', token_type_ids_2[0])


        # Set to true if data generator is used for training/validation.
        if self.include_targets:
            labels = np.array(self.labels[indexes], dtype="int32")
            return [input_ids_1, input_ids_2, attention_masks_1, attention_masks_2, token_type_ids_1, token_type_ids_2], labels
        else:
            return [input_ids_1, input_ids_2, attention_masks_1, attention_masks_2, token_type_ids_1, token_type_ids_2]

    def on_epoch_end(self):
        # Shuffle indexes after each epoch if shuffle is set to True.
        if self.shuffle:
            np.random.RandomState(42).shuffle(self.indexes)


''' Build the model '''
def BERT_model(L, H, A, trainable):


    # Encoded token ids from BERT tokenizer.
    input_ids = tf.keras.layers.Input(
        shape=(config.MAX_LEN,), dtype=tf.int32, name="input_ids"
    )

    # Attention masks indicates to the model which tokens should be attended to.
    attention_masks = tf.keras.layers.Input(
        shape=(config.MAX_LEN,), dtype=tf.int32, name="attention_masks"
    )

    # Token type ids are binary masks identifying different sequences in the model.
    token_type_ids = tf.keras.layers.Input(
        shape=(config.MAX_LEN,), dtype=tf.int32, name="token_type_ids"
    )

    ''' Anamones '''
    encoder_inputs = dict(
        input_word_ids=input_ids,
        input_mask=attention_masks,
        input_type_ids=token_type_ids,
    )



 #   model_path = r"C:\Users\Giorgos\PycharmProjects\SiameseSentenceSimilarity-master\Encoders\BERT\small_bert_bert_en_uncased_L-%s_H-%s_A-%s_2" % (L, H, A)  # [BERT small] 2 Layers, 128, 2 Att. heads
    model_path = r"https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-512_A-8/2"  # [BERT small] 2 Layers, 128, 2 Att. heads
    print('\n\nModel to be used:\t', model_path, '\n\n')
    bert_model = hub.KerasLayer(model_path)


    # Freeze the BERT model to reuse the pretrained features without modifying them.
    bert_model.trainable = trainable


    sequence_output = bert_model(encoder_inputs)["sequence_output"]

    ''' Add trainable layers on top of frozen layers to adapt the pretrained features on the new data. '''
    bi_lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(H, return_sequences=True))(sequence_output)
    dropout = tf.keras.layers.Dropout(0.3)(bi_lstm)

    bi_lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True))(dropout)
    dropout = tf.keras.layers.Dropout(0.3)(bi_lstm)


    bi_lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True))(dropout)

    # Applying hybrid pooling approach to bi_lstm sequence output.
    avg_pool = tf.keras.layers.GlobalAveragePooling1D()(bi_lstm)
    max_pool = tf.keras.layers.GlobalMaxPooling1D()(bi_lstm)


    concat = tf.keras.layers.concatenate([avg_pool, max_pool])
    dropout = tf.keras.layers.Dropout(0.3)(concat)

    output = tf.keras.layers.Dense(3, activation="softmax")(dropout)



    model = tf.keras.models.Model(
        inputs=[input_ids, attention_masks, token_type_ids], outputs=output
    )

    opt = RAdam(total_steps=5000, warmup_proportion=0.2, min_lr=1e-5)
    model.compile(loss="categorical_crossentropy",
                  optimizer=opt,
                  metrics=['accuracy'],
                  )

 #   plot_model(model, to_file='model/Saved Models/bert/BERT_LSTM_model_plot.png', show_shapes=True, show_layer_names=True)

    print('\n')
    model.summary()
    print('\n')

    if L == 2 and H == 256:
        model_path = r'C:\Users\Giorgos\Desktop\Thesis_backup\model_Checkpoints/BERT_small_BiLSTM_finetuned_L-2_H-256_A-4_0.8369.h5'
    elif L == 8 and H == 128:
        model_path = r'BERT_small_BiLSTM_finetuned_L-8_H-128_A-4_L0.4344_A0.8396.h5'
    else:
        model_path = r'BERT_small_BiLSTM_finetuned_L-12_H-512_A-8_0.8744.h5'

    model.load_weights(model_path)

    return model



'''Compute manhattan distance Similarity'''
def exponent_neg_manhattan_distance(inputX):
    (sent_left, sent_right) = inputX
    return K.exp(-K.sum(K.abs(sent_left - sent_right), axis=1, keepdims=True))

'''Create a coding level network for weight distribution'''
def create_base_network(H, input_shape):
    print('input_shape=', input_shape)
    input = tf.keras.layers.Input(shape=input_shape)

    # Add trainable layers on top of frozen layers to adapt the pretrained features on the new data.
    bi_lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(H,
                                                                 return_sequences=True
                                                                 )
                                            )(input)
    # Applying hybrid pooling approach to bi_lstm sequence output.
    dropout = tf.keras.layers.Dropout(0.3)(bi_lstm)

    bi_lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128,
                                                                 return_sequences=True
                                                                 )
                                            )(dropout)
    dropout = tf.keras.layers.Dropout(0.3)(bi_lstm)

    bi_lstm = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(64,
                             return_sequences=True
                             )
    )(dropout)
    dropout = tf.keras.layers.Dropout(0.3)(bi_lstm)

    model = tf.keras.models.Model(input, dropout)

#    plot_model(model, to_file='model/Saved Models/bert/Siamese BERT/Siamsese_BERT_base_network_plot.png', show_shapes=True, show_layer_names=True)
    return model

'''Build a network'''
def siamese_bert_model(L, H, A, trainable):
    # Encoded token ids from BERT tokenizer.
    input_ids_1 = tf.keras.layers.Input(
        shape=(config.MAX_LEN,), dtype=tf.int32, name="input_ids_1"
    )

    input_ids_2 = tf.keras.layers.Input(
        shape=(config.MAX_LEN,), dtype=tf.int32, name="input_ids_2"
    )

    # Attention masks indicates to the model which tokens should be attended to.
    attention_masks_1 = tf.keras.layers.Input(
        shape=(config.MAX_LEN,), dtype=tf.int32, name="attention_masks_1"
    )

    # Attention masks indicates to the model which tokens should be attended to.
    attention_masks_2 = tf.keras.layers.Input(
        shape=(config.MAX_LEN,), dtype=tf.int32, name="attention_masks_2"
    )

    # Token type ids are binary masks identifying different sequences in the model.
    token_type_ids_1 = tf.keras.layers.Input(
        shape=(config.MAX_LEN,), dtype=tf.int32, name="token_type_ids_1"
    )

    # Token type ids are binary masks identifying different sequences in the model.
    token_type_ids_2 = tf.keras.layers.Input(
        shape=(config.MAX_LEN,), dtype=tf.int32, name="token_type_ids_2"
    )

    encoder_1_inputs = dict(
        input_word_ids=input_ids_1,
        input_mask=attention_masks_1,
        input_type_ids=token_type_ids_1,
    )

    encoder_2_inputs = dict(
        input_word_ids=input_ids_2,
        input_mask=attention_masks_2,
        input_type_ids=token_type_ids_2,
    )

    ''' Loading pretrained BERT model '''
 #   model_path = r"C:\Users\Giorgos\PycharmProjects\SiameseSentenceSimilarity-master\Encoders\BERT\small_bert_bert_en_uncased_L-%s_H-%s_A-%s_2" % (L, H, A)  # [BERT small] 2 Layers, 128, 2 Att. heads
    model_path = r"https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-128_A-2/2"  # [BERT small] 2 Layers, 128, 2 Att. heads
    print('\n\nModel to be used:\t', model_path, '\n\n')
    bert_model = hub.KerasLayer(model_path)

    bert_model.trainable = trainable

    sequence_output_1 = bert_model(encoder_1_inputs)["sequence_output"]
    sequence_output_2 = bert_model(encoder_2_inputs)["sequence_output"]

    shared_lstm = create_base_network(H=H, input_shape=(config.MAX_LEN, H))

    left_output = shared_lstm(sequence_output_1)
    right_output = shared_lstm(sequence_output_2)

    # Applying hybrid pooling approach to bi_lstm sequence output.
    max_pool_1 = tf.keras.layers.GlobalMaxPooling1D()(left_output)
    max_pool_2 = tf.keras.layers.GlobalMaxPooling1D()(right_output)

    distance = tf.keras.layers.Lambda(exponent_neg_manhattan_distance)([max_pool_1, max_pool_2])

    model = tf.keras.models.Model(
        inputs=[input_ids_1, input_ids_2, attention_masks_1, attention_masks_2, token_type_ids_1, token_type_ids_2],
        outputs=distance
    )

    opt = RAdam(total_steps=5000, warmup_proportion=0.2, min_lr=1e-5)
    model.compile(
        loss="binary_crossentropy",
        optimizer=opt,
        metrics=['accuracy'],
    )

#    plot_model(model, to_file='Model_plots/Siamese_BERT_LSTM/Siamsese_BERT_LSTM_model_plot.png', show_shapes=True,
#              show_layer_names=True)


    model.load_weights(r'Siamese_BERT_small_BiLSTM_L-8_H-128_A-2_0.8197.h5')
    model.summary()
    return model