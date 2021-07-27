"""
Title: Semantic Similarity with BERT
Author: [Mohamad Merchant](https://twitter.com/mohmadmerchant1)
Date created: 2020/08/15
Last modified: 2020/08/29
Description: Natural Language Inference by fine-tuning BERT model on SNLI Corpus.
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from tensorflow.python.client import device_lib
from tensorflow_model_optimization import sparsity

import transformers

import keras
from keras.utils.vis_utils import plot_model
from keras_radam import RAdam
from keras import backend as K

import config

"""
## Introduction
Semantic Similarity is the task of determining how similar
two sentences are, in terms of what they mean.
This example demonstrates the use of SNLI (Stanford Natural Language Inference) Corpus
to predict sentence semantic similarity with Transformers.
We will fine-tune a BERT model that takes two sentences as inputs
and that outputs a similarity score for these two sentences.
### References
* [BERT](https://arxiv.org/pdf/1810.04805.pdf)
* [SNLI](https://nlp.stanford.edu/projects/snli/)
"""

"""
## Setup
Note: install HuggingFace `transformers` via `pip install transformers` (version >= 2.11.0).
"""


#from keras.legacy import interfaces
from keras.optimizers import Optimizer

#import runai.ga

#from keras_gradient_accumulation import GradientAccumulation
#from keras_gradient_accumulation import AdamAccumulated
#from keras_gradient_accumulation import GradientAccumulation

#from keras import legacy_tf_layers
#from keras_gradient_accumulation import optimizer_v1





# Labels in our dataset.
#labels = ["contradiction", "entailment", "neutral"]
labels = [0, 1]

"""
## Load the Data
"""

"""shell
curl -LO https://raw.githubusercontent.com/MohamadMerchant/SNLI/master/data.tar.gz
tar -xvzf data.tar.gz
"""
# There are more than 550k samples in total; we will use 100k for this example.
'''train_df = pd.read_csv("SNLI_Corpus/snli_1.0_train.csv", nrows=9000)
valid_df = pd.read_csv("SNLI_Corpus/snli_1.0_dev.csv", nrows=700)
test_df = pd.read_csv("SNLI_Corpus/snli_1.0_test.csv", nrows=1000)'''


#train_df = pd.read_csv("data/quora_duplicate_questions_train.txt", sep='\t', nrows=80000)
#valid_df = pd.read_csv("data/quora_duplicate_questions_val.txt", sep='\t', nrows=10000)
#test_df = pd.read_csv("data/quora_duplicate_questions_test.txt", sep='\t', nrows=40000)


# ---- Check for gpu ----------
def system_info():
    local_device_protos = device_lib.list_local_devices()
    #    print('\n--------- System Info ---------')
    #    print('\n_______________________________ System Info ______________________________________________')
    print('\n|----------------------------------------- System Info -----------------------------------------|')
    print('|\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t|')
    print('| Avaliable devices:\t', [x.name for x in local_device_protos], '\t\t\t\t\t\t\t\t\t|')
    # my output was => ['/device:CPU:0']
    # good output must be => ['/device:CPU:0', '/device:GPU:0']

    print('| Python Version:\t\t', sys.version, '\t|')
    # 3.6.4 |Anaconda custom (64-bit)| (default, Jan 16 2018, 10:22:32) [MSC v.1900 64 bit (AMD64)]

    print('| Tensorflow Version: \t', tf.__version__, '\t\t\t\t\t\t\t\t\t\t\t\t\t\t|')
    print('| Keras Version: \t\t', keras.__version__, '\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t|')
    print('| 3.6.4 |Anaconda custom (64-bit)| (default, Jan 16 2018, 10:22:32) [MSC v.1900 64 bit (AMD64)] |')
    print('|\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t|')
    print('|_______________________________________________________________________________________________|')
    print('\n')

    return [x.name for x in local_device_protos]

system_info()


def preprocess(train_df, valid_df, test_df):
    ''' Preprocessing '''
    # We have some NaN entries in our train data, we will simply drop them.
    print("\nNumber of missing values")
    print(train_df.isnull().sum(), '\n')
    train_df.dropna(axis=0, inplace=True)

    '''
    The value "-" appears as part of our training and validation targets.
    We will skip these samples.
    '''
    train_df = (
        train_df[train_df.similarity != "-"]
            .sample(frac=1.0, random_state=42)
            .reset_index(drop=True)
    )
    valid_df = (
        valid_df[valid_df.similarity != "-"]
            .sample(frac=1.0, random_state=42)
            .reset_index(drop=True)
    )
    test_df = (
        test_df[test_df.similarity != "-"]
            .sample(frac=1.0, random_state=42)
            .reset_index(drop=True)
    )

    return train_df, valid_df, test_df


def one_hot(train_df, valid_df, test_df):
    y_train = train_df["similarity"]
    y_val = valid_df["similarity"]
    y_test = test_df["similarity"]

    """
    ''' One-hot encode training, validation, and test labels '''
    train_df["label"] = train_df["similarity"].apply(
        lambda x: 0 if x == "contradiction" else 1 if x == "entailment" else 2
    )
    y_train = tf.keras.utils.to_categorical(train_df.label, num_classes=3)

    valid_df["label"] = valid_df["similarity"].apply(
        lambda x: 0 if x == "contradiction" else 1 if x == "entailment" else 2
    )
    y_val = tf.keras.utils.to_categorical(valid_df.label, num_classes=3)

    test_df["label"] = test_df["similarity"].apply(
        lambda x: 0 if x == "contradiction" else 1 if x == "entailment" else 2
    )
    y_test = tf.keras.utils.to_categorical(test_df.label, num_classes=3)
    """

    return y_train, y_val, y_test


def draw_train(history, accuracy_fig_path, loss_fig_path):
  #  accuracy_fig_path = r'model/Saved Models/bert/small_bert_bert_en_uncased_L-8_H-128_A-2_2 - BiLSTM/%s/Chunk %s/Accuracy-%s-%s-%s.jpg' % (path, chunk, epoch, val_loss, val_acc)
    #  Plot training & validation accuracy values
    fig1 = plt.figure(figsize=(12, 8))
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='lower right')

    axes = plt.gca()
    axes.set_ylim([0, 1])

 #   plt.show()

    fig1.savefig(accuracy_fig_path)
    plt.close(fig1)

 #   loss_fig_path = r'model/Saved Models/bert/small_bert_bert_en_uncased_L-8_H-128_A-2_2 - BiLSTM/%s/Chunk %s/Loss-%s-%s-%s.jpg' % (path, chunk, epoch, val_loss, val_acc)
    #  Plot training & validation loss values
    fig2 = plt.figure(figsize=(12, 8))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')

    axes = plt.gca()
    axes.set_ylim([0, 7])

 #   plt.show()
    fig2.savefig(loss_fig_path)
    plt.close(fig2)

    '''
    80000/80000 [==============================] - 379s 5ms/step - loss: 0.9952 - acc: 0.5792 - val_loss: 0.6363 - val_acc: 0.6319
    80000/80000 [==============================] - 373s 5ms/step - loss: 0.6295 - acc: 0.6491 - val_loss: 0.6823 - val_acc: 0.5935
    80000/80000 [==============================] - 375s 5ms/step - loss: 0.5932 - acc: 0.6821 - val_loss: 0.5657 - val_acc: 0.6982
    80000/80000 [==============================] - 373s 5ms/step - loss: 0.5662 - acc: 0.7043 - val_loss: 0.5363 - val_acc: 0.7248
    80000/80000 [==============================] - 373s 5ms/step - loss: 0.5402 - acc: 0.7259 - val_loss: 0.5782 - val_acc: 0.6806
    80000/80000 [==============================] - 374s 5ms/step - loss: 0.5225 - acc: 0.7405 - val_loss: 0.5719 - val_acc: 0.6972
    80000/80000 [==============================] - 372s 5ms/step - loss: 0.5049 - acc: 0.7503 - val_loss: 0.5146 - val_acc: 0.7370
    80000/80000 [==============================] - 371s 5ms/step - loss: 0.4896 - acc: 0.7629 - val_loss: 0.5571 - val_acc: 0.7109
    80000/80000 [==============================] - 371s 5ms/step - loss: 0.4754 - acc: 0.7721 - val_loss: 0.4836 - val_acc: 0.7655
    80000/80000 [==============================] - 371s 5ms/step - loss: 0.4639 - acc: 0.7792 - val_loss: 0.4713 - val_acc: 0.7731
    80000/80000 [==============================] - 544s 7ms/step - loss: 0.4519 - acc: 0.7864 - val_loss: 0.4567 - val_acc: 0.7824
    80000/80000 [==============================] - 33654s 421ms/step - loss: 0.4448 - acc: 0.7914 - val_loss: 0.4636 - val_acc: 0.7754
    80000/80000 [==============================] - 387s 5ms/step - loss: 0.4386 - acc: 0.7967 - val_loss: 0.4710 - val_acc: 0.7733
    80000/80000 [==============================] - 384s 5ms/step - loss: 0.4300 - acc: 0.8004 - val_loss: 0.5132 - val_acc: 0.7538
    80000/80000 [==============================] - 400s 5ms/step - loss: 0.4245 - acc: 0.8029 - val_loss: 0.4523 - val_acc: 0.7844
    80000/80000 [==============================] - 407s 5ms/step - loss: 0.4195 - acc: 0.8048 - val_loss: 0.4647 - val_acc: 0.7803
    80000/80000 [==============================] - 427s 5ms/step - loss: 0.4171 - acc: 0.8086 - val_loss: 0.4927 - val_acc: 0.7629
    80000/80000 [==============================] - 432s 5ms/step - loss: 0.4133 - acc: 0.8092 - val_loss: 0.4517 - val_acc: 0.7859
    80000/80000 [==============================] - 425s 5ms/step - loss: 0.4075 - acc: 0.8125 - val_loss: 0.4447 - val_acc: 0.7956
    80000/80000 [==============================] - 415s 5ms/step - loss: 0.4022 - acc: 0.8176 - val_loss: 0.4657 - val_acc: 0.7762
    '''



"""
## Create a custom data generator
"""
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

        '''self.tokenizer = transformers.AlbertTokenizer.from_pretrained(
            "albert-base-v2", do_lower_case=True
        )'''
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

        '''print('\nSentence1\n', sentence_1[0])
        print(input_ids_1[0])
        print('\n', input_ids_2[0])
        print('Sentence2\n', sentence_2[0])'''


    #    print('\nInput IDs ', len(input_ids_1), '\n', input_ids_1)

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



'''Compute manhattan distance Similarity'''
def exponent_neg_manhattan_distance(inputX):
    (sent_left, sent_right) = inputX
    return K.exp(-K.sum(K.abs(sent_left - sent_right), axis=1, keepdims=True))



def cosine_distance(inputX):
    sent_left, sent_right = inputX
    sent_left = K.l2_normalize(sent_left, axis=-1)
    sent_right = K.l2_normalize(sent_right, axis=-1)
    return -K.mean(sent_left * sent_right, axis=-1, keepdims=True)

def cos_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0],1)


'''Create a coding level network for weight distribution'''
def create_base_network(H, input_shape):


    print('input_shape=' , input_shape)
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

    plot_model(model, to_file='model/Saved Models/bert/Siamese BERT/Siamsese_BERT_base_network_plot.png', show_shapes=True, show_layer_names=True)
    return model


'''Build a network'''
def bilstm_siamese_model(L, H, A, trainable):
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
    model_path = r"Encoders\BERT\small_bert_bert_en_uncased_L-%s_H-%s_A-%s_2" % (L, H, A)  # [BERT small] 2 Layers, 128, 2 Att. heads
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

 #   concat = tf.keras.layers.concatenate([left_output, right_output])
 #   avg_pool = tf.keras.layers.GlobalAveragePooling1D()(concat)
 #   max_pool = tf.keras.layers.GlobalMaxPooling1D()(concat)

 #   concat = tf.keras.layers.concatenate([max_pool_1, max_pool_2])
 #   dropout = tf.keras.layers.Dropout(0.3)(concat)

#    avg_pooling = tf.keras.layers.GlobalAveragePooling1D()(concat)


#    dropout = tf.keras.layers.Dropout(0.3)(avg_pooling)


    distance = tf.keras.layers.Lambda(exponent_neg_manhattan_distance)([max_pool_1, max_pool_2])
 #   distance = tf.keras.layers.Lambda(cosine_distance, output_shape=cos_dist_output_shape)([avg_pool_1, avg_pool_2])
 #   output = tf.keras.layers.Dense(3, activation="softmax")(dropout)


    model = tf.keras.models.Model(
        inputs=[input_ids_1, input_ids_2, attention_masks_1, attention_masks_2, token_type_ids_1, token_type_ids_2], outputs=distance
    )


    opt = RAdam(total_steps=5000, warmup_proportion=0.2, min_lr=1e-5)
    model.compile(
    #    loss="categorical_crossentropy",
        loss="binary_crossentropy",
        optimizer=opt,
        metrics=['accuracy'],
    )


    plot_model(model, to_file='Model_plots/Siamese_BERT_LSTM/Siamsese_BERT_LSTM_model_plot.png', show_shapes=True, show_layer_names=True)

    model.summary()
    return model


L = int(input('Num of Layers: '))

H = int(input('Emb size: '))



if H==128:
    A=2
elif H==256:
    A=4
elif H==512:
    A=8
elif H == 768:
    A = 12

siamese_checkpoint_filepath = r'model/Saved Models/bert/Siamese BERT/small_bert_bert_en_uncased_L-%s_H-%s_A-%s_2 - %s/Siamese_BERT_small_BiLSTM.h5' % (L, H, A, config.model_type)

# Freeze the BERT model to reuse the pretrained features without modifying them.
trainable = False
model = bilstm_siamese_model(L, H, A, trainable)


chunk_count = 1
num_of_cunks = int(config.nrows / config.chunksize)

#with pd.read_csv("SNLI_Corpus/snli_1.0_train.csv", chunksize=config.chunksize, nrows=config.nrows) as reader:
#    valid_df = pd.read_csv("SNLI_Corpus/snli_1.0_dev.csv", nrows=10000)
#    test_df = pd.read_csv("SNLI_Corpus/snli_1.0_test.csv", nrows=10000)

valid_df = pd.read_csv("data/quora_duplicate_questions_val.txt", nrows=10000, delimiter='\t')
test_df = pd.read_csv("data/quora_duplicate_questions_test.txt", nrows=10000, delimiter='\t')

with pd.read_csv("data/quora_duplicate_questions_train.txt", chunksize=config.chunksize, nrows=config.nrows, delimiter='\t') as reader:

    for train_df in reader:

        print('\n\nChunk ', chunk_count, '/', num_of_cunks)

        ''' prepare dataframes and targets '''
        train_df, valid_df, test_df = preprocess(train_df, valid_df, test_df)
        y_train, y_val, y_test = one_hot(train_df, valid_df, test_df)


        """
        Create train and validation data generators
        """
        train_data = BertSemanticDataGenerator(
        #    train_df[["sentence1", "sentence2"]].values.astype("str"),
            train_df["sentence1"].values.astype("str"),
            train_df["sentence2"].values.astype("str"),
            y_train,
            batch_size=config.TRAIN_BATCH_SIZE,
            shuffle=False,
        )
        valid_data = BertSemanticDataGenerator(
        #    valid_df[["sentence1", "sentence2"]].values.astype("str"),
            valid_df["sentence1"].values.astype("str"),
            valid_df["sentence2"].values.astype("str"),
            y_val,
            batch_size=config.VALID_BATCH_SIZE,
            shuffle=False,
        )

        """
        ## Train the Model
        Training is done only for the top layers to perform "feature extraction",
        which will allow the model to use the representations of the pretrained model.
        """
        siamese_checkpoint_filepath = r'model/Saved Models/bert/Siamese BERT/small_bert_bert_en_uncased_L-%s_H-%s_A-%s_2 - %s/Siamese_BERT_small_BiLSTM.h5' % (L, H, A, config.model_type)


        model_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=siamese_checkpoint_filepath,
                                                              monitor='val_accuracy',
                                                              mode='auto',
                                                              save_best_only=True,
                                                              save_weights_only=True,
                                                              )

        history = model.fit(
            train_data,
            validation_data=valid_data,
            epochs=config.EPOCHS,
            callbacks=[model_checkpoint],
        )



        ''' Draw training curves '''
        accuracy_fig_path = r'model/Saved Models/bert/Siamese BERT/small_bert_bert_en_uncased_L-%s_H-%s_A-%s_2 - %s/%s/Chunk %s/Accuracy.jpg' % (L, H, A, config.model_type, 'Pre-train', chunk_count)
        loss_fig_path = r'model/Saved Models/bert/Siamese BERT/small_bert_bert_en_uncased_L-%s_H-%s_A-%s_2 - %s/%s/Chunk %s/Loss.jpg' % (L, H, A, config.model_type, 'Pre-train', chunk_count)
        draw_train(history, accuracy_fig_path, loss_fig_path)

        chunk_count = chunk_count + 1

        model.save_weights(siamese_checkpoint_filepath)






"""
## Fine-tuning
This step must only be performed after the feature extraction model has
been trained to convergence on the new data.
This is an optional last step where `bert_model` is unfreezed and retrained
with a very low learning rate. This can deliver meaningful improvement by
incrementally adapting the pretrained features to the new data.
"""



# Unfreeze the bert_model.
# Freeze the BERT model to reuse the pretrained features without modifying them.
trainable = True
model = bilstm_siamese_model(L, H, A, trainable)

model.load_weights(siamese_checkpoint_filepath)

chunk_count = 1
num_of_cunks = int(config.nrows / config.chunksize)

#with pd.read_csv("SNLI_Corpus/snli_1.0_train.csv", chunksize=config.chunksize, nrows=config.nrows) as reader:
#    valid_df = pd.read_csv("SNLI_Corpus/snli_1.0_dev.csv", nrows=10000)
#    test_df = pd.read_csv("SNLI_Corpus/snli_1.0_test.csv", nrows=10000)

valid_df = pd.read_csv("data/quora_duplicate_questions_val.txt", nrows=10000, delimiter='\t')
test_df = pd.read_csv("data/quora_duplicate_questions_test.txt", nrows=10000, delimiter='\t')

with pd.read_csv("data/quora_duplicate_questions_train.txt", chunksize=config.chunksize, nrows=config.nrows, delimiter='\t') as reader:

    for train_df in reader:

        print('\n\nChunk ', chunk_count, '/', num_of_cunks)

        ''' prepare dataframes and targets '''
        train_df, valid_df, test_df = preprocess(train_df, valid_df, test_df)
        y_train, y_val, y_test = one_hot(train_df, valid_df, test_df)


        """
        Create train and validation data generators
        """
        train_data = BertSemanticDataGenerator(
        #    train_df[["sentence1", "sentence2"]].values.astype("str"),
            train_df["sentence1"].values.astype("str"),
            train_df["sentence2"].values.astype("str"),
            y_train,
            batch_size=config.TRAIN_BATCH_SIZE,
            shuffle=True,
        )
        valid_data = BertSemanticDataGenerator(
        #    valid_df[["sentence1", "sentence2"]].values.astype("str"),
            valid_df["sentence1"].values.astype("str"),
            valid_df["sentence2"].values.astype("str"),
            y_val,
            batch_size=config.VALID_BATCH_SIZE,
            shuffle=False,
        )

        """
        ## Train the Model
        Training is done only for the top layers to perform "feature extraction",
        which will allow the model to use the representations of the pretrained model.
        """
        model_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=siamese_checkpoint_filepath,
                                                              monitor='val_accuracy',
                                                              mode='auto',
                                                              save_best_only=True,
                                                              save_weights_only=True,
                                                              )


        history = model.fit(
            train_data,
            validation_data=valid_data,
            epochs=config.EPOCHS,
            callbacks=[model_checkpoint],
        )

    #    siamese_checkpoint_filepath = r'model/Saved Models/bert/Siamese BERT/small_bert_bert_en_uncased_L-%s_H-%s_A-%s_2 - %s/Siamese_BERT_small_BiLSTM.h5' % (L, H, A, config.model_type)
        model.save_weights(siamese_checkpoint_filepath)


        ''' Draw training curves '''
        accuracy_fig_path = r'model/Saved Models/bert/Siamese BERT/small_bert_bert_en_uncased_L-%s_H-%s_A-%s_2 - %s/%s/Chunk %s/Accuracy.jpg' % (
        L, H, A, config.model_type, 'Finetune', chunk_count)
        loss_fig_path = r'model/Saved Models/bert/Siamese BERT/small_bert_bert_en_uncased_L-%s_H-%s_A-%s_2 - %s/%s/Chunk %s/Loss.jpg' % (
        L, H, A, config.model_type, 'Finetune', chunk_count)
        draw_train(history, accuracy_fig_path, loss_fig_path)

        chunk_count = chunk_count + 1

"""
## Evaluate model on the test set
"""
test_data = BertSemanticDataGenerator(
#    test_df[["sentence1", "sentence2"]].values.astype("str"),
    test_df["sentence1"].values.astype("str"),
    test_df["sentence2"].values.astype("str"),
    y_test,
    batch_size=config.VALID_BATCH_SIZE,
    shuffle=False,
)
model.evaluate(test_data, verbose=1)

