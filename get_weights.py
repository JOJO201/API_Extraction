import progressbar
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Conv1D, MaxPooling1D, Embedding, Conv2D, MaxPooling2D, Input, Bidirectional, concatenate
from keras.models import Sequential, Model, load_model
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Dropout, Reshape, Permute
from keras.layers.wrappers import TimeDistributed
from keras.optimizers import Adam
from keras import regularizers
import keras.backend as K
import io
import operator
# import progressbar
import numpy as np
import sys

sys.path.append('../')
from util import load_conll, eval_result, dev_and_test_comb, gen_data, pad_data, pad_label, pad_word_input, load_index, \
    data_to_seq
from config import config

# np.random.seed(7)

texts, labels = load_conll(config.train_path_transfer, config.labels_index)
val_texts, val_labels = load_conll(config.dev_path_transfer, config.labels_index)
# texts, labels = load_conll('keras_data/1.txt')
test_texts, test_labels = load_conll(config.test_path_transfer, config.labels_index)

print(config.train_path)
print(config.train_path_transfer)

# =====================
# build char cnn
# =====================
index_char = load_index(config.char_index)

MAX_WORD_LENGTH = config.word_length
wl = MAX_WORD_LENGTH

train_char, sl, wl = gen_data(texts, 0, wl, index_char)
val_char, sl, wl = gen_data(val_texts, sl, wl, index_char)
test_char, sl, wl = gen_data(test_texts, sl, wl, index_char)

MAX_SEQUENCE_LENGTH = sl + 1

if MAX_SEQUENCE_LENGTH % 2 == 1:
    MAX_SEQUENCE_LENGTH += 1
print(MAX_WORD_LENGTH)
print(MAX_SEQUENCE_LENGTH)

train_data_char = pad_data(train_char, MAX_SEQUENCE_LENGTH, MAX_WORD_LENGTH)
val_data_char = pad_data(val_char, MAX_SEQUENCE_LENGTH, MAX_WORD_LENGTH)
test_data_char = pad_data(test_char, MAX_SEQUENCE_LENGTH, MAX_WORD_LENGTH)

print(np.shape(train_char))

labels = pad_label(labels, MAX_SEQUENCE_LENGTH)
val_labels = pad_label(val_labels, MAX_SEQUENCE_LENGTH)
test_labels = pad_label(test_labels, MAX_SEQUENCE_LENGTH)

num_chars = len(index_char)

model_char = Sequential()
# embedding shape (None/word num in a sent, max_char_num_in_word, char_emb_dim)
model_char.add(
    Embedding(input_dim=num_chars, output_dim=config.dim_char_emb, input_length=MAX_SEQUENCE_LENGTH * MAX_WORD_LENGTH,
              trainable=config.train_char_embed, name='sor_charembed'))
model_char.add(Reshape((MAX_SEQUENCE_LENGTH, MAX_WORD_LENGTH, config.dim_char_emb), name='tar_char_reshape'))

model_char.add(Permute((3, 1, 2), name='tar_char_per'))
model_char.add(Conv2D(config.windows_size, (1, 2), padding='same', trainable=config.train_char_cnn, name='sor_conv'))
# 5 -> 125, 3 -> 75
model_char.add(Permute((2, 1, 3), name='tar_per_2'))
model_char.add(MaxPooling2D((2, 2), name='tar_char_maxpool'))
model_char.add(Reshape((MAX_SEQUENCE_LENGTH, config.dim_reshape), name='tar_char_reshape_2'))
# model_char.load_weights(config.char_source_weight, by_name=True)
model_char.summary()
char_input = Input(shape=(MAX_SEQUENCE_LENGTH * MAX_WORD_LENGTH,))
char_feature = model_char(char_input)

# ================
# end build char
# ================

# ================
# start build word
# ================

index_word = load_index(config.word_index)

train_data_word = data_to_seq(texts, index_word)
val_data_word = data_to_seq(val_texts, index_word)
test_data_word = data_to_seq(test_texts, index_word)

# pad word input
train_data_word = pad_word_input(train_data_word, MAX_SEQUENCE_LENGTH)
val_data_word = pad_word_input(val_data_word, MAX_SEQUENCE_LENGTH)
test_data_word = pad_word_input(test_data_word, MAX_SEQUENCE_LENGTH)

print(np.shape(train_data_word))

embeddings_index = {}
with io.open(config.word_embedding_path, 'r', encoding='utf8') as f:
    for line in f.readlines():
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

num_words = len(index_word)
embedding_matrix = np.zeros((num_words + 1, config.dim_word_emb))
for word, i in index_word.items():
    if i >= config.MAX_NB_WORDS:
        continue
    embedding_vector = embeddings_index.get(word)
    # if i == 1:
    # print(embedding_vector)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

model_word = Sequential()  # or Graph or whatever
# embedding shape = (None, word_in_sent, word_emb_dim)
model_word.add(
    Embedding(output_dim=config.dim_word_emb, input_dim=num_words + 1, mask_zero=False, weights=[embedding_matrix],
              trainable=config.train_word_embed, name='sor_emb'))
# model_word.load_weights(config.word_source_weight, by_name=True)
model_word.summary()

word_input = Input(shape=(MAX_SEQUENCE_LENGTH,))
word_feature = model_word(word_input)

# ===============
# end build word
# ===============

merged = concatenate([word_feature, char_feature], name='tar_concat')
# merged = word_feature
if config.bilstm:
    lstm_output = Bidirectional(
        LSTM(config.lstm_dim, return_sequences=True, trainable=config.train_lstm, name='sor_bilstm'))(merged)
else:
    lstm_output = LSTM(config.lstm_dim, return_sequences=True, name='sor_bilstm')(merged)

final_output = Dropout(0.5)(lstm_output)
prediction = TimeDistributed(Dense(config.index_dim, activation='softmax', activity_regularizer=regularizers.l1(0.01),
                                   trainable=config.train_softmax, name='sor_clsfier'))(final_output)
# train_data_word = np.array(train_data_word)
# train_char = np.array(train_char)
model = Model(inputs=[word_input, char_input], output=prediction)

# ada = Adam(lr=config.lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=config.decay)

model_old = load_model(config.source_model)
if config.load_char_cnn:
    char_weight = model_old.layers[2].get_weights()
    model.layers[2].set_weights(char_weight)
if config.load_word_embed:
    word_weight = model_old.layers[3].get_weights()
    model.layers[3].set_weights(word_weight)
if config.load_lstm:
    softmax_weight = model_old.layers[5].get_weights()
    model.layers[5].set_weights(softmax_weight)
if config.load_softmax:
    softmax_weight = model_old.layers[7].get_weights()
    model.layers[7].set_weights(softmax_weight)

model.summary()

# model.load_weights(config.source_weight, by_name=True)

model.compile(loss='categorical_crossentropy',
              optimizer='adam')

