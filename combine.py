from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Conv1D, MaxPooling1D, Embedding, Conv2D, MaxPooling2D, Input, Bidirectional, concatenate
from keras.models import Sequential, Model
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Dropout, Reshape, Permute
from keras.layers.wrappers import TimeDistributed
from keras.optimizers import Adam
from keras import regularizers
import keras.backend as K

import io
import operator
import progressbar
import numpy as np

from util import load_conll, eval_result, dev_and_test_comb, gen_data, pad_data, pad_label, pad_word_input, load_index, \
    data_to_seq
from config import config

# np.random.seed(7)

texts, labels = load_conll(config.train_path, config.labels_index)
val_texts, val_labels = load_conll(config.dev_path, config.labels_index)
# texts, labels = load_conll('keras_data/1.txt')
test_texts, test_labels = load_conll(config.test_path, config.labels_index)

# =====================
# build char cnn
# =====================
index_char = load_index(config.char_index)
# print(index_char)

MAX_WORD_LENGTH = config.word_length
wl = MAX_WORD_LENGTH

train_char, sl, wl = gen_data(texts, 0, wl, index_char)
val_char, sl, wl = gen_data(val_texts, sl, wl, index_char)
test_char, sl, wl = gen_data(test_texts, sl, wl, index_char)

MAX_SEQUENCE_LENGTH = sl

if MAX_SEQUENCE_LENGTH % 2 == 1:
    MAX_SEQUENCE_LENGTH += 1
print(MAX_WORD_LENGTH)
print(MAX_SEQUENCE_LENGTH)

train_data_char = pad_data(train_char, MAX_SEQUENCE_LENGTH, MAX_WORD_LENGTH)
val_data_char = pad_data(val_char, MAX_SEQUENCE_LENGTH, MAX_WORD_LENGTH)
test_data_char = pad_data(test_char, MAX_SEQUENCE_LENGTH, MAX_WORD_LENGTH)

# print(np.shape(train_char))

labels = pad_label(labels, MAX_SEQUENCE_LENGTH)
val_labels = pad_label(val_labels, MAX_SEQUENCE_LENGTH)
test_labels = pad_label(test_labels, MAX_SEQUENCE_LENGTH)

num_chars = len(index_char)

model_char = Sequential()
# embedding shape (None/word num in a sent, max_char_num_in_word, char_emb_dim)
model_char.add(
    Embedding(input_dim=num_chars, output_dim=config.dim_char_emb, input_length=MAX_SEQUENCE_LENGTH * MAX_WORD_LENGTH,
              name='sor_charembed'))
model_char.add(Reshape((MAX_SEQUENCE_LENGTH, MAX_WORD_LENGTH, config.dim_char_emb), name='sor_char_reshape'))

model_char.add(Permute((3, 1, 2), name='sor_char_per'))
##
# -----------model_char.add(Conv2D(config.windows_size, 1, 2, padding='same', name = 'sor_conv'))
# https://github.com/napsternxg/DeepSequenceClassification/blob/master/model.py
##
model_char.add(Conv2D(config.windows_size, (1, 2), padding='same', name='sor_conv'))
# 5 -> 125, 3 -> 75
model_char.add(Permute((2, 1, 3), name='sor_per_2'))
model_char.add(MaxPooling2D((2, 2), name='sor_char_maxpool'))
model_char.add(Reshape((MAX_SEQUENCE_LENGTH, config.dim_reshape), name='sor_char_reshape_2'))
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

# print(np.shape(train_data_word))

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
              name='sor_wordemb'))
model_word.summary()
word_input = Input(shape=(MAX_SEQUENCE_LENGTH,))
word_feature = model_word(word_input)

# ===============
# end build word
# ===============

merged = concatenate([word_feature, char_feature], name='sor_concat')

if config.bilstm:
    lstm_output = Bidirectional(LSTM(config.lstm_dim, return_sequences=True, name='sor_bilstm'))(merged)
else:
    lstm_output = LSTM(config.lstm_dim, return_sequences=True, name='sor_bilstm')(merged)

final_output = Dropout(0.5, name='sor_dropout')(lstm_output)
prediction = TimeDistributed(
    Dense(config.index_dim, activation='softmax', activity_regularizer=regularizers.l1(0.01), name='sor_clsfier'))(
    final_output)
# train_data_word = np.array(train_data_word)
# train_char = np.array(train_char)
model = Model(inputs=[word_input, char_input], output=prediction)

# adam = Adam(lr=config.lr, decay=config.decay)

model.summary()
model.compile(loss='categorical_crossentropy',
              optimizer='adam')

print(config.train_path)
print(config.dev_path)
print(config.test_path)

best_f1 = 0
no_improve = 0
result = open(config.log, 'w')

for i in range(config.n_epochs):
    print("Epoch {}".format(i))
    print("Training")

    train_pred_label = []
    avgLoss = 0
    bar = progressbar.ProgressBar(max_value=len(train_data_char))

    for n_batch, sent in bar(enumerate(train_data_char)):
        try:
            # print(n_batch)
            label = labels[n_batch]
            # print(label)
            label = np.array(label)
            label = label[np.newaxis, :]
            # print(label)
            label = np.eye(config.index_dim)[label]
            # print(label)
            # print(sent)
            char_input = np.array([sent])
            word_input = np.array([train_data_word[n_batch]])
            # print(sent)
            # print(word_input.shape)
            # print(char_input.shape)
            # if sent.shape[1] > 1:

            loss = model.train_on_batch([word_input, char_input], label)
            # for l in loss:
            #    print(l)
            avgLoss += loss
            # print(len(loss))
            # avgLoss = avgLoss/len(loss)
            # the reason why get two return of loss is accurcary
            pred = model.predict_on_batch([word_input, char_input])
            pred = np.argmax(pred, -1)[0]
            train_pred_label.append(pred)
        except:print("error")
        # print(i)
        # count += 1
    # print('lr' + str(float(K.get_value(model.optimizer.lr))) + ' decay: ' + str(float(K.get_value(model.optimizer.decay))))
    avgLoss = avgLoss / len(train_data_char)
    print(avgLoss)
    # print(train_pred_label)

    predword_train = [list(map(lambda x: config.l_in[x], y)) for y in train_pred_label]
    # print(predword_train)

    eval_result(labels, predword_train, config.labels_index)

    print("Validating")
    acc, p, r, f1 = dev_and_test_comb(val_data_word, val_data_char, val_labels, model, config, test=False)
    result.write('train lr' + str(float(K.get_value(model.optimizer.lr))) + ' decay: ' + str(
        float(K.get_value(model.optimizer.decay))) + "\nValidating " + ' acc: ' + str(acc) + ' p: ' + str(
        p) + ' r: ' + str(r) + ' f1: ' + str(f1) + '\n')
    if f1 >= best_f1:
        best_f1 = f1
        no_improve = 0
        model.save_weights(config.model_weight_path)
        result.write('new best score\n')
        model.save(config.model_path)
        print("new best score!")
    else:
        no_improve += 1
        print("no_improve: " + str(no_improve))
        if no_improve >= config.train_limit:
            print("early stopping")
            result.write("early stopping\n")
            break

    print("Testing")
    acc, p, r, f1 = dev_and_test_comb(test_data_word, test_data_char, test_labels, model, config, test=True)
    result.write("Testing " + ' acc: ' + str(acc) + ' p: ' + str(p) + ' r: ' + str(r) + ' f1: ' + str(f1) + '\n')
    result.write('\n')
