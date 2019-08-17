from keras.preprocessing.text import Tokenizer
from util import save_index, load_conll
from config import config

mpl_texts, labels = load_conll('data_0120/60_mpl_train.txt',config.labels_index)
mpl_val_texts, val_labels = load_conll('data_0120/20_mpl_dev.txt',config.labels_index)
#texts, labels = load_conll('data_0120/1.txt')
mpl_test_texts, test_labels = load_conll('data_0120/20_mpl_test.txt',config.labels_index)

react_texts, labels = load_conll('data_0120/60_react_train.txt',config.labels_index)
react_val_texts, val_labels = load_conll('data_0120/20_react_test.txt',config.labels_index)
#texts, labels = load_conll('data_0120/1.txt')
react_test_texts, test_labels = load_conll('data_0120/20_react_dev.txt',config.labels_index)

opengl_texts, labels = load_conll('data_0120/60_opengl_train.txt',config.labels_index)
opengl_val_texts, val_labels = load_conll('data_0120/20_opengl_dev.txt',config.labels_index)
#texts, labels = load_conll('data_0120/1.txt')
opengl_test_texts, test_labels = load_conll('data_0120/20_opengl_test.txt',config.labels_index)

jdbc_texts, labels = load_conll('data_0120/60_jdbc_train.txt',config.labels_index)
jdbc_val_texts, val_labels = load_conll('data_0120/20_jdbc_dev.txt',config.labels_index)
#texts, labels = load_conll('data_0120/1.txt')
jdbc_test_texts, test_labels = load_conll('data_0120/20_jdbc_test.txt',config.labels_index)

pd_texts, labels = load_conll('data_0120/60_pd_train.txt',config.labels_index)
pd_val_texts, val_labels = load_conll('data_0120/20_pd_dev.txt',config.labels_index)
#texts, labels = load_conll('data_0120/1.txt')
pd_test_texts, test_labels = load_conll('data_0120/20_pd_test.txt',config.labels_index)

np_texts, labels = load_conll('data_0120/60_np_train.txt',config.labels_index)
np_val_texts, val_labels = load_conll('data_0120/20_np_dev.txt',config.labels_index)
#texts, labels = load_conll('data_0120/1.txt')
np_test_texts, test_labels = load_conll('data_0120/20_np_test.txt',config.labels_index)
'''
np_texts, labels = load_conll('data_0120/np_0921/60_np_train.txt')
np_val_texts, val_labels = load_conll('data_0120/np_0921/20_np_dev.txt')
#texts, labels = load_conll('data_0120/1.txt')
np_test_texts, test_labels = load_conll('data_0120/np_0921/20_np_test.txt')
'''
texts_all = mpl_texts + mpl_val_texts + mpl_test_texts + pd_texts + pd_val_texts + pd_test_texts + np_texts + np_val_texts + np_test_texts + react_test_texts \
    +react_texts+react_val_texts+jdbc_test_texts+jdbc_texts+jdbc_val_texts+opengl_test_texts+opengl_val_texts+opengl_texts

tokenizer = Tokenizer(num_words=config.MAX_NB_WORDS, split=" ", lower=False, char_level=False, filters='')
tokenizer.fit_on_texts(texts_all)
index = tokenizer.word_index
save_index(index, 'word_index')
