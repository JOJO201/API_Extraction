import os
import sys

sys.path.append('../')


class config():
    # for char only and char+word
    labels_index = {"O": 1, "B-API": 2}
    l_in = [0, 1, 2]
    index_dim = 3

    # for only word train
    '''
    labels_index = {"O":0, "B-API":1}
    l_in = [0,1]
    index_dim = 2
    '''
    # ==============

    MAX_NB_WORDS = 10000
    word_length = 37
    # EMBEDDING_DIM = 200
    # n_classes = 2
    n_epochs = 30
    lr = 0.05
    # decay = 0.9
    train_limit = 5

    dim_word_emb = 200
    '''
    windows_size = 3
    dim_char_emb = 100
    dim_reshape = 75
    '''
    '''
    windows_size = 5
    dim_char_emb = 100
    dim_reshape = 125
    '''

    windows_size = 3
    dim_char_emb = 40
    dim_reshape = 30
    '''
    windows_size = 3
    dim_char_emb = 20
    dim_reshape = 15
    '''
    bilstm = True
    lstm_dim = 50

    transfer = False
    load_char_embed = True
    load_char_cnn = True
    load_lstm = True
    load_softmax = True
    load_word_embed = True

    train_char_embed = True
    train_char_cnn = True
    train_word_embed = True
    train_lstm = True
    train_softmax = True

    data_root = ''

    corpus = 'opengl'

    if corpus == 'pd':
        train_path = data_root + 'data_0120/60_pd_train.txt'
        # train_path = 'data_0120/pd_train_1-2.txt'
        # train_path = 'data_0120/pd_train_1-4.txt'
        # train_path = 'data_0120/pd_train_1-8.txt'
        # train_path = 'data_0120/pd_train_1-16.txt'
        # train_path = 'data_0120/pd_train_1-32.txt'
        dev_path = data_root + 'data_0120/20_pd_dev.txt'
        test_path = data_root + 'data_0120/20_pd_test.txt'
    elif corpus == 'opengl':
        # train_path = data_root + 'data_0120/60_opengl_train.txt'
        # train_path = 'data_0120/pd_train_1-2.txt'
        # train_path = 'data_0120/opengl_train_1-4.txt'
        train_path = 'data_0120/opengl_train_1-8.txt'
        # train_path = 'data_0120/pd_train_1-16.txt'
        # train_path = 'data_0120/opengl_train_1-32.txt'
        dev_path = data_root + 'data_0120/20_opengl_dev.txt'
        test_path = data_root + 'data_0120/20_opengl_test.txt'
    elif corpus == 'jdbc':
        # train_path = data_root + 'data_0120/60_jdbc_train.txt'
        # train_path = 'data_0120/jdbc_train_1-2.txt'
        # train_path = 'data_0120/jdbc_train_1-4.txt'
        train_path = 'data_0120/jdbc_train_1-8.txt'
        # train_path = 'data_0120/jdbc_train_1-16.txt'
        # train_path = 'data_0120/jdbc_train_1-32.txt'
        dev_path = data_root + 'data_0120/20_jdbc_dev.txt'
        test_path = data_root + 'data_0120/20_jdbc_test.txt'
    elif corpus == 'react':
        train_path = data_root + 'data_0120/60_react_train.txt'
        # train_path = 'data_0120/pd_train_1-2.txt'
        # train_path = 'data_0120/pd_train_1-4.txt'
        # train_path = 'data_0120/pd_train_1-8.txt'
        # train_path = 'data_0120/pd_train_1-16.txt'
        # train_path = 'data_0120/pd_train_1-32.txt'
        dev_path = data_root + 'data_0120/20_react_dev.txt'
        test_path = data_root + 'data_0120/20_react_test.txt'
    elif corpus == 'mpl':
        train_path = data_root + 'data_0120/60_mpl_train.txt'
        # train_path = 'data_0120/mpl_train_1-2.txt'
        # train_path = 'data_0120/mpl_train_1-4.txt'
        # train_path = 'data_0120/mpl_train_1-8.txt'
        # train_path = 'data_0120/mpl_train_1-16.txt'
        # train_path = 'data_0120/mpl_train_1-32.txt'
        dev_path = data_root + 'data_0120/20_mpl_dev.txt'
        test_path = data_root + 'data_0120/20_mpl_test.txt'
        '''
        train_path = 'keras_data/2.txt'
        dev_path = 'keras_data/2.txt'
        test_path = 'keras_data/2.txt'
        '''
    elif corpus == 'np':
        train_path = data_root + 'data_0120/60_np_train.txt'
        # train_path = 'data_0120/np_train_1-2.txt'
        # train_path = 'data_0120/np_train_1-4.txt'
        # train_path = 'data_0120/np_train_1-8.txt'
        # train_path = 'data_0120/np_train_1-16.txt'
        dev_path = data_root + 'data_0120/20_np_dev.txt'
        test_path = data_root + 'data_0120/20_np_test.txt'

    elif corpus == 'jfc':
        train_path = data_root + 'data_0120/60_jfc_train.txt'
        # train_path = 'data_0120/np_train_1-2.txt'
        # train_path = 'data_0120/np_train_1-4.txt'
        # train_path = 'data_0120/np_train_1-8.txt'
        # train_path = 'data_0120/np_train_1-16.txt'
        dev_path = data_root + 'data_0120/20_jfc_dev.txt'
        test_path = data_root + 'data_0120/20_jfc_test.txt'

    corpus_transfer = 'opengl'
    if corpus_transfer == 'opengl':
        train_path_transfer = data_root + 'data_0120/60_opengl_train.txt'
        # train_path_transfer = 'data_0120/pd_train_1-2.txt'
        # train_path_transfer = 'data_0120/pd_train_1-4.txt'
        # train_path_transfer = 'data_0120/pd_train_1-8.txt'
        # train_path_transfer = 'data_0120/pd_train_1-16.txt'
        # train_path_transfer = 'data_0120/pd_train_1-32.txt'

        dev_path_transfer = data_root + 'data_0120/20_opengl_dev.txt'
        test_path_transfer = data_root + 'data_0120/20_opengl_test.txt'
    if corpus_transfer == 'jdbc':
        train_path_transfer = data_root + 'data_0120/60_jdbc_train.txt'
        # train_path_transfer = 'data_0120/pd_train_1-2.txt'
        # train_path_transfer = 'data_0120/pd_train_1-4.txt'
        # train_path_transfer = 'data_0120/pd_train_1-8.txt'
        # train_path_transfer = 'data_0120/pd_train_1-16.txt'
        # train_path_transfer = 'data_0120/pd_train_1-32.txt'

        dev_path_transfer = data_root + 'data_0120/20_jdbc_dev.txt'
        test_path_transfer = data_root + 'data_0120/20_jdbc_test.txt'
    if corpus_transfer == 'react':
        train_path_transfer = data_root + 'data_0120/60_react_train.txt'
        # train_path_transfer = 'data_0120/pd_train_1-2.txt'
        # train_path_transfer = 'data_0120/pd_train_1-4.txt'
        # train_path_transfer = 'data_0120/pd_train_1-8.txt'
        # train_path_transfer = 'data_0120/pd_train_1-16.txt'
        # train_path_transfer = 'data_0120/pd_train_1-32.txt'

        dev_path_transfer = data_root + 'data_0120/20_react_dev.txt'
        test_path_transfer = data_root + 'data_0120/20_react_test.txt'

    if corpus_transfer == 'pd':
        train_path_transfer = data_root + 'data_0120/60_pd_train.txt'
        # train_path_transfer = 'data_0120/pd_train_1-2.txt'
        # train_path_transfer = 'data_0120/pd_train_1-4.txt'
        # train_path_transfer = 'data_0120/pd_train_1-8.txt'
        # train_path_transfer = 'data_0120/pd_train_1-16.txt'
        # train_path_transfer = 'data_0120/pd_train_1-32.txt'

        dev_path_transfer = data_root + 'data_0120/20_pd_dev.txt'
        test_path_transfer = data_root + 'data_0120/20_pd_test.txt'
    elif corpus_transfer == 'mpl':

        train_path_transfer = data_root + 'data_0120/60_mpl_train.txt'
        # train_path_transfer  = 'data_0120/mpl_train_1-2.txt'
        # train_path_transfer = 'data_0120/mpl_train_1-4.txt'
        # train_path_transfer = 'data_0120/mpl_train_1-8.txt'
        # train_path_transfer = 'data_0120/mpl_train_1-16.txt'
        dev_path_transfer = data_root + 'data_0120/20_mpl_dev.txt'
        test_path_transfer = data_root + 'data_0120/20_mpl_test.txt'
        '''
        train_path = 'keras_data/2.txt'
        dev_path = 'keras_data/2.txt'
        test_path = 'keras_data/2.txt'
        '''
    elif corpus_transfer == 'np':
        train_path_transfer = data_root + 'data_0120/60_np_train.txt'
        # train_path_transfer = 'data_0120/np_train_1-2.txt'
        # train_path_transfer = 'data_0120/np_train_1-4.txt'
        # train_path_transfer = 'data_0120/np_train_1-8.txt'
        dev_path_transfer = data_root + 'data_0120/20_np_dev.txt'
        test_path_transfer = data_root + 'data_0120/20_np_test.txt'

    elif corpus_transfer == 'jfc':
        train_path_transfer = data_root + 'data_0120/60_jfc_train.txt'
        # train_path_transfer = 'data_0120/np_train_1-2.txt'
        # train_path_transfer = 'data_0120/np_train_1-4.txt'
        # train_path_transfer = 'data_0120/np_train_1-8.txt'
        # train_path_transfer = 'data_0120/np_train_1-16.txt'
        dev_path_transfer = data_root + 'data_0120/20_jfc_dev.txt'
        test_path_transfer = data_root + 'data_0120/20_jfc_test.txt'

    # word_embedding_path = 'keras_data/vectors_mnp_0920.txt'
    word_embedding_path = data_root + 'data_0120/vectors_1013.txt'
    # word_embedding_path = 'data_0120/vectors_0120_dim_50.txt'
    char_index = data_root + 'char_index'
    word_index = data_root + 'word_index'

    result = 'result_' + corpus + '/'
    if not os.path.exists(result):
        os.makedirs(result)

    # out_path = result+'word_lstm_newemb_'+str(corpus)

    # base char
    # out_path_source = result+'char_cnn_bilstm_'+str(corpus)

    # base word
    # out_path_source = result+'word_50_bilstm_'+str(corpus)

    # base combine
    out_path_source = result + 'combine_' + str(corpus)

    # transfer
    # out_path_transfer = result+'4.2add_allload_transfer_char-cnn-word-bilstm_train-softmax'+corpus+'_'+corpus_transfer
    out_path_transfer = result + 'test_' + corpus + '_' + corpus_transfer

    if transfer:
        out_path = out_path_transfer
    else:
        out_path = out_path_source

    if not os.path.exists(out_path):
        os.makedirs(out_path)
    pred_test_out = out_path + '/pred_test.txt'
    pad_test_out = out_path + '/pad_test.txt'
    pred_dev_out = out_path + '/pred_dev.txt'
    pad_dev_out = out_path + '/pad_dev.txt'
    model_weight_path = out_path + '/model_weight.h5'
    model_path = out_path + '/model.h5'
    log = out_path + '/log.txt'

    # source model path
    source_weight = out_path_source + '/model_weight.h5'
    source_model = out_path_source + '/model.h5'

    char_source_weight = result + 'char_cnn_bilstm_' + str(corpus) + '/model_weight.h5'
    word_source_weight = result + 'word_bilstm_' + str(corpus) + '/model_weight.h5'
