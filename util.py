import progressbar
import numpy as np
#from config import config
import pickle



def load_conll(input_path, labels_index):
    input_post = open(input_path, 'r')
    posts = []
    post = ""
    labels = []
    label = []
    count = 0
    f = False
    for line in input_post.readlines():
        if f:
            print(line)
            f = False
        line = line.split()
        if len(line) == 2:
            post += line[0] + " "
            label.append( labels_index[line[1]] )
            #print(line[0]+ ' ' +line[1])
        elif len(line) == 0:
            posts.append(post)
            labels.append(label)
            label = []
            post = ""
            #print(count)
        else:
            print("no" + str(count))
            return
        
        count += 1
    input_post.close()
    return posts, labels

def eval_result(corr, pred, labels_index):
    total_corr_api = 0.0
    total_pred_api = 0.0
    right_pred = 0.0
    if len(corr) != len(pred):
        print("che error")
        return
    for i in range(0, len(corr)):
        c = corr[i]
        p = pred[i]
        
        #print(p)
        for j in range(0, len(c)):
            if c[j] == labels_index[ "B-API"]:
                total_corr_api += 1
            if p[j] == labels_index[ "B-API"]:
                total_pred_api += 1
            if c[j] == p[j] and p[j] == labels_index[ "B-API"]:
                right_pred += 1
    if total_corr_api == 0:
        acc = 0
        r = 0
    else:
        ##acc ???
        acc = (right_pred / total_corr_api)
        r = right_pred / total_corr_api
    if total_pred_api == 0:
        p = 0
    else:
        p = right_pred / total_pred_api
    if r == 0:
        f1 = 0
    else:
        f1 = 2 * (p*r) / (p + r)
    print("corr: "+ str(total_corr_api) + " pred: "+str(total_pred_api) + " right_pred: "+str(right_pred)+" acc: " + str(acc*100) +" p: " + str(p*100) +" r: " + str(r*100)+ " f1: " + str(f1*100))
    return acc, p, r, f1

def dev_and_test_word(data, labels, model, conf, test = False):
    pred_label = []
    avgLoss = 0
    bar = progressbar.ProgressBar(max_value=len(data))
    if test:
        pred_out = open(conf.pred_test_out,'w')
        raw_data = open(conf.test_path, 'r')
    else:
        pred_out = open(conf.pred_dev_out,'w')
        raw_data = open(conf.dev_path, 'r')

    for n_batch, sent in bar(enumerate(data)):
        label = labels[n_batch]
        label = np.array(label)
        label = label[np.newaxis,:]
        label = np.eye(conf.index_dim)[label]
        sent = data[n_batch]
        sent = np.array([sent])
        loss = model.test_on_batch(sent, label)
        avgLoss += loss
        pred = model.predict_on_batch(sent)
        pred = np.argmax(pred,-1)[0]
        pred_label.append(pred)

    avgLoss = avgLoss/len(data)
    print(avgLoss)
    #print(train_pred_label)

    predword_dev = [ list(map(lambda x: conf.l_in[x], y)) for y in pred_label]
    #print(predword_train)
    count = 0
    for pred in predword_dev:
        label = labels[count]
        for pred_label in pred:
            raw = raw_data.readline()
            if "B-API" in raw or pred_label != 0:
                pred_out.write(str(pred_label) + " vs " + raw)
        raw_data.readline()
        pred_out.write("\n")
        count += 1
    acc, p, r, f1 = eval_result(labels, predword_dev, conf.labels_index)
    return acc, p, r, f1

def dev_and_test_char(data, labels, model, conf, test = False):
    pred_label = []
    avgLoss = 0
    bar = progressbar.ProgressBar(max_value=len(data))
    if test:
        pred_out = open(conf.pred_test_out,'w')
        pad_out = open(conf.pad_test_out,'w')
        raw_data = open(conf.test_path, 'r')
    else:
        pred_out = open(conf.pred_dev_out,'w')
        pad_out = open(conf.pad_dev_out,'w')  
        raw_data = open(conf.dev_path, 'r')

    for n_batch, sent in bar(enumerate(data)):
        label = labels[n_batch]
        label = np.array(label)
        label = label[np.newaxis,:]
        label = np.eye(conf.index_dim)[label]
        sent = data[n_batch]
        sent = np.array([sent])
        loss = model.test_on_batch(sent, label)
        avgLoss += loss
        pred = model.predict_on_batch(sent)
        pred = np.argmax(pred,-1)[0]
        pred_label.append(pred)

    avgLoss = avgLoss/len(data)
    print(avgLoss)
    #print(train_pred_label)

    predword_dev = [ list(map(lambda x: conf.l_in[x], y)) for y in pred_label]
    #print(predword_train)
    count = 0
    
    for label in labels:
        pred = predword_dev[count]
        if (len(label) != len(pred)):
            print("label error")
            return
        bound = label.index(0)
        for i in range(0, bound):
            #if label[i] == labels_index[ "B-API"] or pred[i] == labels_index[ "B-API"]:
            raw_label = raw_data.readline()
            pred_out.write(str(pred[i]) + " vs " + str(label[i])+str(raw_label) + '\n')
        raw_data.readline()
        for i in range(bound, len(label)):
            pad_out.write(str(pred[i]) + "\n")
        pred_out.write("\n")
        pad_out.write("\n")
        count += 1
    acc, p, r, f1 = eval_result(labels, predword_dev, conf.labels_index)
    return acc, p, r, f1

def dev_and_test_comb(word_data, char_data, labels, model, conf, test = False):
    pred_label = []
    avgLoss = 0
    bar = progressbar.ProgressBar(max_value=len(word_data))
    if test:
        pred_out = open(conf.pred_test_out,'w')
        pad_out = open(conf.pad_test_out,'w')
        raw_data = open(conf.test_path, 'r')
    else:
        pred_out = open(conf.pred_dev_out,'w')
        pad_out = open(conf.pad_dev_out,'w')  
        raw_data = open(conf.dev_path, 'r')

    for n_batch, sent in bar(enumerate(word_data)):
        label = labels[n_batch]
        label = np.array(label)
        label = label[np.newaxis,:]
        label = np.eye(3)[label]
        
        word_input = np.array([word_data[n_batch]])
        char_input = np.array([char_data[n_batch]])
        loss = model.test_on_batch([word_input, char_input], label)
        avgLoss += loss
        pred = model.predict_on_batch([word_input, char_input])
        pred = np.argmax(pred,-1)[0]
        pred_label.append(pred)

    avgLoss = avgLoss/len(word_data)
    print(avgLoss)
    #print(train_pred_label)

    predword_dev = [ list(map(lambda x: conf.l_in[x], y)) for y in pred_label]
    #print(predword_train)
    count = 0
    for pred in predword_dev:
        label = labels[count]
        for pred_label in pred:
            raw = raw_data.readline()
            if "B-API" in raw or pred_label != 0:
                pred_out.write(str(pred_label) + " vs " + raw)
        raw_data.readline()
        pred_out.write("\n")
        count += 1
    acc, p, r, f1 = eval_result(labels, predword_dev, conf.labels_index)
    return acc, p, r, f1

def gen_data(texts,sequence_length,word_length, index):
    post_set = []
    
    count = 0

    for sent in texts:
        sent_char = []
        sent = sent.split(" ")
        if len(sent) > sequence_length:
            sequence_length = len(sent)
        lenmax = 0
        for word in sent:
            if len(word) > 0:
                chars = []
                for char in word:
                    # if(char not in index): index[char]=len(index)+1
                    print(index[char])
                    chars.append(index[char])
                if len(chars) > word_length:
                    word_length = len(chars)
            #print(chars)
            sent_char.append(chars)
            #count
            count += 1
            #lenmax += 1
        
        #print(count)
        post_set.append(sent_char)
    return post_set,sequence_length,word_length

def pad_data(post_set,sequence_length,word_length):

    post_pad = []
    for sent in post_set:
        word_pad = []
        for word in sent:
            temp = [0]*word_length
            
            pos = int((word_length - len(word))/2)
            #print(pos)
            for i in range(0, len(word)):
                temp[i+pos] = word[i]
            #print(temp)
            #print(len(temp))
            if(len(temp)!= word_length):
                print("no")
                return
            word_pad += temp
            #print(word_pad)
        for i in range(0, sequence_length*word_length - len(word_pad)):
            word_pad.append(0)
        post_pad.append(word_pad)
    return post_pad

def pad_label(labels, sequence_length):
    for label in labels:
        for i in range(0, sequence_length-len(label)):
            label.append(0)
    return labels
    #print(np.shape(label))

def pad_word_input(posts, sequence_length):
    for post in posts:
        for i in range(0, sequence_length-len(post)):
            post.append(0)
    return posts

def data_to_seq(texts, word_index):
    posts = []
    for text in texts:
        temp = text.split(' ')[:-1]
        posts.append(temp)   
    #print(word_index)
    sequences = []
    for post in posts:
        seq = []
        for word in post:
            if word in word_index:
                seq.append(word_index[word])
            else:
                print('error'+word)
                break
        sequences.append(seq)
    return sequences

def save_index(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_index(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f,encoding='bytes')
