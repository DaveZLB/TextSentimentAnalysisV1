#!/usr/bin/env python3
# coding: utf-8
# File: lstm_sentiment.py
# Author: lhy<lhy_in_blcu@126.com,https://huangyong.github.io>
# Date: 18-3-19
import gensim
import jieba as jieba
import numpy as np
from keras.layers import Dropout
from keras.models import load_model
import keras.backend as K
import keras_metrics as km

recall = km.categorical_recall(label=0)
precision = km.categorical_precision(label=0)
f1 = km.categorical_f1_score(label = 0)

def cal_recall(y_true, y_pred):
    """Recall metric.
    Only computes a batch-wise average of recall.
    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def cal_precision(y_true, y_pred):
    """Precision metric.
    Only computes a batch-wise average of precision.
    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def cal_f1(y_true, y_pred):
    precision = cal_precision(y_true, y_pred)
    recall = cal_recall(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))

def plot_acc_and_loss(history):
    from matplotlib import pyplot as plt
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)

    plt.plot(epochs, acc, 'go-.', label='Training acc')
    plt.plot(epochs, val_acc, 'g', label='Validation acc')
    plt.plot(epochs, loss, 'ro-.', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Accuracy and Loss')
    plt.legend()

    plt.show()

def plot_precision_recall_f1(history):
    from matplotlib import pyplot as plt
    acc = history.history['acc']
    precision = history.history['precision']
    recall = history.history['recall']
    f1 = history.history['f1_score']
    epochs = range(1, len(acc) + 1)
    plt.plot(epochs, precision, 'go-.', label='precision')
    plt.plot(epochs, recall, 'yo-.', label='recall')
    plt.plot(epochs, f1, 'bo-.', label='f1')
    plt.title('Training Performance')
    plt.legend()
    plt.show()

VECTOR_DIR = './embedding/word_vector.bin'  # 词向量模型文件
model = gensim.models.KeyedVectors.load_word2vec_format(VECTOR_DIR, binary=False)

'''基于wordvector，通过lookup table的方式找到句子的wordvector的表示'''
def rep_sentencevector(sentence):
    bd = '[’!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]+，。！？“”《》：、． '
    for i in bd:
        sentence = sentence.replace(i, '')  # 字符串替换去标点符号
    word_list=jieba.lcut(sentence)
   # word_list = [word for word in sentence.split(' ')]
    max_words = 100
    embedding_dim = 200
    embedding_matrix = np.zeros((max_words, embedding_dim))
    for index, word in enumerate(word_list):
        try:
            embedding_matrix[index] = model[word]
        except:
            pass

    return embedding_matrix

'''构造训练数据'''
def build_traindata():
    X_train = list()
    Y_train = list()
    X_test = list()
    Y_test = list()

    for line in open('./data/train.txt'):
        line = line.strip().strip().split('\t')
        sent_vector = rep_sentencevector(line[-1])

        X_train.append(sent_vector)
        if line[0] == '1':
            Y_train.append([0, 1])
        else:
            Y_train.append([1, 0])

    for line in open('./data/test.txt'):
        line = line.strip().strip().split('\t')
        sent_vector = rep_sentencevector(line[-1])
        X_test.append(sent_vector)
        if line[0] == '1':
            Y_test.append([0, 1])
        else:
            Y_test.append([1, 0])

    return np.array(X_train), np.array(Y_train), np.array(X_test), np.array(Y_test),

'''三层lstm进行训练，迭代20次'''
def train_lstm(X_train, Y_train, X_test, Y_test):
    from keras.models import Sequential
    from keras.layers import LSTM, Dense

    import numpy as np
    data_dim = 200  # 对应词向量维度
    timesteps = 100  # 对应序列长度
    # expected input data shape: (batch_size, timesteps, data_dim)
    model = Sequential()
    model.add(LSTM(32, return_sequences=True,
                   input_shape=(timesteps, data_dim)))  # returns a sequence of vectors of dimension 32
    model.add(LSTM(32, return_sequences=True))  # returns a sequence of vectors of dimension 32
    model.add(LSTM(32))  # return a single vector of dimension 32
    model.add(Dense(2, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=["acc",recall,precision,f1])
    history = model.fit(X_train, Y_train, batch_size=100, epochs=20, validation_data=(X_test, Y_test))

    plot_acc_and_loss(history)
    plot_precision_recall_f1(history)
    model.save('./model/sentiment_lstm_model.h5')
    model.summary()
    rets = model.evaluate(X_test, Y_test)
    print(rets)
    '''/
    1 [==============================] - 41s 2ms/step - loss: 0.5384 - acc: 0.7142 - val_loss: 0.4223 - val_acc: 0.8281
    5 [==============================] - 38s 2ms/step - loss: 0.2885 - acc: 0.8904 - val_loss: 0.3618 - val_acc: 0.8531
    10 [==============================] - 40s 2ms/step - loss: 0.1965 - acc: 0.9357 - val_loss: 0.3815 - val_acc: 0.8515
    15 [==============================] - 39s 2ms/step - loss: 0.1420 - acc: 0.9577 - val_loss: 0.5172 - val_acc: 0.8501
    20 [==============================] - 37s 2ms/step - loss: 0.1055 - acc: 0.9729 - val_loss: 0.5309 - val_acc: 0.8505
    '''

'''实际应用，测试'''
def predict_lstm(model_filepath):
    model = load_model(model_filepath)
    sentence = '这个 电视 真 尼玛 垃圾 ， 老子 再也 不买 了'#[[0.01477097 0.98522896]]
    #sentence = '这件 衣服 真的 太 好看 了 ！ 好想 买 啊 '#[[0.9843225  0.01567744]]
    sentence_vector = np.array([rep_sentencevector(sentence)])
    print(sentence_vector)
    print(sentence_vector.shape)
    print('test after load: ', model.predict(sentence_vector))
    # result = model.predict(sentence_vector)
    # print('lstm model accuray is :{0}'.format(result[0][0]))

def lstm_train():
    X_train, Y_train, X_test, Y_test = build_traindata()
    print(X_train.shape, Y_train.shape)
    print(X_test.shape, Y_test.shape)
    train_lstm(X_train, Y_train, X_test, Y_test)
def lstm_test():
    model_filepath = './model/sentiment_lstm_model1.h5'
    X_train, Y_train, X_test, Y_test = build_traindata()
    model = load_model(model_filepath)
    rets = model.evaluate(X_test, Y_test,verbose = 0)
    print('lstm model accuray is :{0}'.format(rets[1]))

def lstm_predict():
    model_filepath = './model/sentiment_lstm_model1.h5'
    model = load_model(model_filepath)
    sentence_list = ['牛X的手机，从3米高的地方摔下去都没坏，质量非常好'
        ,'酒店的环境非常好，价格也便宜，值得推荐'
        ,'质量不错，是正品 ，安装师傅也很好，才要了83元材料费'
        ,'东西非常不错，安装师傅很负责人，装的也很漂亮，精致，谢谢安装师傅！'
        ,'手机质量太差了，傻X店家，赚黑心钱，以后再也不会买了'
        ,'屏幕较差，拍照也很粗糙。']
    for sentence in sentence_list:
        sentence_vector = np.array([rep_sentencevector(sentence)])
        result = model.predict(sentence_vector)
        print(sentence)
        print(result)

def lstm_predict1():
    model_filepath = './model/sentiment_lstm_model1.h5'
    model = load_model(model_filepath)
    sentence_list = ['这个电影太差劲了！']
    for sentence in sentence_list:
        sentence_vector = np.array([rep_sentencevector(sentence)])
        result = model.predict(sentence_vector)
        print(sentence)
        print(result)

def lstm_precision_recall():
    from sklearn.metrics import classification_report
    model_filepath = './model/sentiment_lstm_model1.h5'
    model = load_model(model_filepath)
    X_train, Y_train, X_test, Y_test = build_traindata()
    y_pred = model.predict(X_test, batch_size=100, verbose=1)
    y_pred_bool = np.argmax(y_pred, axis=1)
    y_test_bool = np.argmax(Y_test, axis=1)
    target_names = ['Negative', 'Positive']
    print(classification_report(y_test_bool, y_pred_bool))

#test
if __name__ == '__main__':
    lstm_train()

