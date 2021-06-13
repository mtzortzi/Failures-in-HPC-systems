#!/home/users/nikela/tensorflow-env/bin/python
import os
import sys
#CPU


#GPU
#os.environ['CUDA_VISIBLE_DEVICES'] ="0"

import keras
from keras.models import Model, Sequential
from keras.layers import Lambda, Dense, Flatten, LSTM, GlobalMaxPooling1D, Input, Merge
from keras.layers.merge import Concatenate
from keras.layers.embeddings import Embedding
from keras.layers.core import Activation, Dropout, Dense, Reshape
from keras.preprocessing import sequence, text
from keras.preprocessing.text import one_hot, Tokenizer
from keras.preprocessing.sequence import pad_sequences, skipgrams
from keras.optimizers import Adam, RMSprop, SGD
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard,  ReduceLROnPlateau
from keras.models import load_model

import keras.utils as ku

from sklearn.utils import class_weight

import pandas as pd
import csv
import pickle
import numpy as np
import io
import fileinput
import random
import math
import time

import faulthandler
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import fileinput
import io
from tqdm import tqdm
import re
from keras.models import model_from_json
import tensorflow as tf

from tensorflow.python.client import device_lib
from metrics import *


##Custom
##from skipgram import vocab_size, wids, id2word, word2id, word_embeddings
import skipgram
#import tensorflow as tf
#sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
#:print('sess', sess)


##faulthandler.enable(file=sys.stderr, all_threads=True)

import skipgram
#import tensorflow as tf
#sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
#:print('sess', sess)


##faulthandler.enable(file=sys.stderr, all_threads=True)
import phase1_exp5
import multiwords_train30_test70

def main():

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    device_lib.list_local_devices()


    print('....read wids -- to_supervised........')
    print('....I had already reorder phase1_nodewise_wids.csv by descending order....')
    phase1_nodewise_wids = pd.read_csv('/various/mtzortzi/from_nikela/ASCENDING_ORDER_METHOD/DESCENDING_ORDER_trains/phase1_nodewise_wids.csv')
    wids = phase1_nodewise_wids["WIDS"]
    wids = list(wids)
    n_sequences = phase1_exp5.series_to_supervised(wids, 15, 3)
    print(n_sequences.head(10))
    values = n_sequences.values
    print('DONE')

    print('....read embedding.....')
    print('.....I have loaded the word_embeddings from ASCENDING_ORDER_METHOD, the only difference with the previous desc is that I just kept some more phrases, this is the reason I run again here descending order....')
    word_embeddings = pd.read_csv('/various/mtzortzi/from_nikela/ASCENDING_ORDER_METHOD/withoutNAN_word_embeddings_phase1.csv')
    print(word_embeddings.head(10))
    print('DONE')

    print('....getting vocabulary sizes.....')
    merged_static = multiwords_train30_test70.create_merged('/various/mtzortzi/LogAider-master/RAS-JOB/merged_static.csv')
    corpus_all, vocab_size_all, wids_all, id2word_all, word2id_all, tokenizer_all = multiwords_train30_test70.build_corpus(merged_static['MULTIWORDS'])
    vocab_size = skipgram.compute_vocab_size_all(word2id_all)
    print('DONE')
    embed_size = 100


    # split into input and outputs
    print('.....splitting into input/output.....')
    n_features = 1 #only wids
    n_obs = 15 * n_features
    train_inputs = values[:, 0:n_obs]
    train_targets = values[:, 15:]
    print('DONE')
    print('train_inputs shape', train_inputs.shape)
    print(train_inputs[0:20])
    print('train target shape', train_targets.shape)
    print(train_targets[0:20])

    print('....train targets to categorical....')
    train_targets = ku.to_categorical(train_targets, num_classes=vocab_size_all)
    print('DONE')
    print('train_targets.shape', train_targets.shape)
    print('train_inputs.shape', train_inputs.shape)



    print('............load almost trained model from phase1..............')
    #dependencies = {'jaccard_coef': jaccard_coef, 'jaccard_coef_int': jaccard_coef_int}
    loaded_model = load_model('/various/mtzortzi/from_nikela/ASCENDING_ORDER_METHOD/DESCENDING_ORDER_trains/phase1_exp5/models/model.3696-0.6572.hdf5', custom_objects={'jaccard_coef': jaccard_coef, 'jaccard_coef_int': jaccard_coef_int})#dependencies)
    history = pd.read_csv('/various/mtzortzi/from_nikela/ASCENDING_ORDER_METHOD/DESCENDING_ORDER_trains/phase1_exp5/training.log', sep=',', engine='python')
    print('DONE')


    rnn_size = 500  # size of RNN
    num_epochs = 3700 # number of epochs
    batch_size = 1000
    lr_base = 0.01 * (float(batch_size) / 16)
    print('split the data into train and validation set')
    # Split the data
    #train_inputs_new, inputs_valid, train_targets_new, targets_valid = train_test_split(train_inputs,train_targets, test_size=0.3, shuffle= True)
    #print('DONE')
    #print('....training lstm model.....')
    #model, history = lstm_model(train_inputs_new, inputs_valid, train_targets_new, targets_valid, rnn_size, vocab_size, word_embeddings, embed_size, train_inputs, train_targets, lr_base, num_epochs, batch_size, vocab_size_all)
    #print('DONE')


    optimizer = SGD(lr=lr_base, momentum=0.9) 
    #evaluate loaded model on test data
    loaded_model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['categorical_accuracy', jaccard_coef, jaccard_coef_int])

    
    print('......evaluate loaded_model........')
    score = loaded_model.evaluate(train_inputs, train_targets, verbose=1)
    print("%s: %.2f%%" % (loaded_model.loss, score[0]*100))
    print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
    print("%s: %.2f%%" % (loaded_model.metrics_names[2], score[2]*100))
    print("%s: %.2f%%" % (loaded_model.metrics_names[3], score[3]*100))


    predicted = loaded_model.predict(train_inputs) #possibilities
    print('..............actual-real classes..................')
    print(np.argmax(train_targets, axis=-1)) # actual classes
    clas1 = loaded_model.predict_classes(train_inputs) #predicted classes
    print('.............predicted classes.....................')
    print(clas1)
    # I take the possibilities, I create the predicted classes & I make them categorical so as to fit with categorical train_targets
    pred_max = ku.to_categorical(np.argmax(predicted, axis=-1), num_classes = vocab_size_all)
    print('-------pred_max-------')
    print(pred_max)
    print(pred_max.shape)
    print(np.square(np.subtract(train_targets, pred_max)).mean())
    print('DONE')

    print('.......create a new column AN_WIDS using the predicted wids from phase1....')
    new_df = phase1_nodewise_wids.iloc[15:] # 15 is the History Size in phase1
    alist = [l for sublist in clas1 for l in sublist]
    # print(alist[0:5]) 
    pred_wids = alist[0:3]
    for n, l in enumerate(alist[3:]):
        if (n%3==2):
    #         print(n, l)
            pred_wids.append(l)
    new_df['AN_wids'] = pred_wids
    print('new_df', new_df.shape)
    print(new_df.head(2))
    print('DONE')

    print('.........match the predicted wids with its multiwords........')
    msgs=[]
    for wid in new_df['AN_wids']:
        msgs.append(id2word_all[wid])
    new_df['AN_multiwords'] = msgs
    print(new_df.head(2))
    print('DONE')

    new_df.to_csv('/various/mtzortzi/from_nikela/ASCENDING_ORDER_METHOD/DESCENDING_ORDER_trains/anwesha_df.csv', index=False, sep=',', quoting=csv.QUOTE_ALL)


if __name__=="__main__":
        main()
