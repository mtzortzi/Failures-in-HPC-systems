#!/home/users/nikela/tensorflow-env/bin/python
import os
import sys
#CPU


#GPU
#os.environ['CUDA_VISIBLE_DEVICES'] ="0"

import keras
from keras.models import Model, Sequential
from keras.models import load_model
from keras.layers import Lambda, Dense, Flatten, LSTM, GlobalMaxPooling1D, Input, Merge
from keras.layers import Conv1D, MaxPooling1D
from keras.layers.merge import Concatenate
from keras.layers.embeddings import Embedding
from keras.layers.core import Activation, Dropout, Dense, Reshape
from keras.preprocessing import sequence, text
from keras.preprocessing.text import one_hot, Tokenizer
from keras.preprocessing.sequence import pad_sequences, skipgrams
from keras.optimizers import Adam, RMSprop, SGD
from keras.regularizers import l2
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard,  ReduceLROnPlateau, CSVLogger
import keras.utils as ku 

from sklearn.model_selection import train_test_split
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

import multiwords_train30_test70
def main():

    ##form now on I need
    ##wids
    ##vocab_size_all
    ##vocab_size
    ##word_embeddings
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    device_lib.list_local_devices()

    print('....read wids -- to_supervised........')
    print('....I had already reorder phase1_nodewise_wids.csv by descending order....')
    phase1_nodewise_wids = pd.read_csv('/various/mtzortzi/from_nikela/ASCENDING_ORDER_METHOD/DESCENDING_ORDER_trains/phase1_nodewise_wids.csv')
    wids = phase1_nodewise_wids["WIDS"]
    wids = list(wids)
    n_sequences = series_to_supervised(wids, 15, 3)
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


    rnn_size = 500  # size of RNN
    num_epochs = 3700 # number of epochs
    batch_size = 1000 
    lr_base = 0.01 * (float(batch_size) / 16)
    print('split the data into train and validation set')
    # Split the data
    train_inputs_new, inputs_valid, train_targets_new, targets_valid = train_test_split(train_inputs,train_targets, test_size=0.3, shuffle= True)
    print('DONE')
    print('....training lstm model.....')    
    model, history = lstm_model(train_inputs_new, inputs_valid, train_targets_new, targets_valid, rnn_size, vocab_size, word_embeddings, embed_size, train_inputs, train_targets, lr_base, num_epochs, batch_size, vocab_size_all)
    print('DONE')



    #If you look at the last layer of your neural network you can see that we are setting the output to be equal to number
    #of classes which mean the model will give us the probability that the input is belong to a particular class. 
    #Hence to get the predicted we need to use argmax to find the one with highest probability 
    
    # evaluate the model
    loss, categorical_accuracy, jaccard_coef, jaccard_coef_int  = model.evaluate(train_inputs, train_targets, verbose=1)
    print('Categorical Accuracy: %f' % (categorical_accuracy*100))
    print('loss, jaccard_coef, jaccard_coef_int', loss, jaccard_coef, jaccard_coef_int)
    predicted = model.predict(train_inputs) #possibilities
    print('..............actual-real classes..................')
    print(np.argmax(train_targets, axis=-1)) # actual classes
    clas1 = model.predict_classes(train_inputs) #predicted classes
    print('.............predicted classes.....................')
    print(clas1)
    # I take the possibilities, I create the predicted classes & I make them categorical so as to fit with categorical train_targets
    pred_max = ku.to_categorical(np.argmax(predicted, axis=-1), num_classes = vocab_size_all)
    print('-------pred_max-------')
    print(pred_max)
    print(pred_max.shape)
    print(np.square(np.subtract(train_targets, pred_max)).mean())




def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    """
    Frame a time series as a supervised learning dataset.
    Arguments:
        data: Sequence of observations as a list or NumPy array.
        n_in: Number of lag observations as input (X).
        n_out: Number of observations as output (y).
        dropnan: Boolean whether or not to drop rows with NaN values.
    Returns:
        Pandas DataFrame of series framed for supervised learning.
    """
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


def lstm_model(train_inputs_new, inputs_valid, train_targets_new, targets_valid, rnn_size, vocab_size, word_embeddings, embed_size, train_inputs,train_targets, lr_base, num_epochs, batch_size, vocab_size_all):

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    print('Build LSTM model.')
    model = Sequential()    
    model.add(Embedding(input_dim=len(word_embeddings),
                        output_dim=embed_size,
                        weights=[word_embeddings], # the matrix holding the trained embeddings
                        input_length=train_inputs.shape[1],
                        trainable=False))
    #model.add(Conv1D(filters=64, kernel_size=1, activation='relu'))
    #model.add(MaxPooling1D(pool_size=2))
    #model.add(LSTM(rnn_size, kernel_regularizer=l2(1E-2), recurrent_regularizer=l2(1E-2), bias_regularizer=l2(1E-2), return_sequences=True))
    model.add(LSTM(rnn_size, dropout=0.5, recurrent_dropout=0.5, return_sequences=True))
    #model.add(Dropout(rate=0.3, training=True))
    model.add(LSTM(rnn_size, dropout=0.5, recurrent_dropout=0.5, return_sequences=True))
    #model.add(LSTM(rnn_size, kernel_regularizer=l2(1E-2), recurrent_regularizer=l2(1E-2), bias_regularizer=l2(1E-2), return_sequences=True))
    #model.add(Dropout(rate=0.3, training=True))
    model.add(Dense(vocab_size_all, activation='softmax', name='dense_1')) #N_FEATURES = vocab_size
    model.add(Lambda(lambda x: x[:,-3:,:]))
    optimizer = SGD(lr=lr_base, momentum=0.9, decay=1E-2/num_epochs)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['categorical_accuracy', jaccard_coef, jaccard_coef_int])
    print("model built!")
    model.summary()
        
 
    ex = '/various/mtzortzi/from_nikela/ASCENDING_ORDER_METHOD/DESCENDING_ORDER_trains/phase1_exp5/models'
    model_dir = os.path.join('/various/mtzortzi/from_nikela/ASCENDING_ORDER_METHOD/DESCENDING_ORDER_trains/phase1_exp5', ex)

    print(model_dir)
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)

    ex1 = '/various/mtzortzi/from_nikela/ASCENDING_ORDER_METHOD/DESCENDING_ORDER_trains/phase1_exp5_tesnorboard'
    logdir = os.path.join('/various/mtzortzi/from_nikela/ASCENDING_ORDER_METHOD/DESCENDING_ORDER_trains', ex1)
    try:
        os.mkdir(logdir)
    except FileExistsError:
        pass    
    

    #callbacks
    tb = TensorBoard(log_dir=logdir)

    csv_logger = CSVLogger('/various/mtzortzi/from_nikela/ASCENDING_ORDER_METHOD/DESCENDING_ORDER_trains/phase1_exp5/training.log', separator=',', append=False)

    earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto')


    checkpoint = ModelCheckpoint(os.path.join(model_dir, 'model.{epoch:02d}-{val_loss:.4f}.hdf5'), 
                                 monitor='val_loss', verbose=1,
                                 save_best_only=True, save_weights_only=False, mode='auto', period=3)

    reduce_lr = ReduceLROnPlateau(monitor="val_loss", verbose=1, factor=0.1, patience=10, min_lr=1E-7)

    #class_weight
    y_ints = [y.argmax() for y in train_targets]
    unique_classes = np.unique(y_ints) #classes
    class_weights = class_weight.compute_class_weight('balanced', unique_classes, y_ints)
    #class_weights_dict = { unique_classes[i]: w for i,w in enumerate(class_weights) }
    
    
    listd=[]
    for i, w in enumerate(class_weights):
        if unique_classes[i] ==1:
            listd.append((unique_classes[i], w/500))
        #elif unique_classes[i] ==2:
        #    listd.append((unique_classes[i], 1.5*w))
        #elif unique_classes[i] ==3:
        #    listd.append((unique_classes[i], w/1.5))
        elif unique_classes[i] ==5:
            listd.append((unique_classes[i], w/100))
        #elif unique_classes[i] ==6:
        #   listd.append((unique_classes[i], w/100))
        #elif unique_classes[i] ==8:
        #   listd.append((unique_classes[i], 10*w))
        #elif unique_classes[i] ==10:
        #    listd.append((unique_classes[i], 100*w))
        #elif unique_classes[i] ==11:
        #    listd.append((unique_classes[i], 10*w))
        #elif unique_classes[i] ==13:
        #    listd.append((unique_classes[i], w/10))
        elif unique_classes[i] ==14:
            listd.append((unique_classes[i], 500*w))
        elif unique_classes[i] ==17:
            listd.append((unique_classes[i], 30*w))
        elif unique_classes[i] ==19:
            listd.append((unique_classes[i], 40*w))
        elif unique_classes[i] ==27:
            listd.append((unique_classes[i], 100*w))
        elif unique_classes[i] ==29:
            listd.append((unique_classes[i], 500*w))
        elif unique_classes[i] ==30:
            listd.append((unique_classes[i], 400*w))
        elif unique_classes[i] ==31:
            listd.append((unique_classes[i], 400*w))
        elif unique_classes[i] ==34:
            listd.append((unique_classes[i], 2000*w))
        elif unique_classes[i] ==38:
            listd.append((unique_classes[i], 5000*w))
        elif unique_classes[i] ==39:
            listd.append((unique_classes[i], 5000*w))
        elif unique_classes[i] ==40:
            listd.append((unique_classes[i], 6000*w))
        elif unique_classes[i] ==41:
            listd.append((unique_classes[i], 6000*w))
        elif unique_classes[i] ==42:
            listd.append((unique_classes[i], 6000*w))
        elif unique_classes[i] ==44:
            listd.append((unique_classes[i], 9000*w))
        elif unique_classes[i] ==47:
            listd.append((unique_classes[i], 10000*w))
        elif unique_classes[i] ==49:
            listd.append((unique_classes[i], 10000*w))
        elif unique_classes[i] ==51:
            listd.append((unique_classes[i], 10000*w))
        elif unique_classes[i] ==52:
            listd.append((unique_classes[i], 10000*w))
        elif unique_classes[i] ==54:
            listd.append((unique_classes[i], 10000*w))
        elif unique_classes[i] ==55: 
            listd.append((unique_classes[i], 300000*w))
        elif unique_classes[i] ==57: 
            listd.append((unique_classes[i], 300000*w))
        elif unique_classes[i] ==59:
            listd.append((unique_classes[i], 30000*w))
        elif unique_classes[i] ==60:
            listd.append((unique_classes[i], 30000*w))
        elif unique_classes[i] ==61:
            listd.append((unique_classes[i], 200000*w))
        elif unique_classes[i] ==62:
            listd.append((unique_classes[i], 200000*w))
        elif unique_classes[i] ==64:
            listd.append((unique_classes[i], 500000*w))
        elif unique_classes[i] ==66:
            listd.append((unique_classes[i], 200000*w))
        elif unique_classes[i] ==67:
            listd.append((unique_classes[i], 100000*w))
        elif unique_classes[i] ==68: 
            listd.append((unique_classes[i], 500000*w))
        elif unique_classes[i] ==69:
            listd.append((unique_classes[i], 60000000*w))
        elif unique_classes[i] ==70: 
            listd.append((unique_classes[i], 500000*w))
        elif unique_classes[i] ==71:
            listd.append((unique_classes[i], 100000*w))
        elif unique_classes[i] ==72: 
            listd.append((unique_classes[i], 500000*w))
        elif unique_classes[i] ==73:
            listd.append((unique_classes[i], 500000*w))
        elif unique_classes[i] ==76:
            listd.append((unique_classes[i], 100000000*w))
        elif unique_classes[i] ==79:
            listd.append((unique_classes[i], 500000*w))
        elif unique_classes[i] ==80:
            listd.append((unique_classes[i], 1000000*w))
        elif unique_classes[i] ==82:
            listd.append((unique_classes[i], 100000*w))
        elif unique_classes[i] ==83:
            listd.append((unique_classes[i], 500000*w))
        elif unique_classes[i] ==85:
            listd.append((unique_classes[i], 100000000*w))
        elif unique_classes[i] ==87:
            listd.append((unique_classes[i], 100000000*w))
        elif unique_classes[i] ==88:
            listd.append((unique_classes[i], 500000*w))
        elif unique_classes[i] ==91:
            listd.append((unique_classes[i], 100000000*w))
        elif unique_classes[i] ==94:
            listd.append((unique_classes[i], 100000000*w))
        elif unique_classes[i] ==96:
            listd.append((unique_classes[i], 100000000*w))
        elif unique_classes[i] ==99:
            listd.append((unique_classes[i], 100000000*w))
        elif unique_classes[i] ==102:
            listd.append((unique_classes[i], 100000000*w))
        else:
            listd.append((unique_classes[i], w))
    class_weights_dict = dict(listd)
    
    
    weights = np.zeros((train_targets.shape[0], train_targets.shape[1], vocab_size_all))
    for i in class_weights_dict:
        weights[:, :, i] += class_weights_dict[i]

    '''
    print('............load almost trained model from phase1..............')
    #dependencies = {'jaccard_coef': jaccard_coef, 'jaccard_coef_int': jaccard_coef_int}
    model = load_model('/various/mtzortzi/from_nikela/ASCENDING_ORDER_METHOD/DESCENDING_ORDER_trains/phase1_exp4/models/model.03-0.6467.hdf5', custom_objects={'jaccard_coef': jaccard_coef, 'jaccard_coef_int': jaccard_coef_int})#dependencies)
    history = pd.read_csv('/various/mtzortzi/from_nikela/ASCENDING_ORDER_METHOD/DESCENDING_ORDER_trains/phase1_exp4/training.log', sep=',', engine='python')
    print('DONE')
    '''

    #fit the model
    print('Training')
    start = time.time()
    history = model.fit(train_inputs_new, train_targets_new,
                     class_weight=weights,
		             verbose=1,
                     batch_size=batch_size,
                     shuffle=True,
                     epochs=num_epochs,
                     callbacks=[tb, checkpoint, csv_logger],# reduce_lr,# earlystop],
                     validation_data =(inputs_valid, targets_valid))
   

    end = time.time() 
    print("Time Took :{:3.2f} min".format( (end-start)/60 ))


    #save the model to file
    model.save('/various/mtzortzi/from_nikela/ASCENDING_ORDER_METHOD/DESCENDING_ORDER_trains/phase1_exp5/model1.h5')

    return model, history


# inherit from the keras.layers.Dropout class and overwrite its call-method. 
# In additon I added the kwarg training=True to the init-method before calling super with the arguments expected by the base-class.
class Dropout(keras.layers.Dropout):
    """Applies Dropout to the input.
    Dropout consists in randomly setting
    a fraction `rate` of input units to 0 at each update during training time,
    which helps prevent overfitting.
    # Arguments
        rate: float between 0 and 1. Fraction of the input units to drop.
        noise_shape: 1D integer tensor representing the shape of the
            binary dropout mask that will be multiplied with the input.
            For instance, if your inputs have shape
            `(batch_size, timesteps, features)` and
            you want the dropout mask to be the same for all timesteps,
            you can use `noise_shape=(batch_size, 1, features)`.
        seed: A Python integer to use as random seed.
    # References
        - [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](
           http://www.jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf)
    """
    def __init__(self, rate, training=None, noise_shape=None, seed=None, **kwargs):
        super(Dropout, self).__init__(rate, noise_shape=None, seed=None,**kwargs)
        self.training = training

        
    def call(self, inputs, training=None):
        if 0. < self.rate < 1.:
            noise_shape = self._get_noise_shape(inputs)

            def dropped_inputs():
                return K.dropout(inputs, self.rate, noise_shape,
                                 seed=self.seed)
            if not training: 
                return K.in_train_phase(dropped_inputs, inputs, training=self.training)
            return K.in_train_phase(dropped_inputs, inputs, training=training)
        return inputs


if __name__=="__main__":
	main()

