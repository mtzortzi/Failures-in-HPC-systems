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
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard,  ReduceLROnPlateau, CSVLogger  
from keras import initializers
import keras.utils as ku 
from keras import backend as K
from keras.models import load_model
from keras.regularizers import l2

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
import phase1_train_lstm
import phase1_Anwesha_lstm
import anwesha_prep2_DT

def main():

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    device_lib.list_local_devices()

    print('............read phase2_anwesha............')
    phase2_new = pd.read_csv('/various/mtzortzi/from_nikela/phase2_anwesha.csv') 
    print('phase2 shape', phase2_new.shape)
    print('DONE')

    #print('...........create a new column [DT,WIDS].....................')
    #phase2_new = create_phrase_vector_DT_WIDS.final_prep('/various/mtzortzi/from_nikela/phase2.csv')
    #print('DONE')

    print('........create a new df with only event_time, wids and DT..................')
    phase2_multi = phase2_new.filter(['EVENT_TIME', 'WIDS', 'DT'])
    phase2_multi = phase2_multi.set_index('EVENT_TIME', drop=True, inplace=False)
    print('phase2_multi shape', phase2_multi.shape)
    #phase2_multi.head(1)
    print('DONE')

    print('.........to_supervised.............')
    values2 = phase2_multi.values
    n_sequences2 = phase1_Anwesha_lstm.series_to_supervised(values2, 5, 1)
    print(n_sequences2.head(2))
    print('DONE')

    print('....read embedding.....')
    #word_embeddings = pd.read_csv('/various/mtzortzi/from_nikela/word_embeddings_phase1.csv')
    word_embeddings = pd.read_csv('/various/mtzortzi/from_nikela/withoutNAN_word_embeddings_phase1.csv')
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
    n_features2 = 2 #DT and wids
    n_obs2 = 5 * n_features2
    values22 = n_sequences2.values
    train_X = values22[:,0:n_obs2] #+1 is the var1t
    train_Y = values22[:, -2]
    #print('train_Y', train_Y)
    print('DONE')

    print('....train targets to categorical....')
    train_Y = ku.to_categorical(train_Y, num_classes=vocab_size_all)
    print('DONE')
    # print('train_Y after categorical:')
    # print(train_Y)
    print('train_Y.shape', train_Y.shape)
    # reshape input to be 3D [samples, timesteps, features]
    print('train_X.shape', train_X.shape)
    train_Y = train_Y.reshape(train_Y.shape[0], 1, train_Y.shape[1])
    print('train_Y shape after reshape', train_Y.shape)

    # Extract word embeddings for only the words in phase2
    print('.........extract word embeddings for only the words in phase2............')
    embedding_matrix2 = np.zeros((len(phase2_new['AN_wids'].unique()) + 1, embed_size))
    word_embeddings2 = word_embeddings.T.to_dict('list')
    embedding_list = []
    word_list = []
    for word, i in word2id_all.items():
        if phase2_new['AN_multiwords'].str.contains(word).any():
            embedding_list.append((word, word_embeddings2.get(word)))
    my_dict = dict(embedding_list) 
    my_dict = {k: v for k, v in my_dict.items() if v is not None}
    print('my_dict', my_dict)
    word_embeddings_phase2 = pd.DataFrame.from_dict(my_dict, orient='index')
    print('word_embeddings_phase2 shape', word_embeddings_phase2.shape)
    print(type(word_embeddings_phase2))
    print('word_embeddings shape', word_embeddings.shape)
    print(type(word_embeddings))
    print('DONE')

    rnn_size = 255 # size of RNN
    #learning_rate = 0.001 #learning rate
    num_epochs = 150 # number of epochs
    batch_size = 72
    print('....training lstm model.....')
    initial_epochs = 250
    model2, history2 = lstm_model2(batch_size, rnn_size, train_X, train_Y, vocab_size_all, word_embeddings_phase2, embed_size, num_epochs)
    print('DONE')

    print('............evaluate the model............')
    loss, accuracy, mae   = model2.evaluate(train_X, train_Y, verbose=1)
    print('Accuracy: %f' % (accuracy*100))
    print('loss, mae', loss, mae)
    print('DONE')


    predicted = model2.predict(train_X) #possibilities
    print('..............actual-real classes..................')
    print(np.argmax(train_Y, axis=-1)) # actual classes
    print('.............predicted classes.....................')
    clas = model2.predict_classes(train_X) #predicted classes
    print(clas)
    # I take the possibilities, I create the predicted classes & I make them categorical so as to fit with categorical train_Y
    pred_max = ku.to_categorical(np.argmax(predicted, axis=-1), num_classes = vocab_size_all)
    pred_max = pred_max.reshape(pred_max.shape[0], 1, pred_max.shape[1]) #predicted    
    print('-------pred_max-------')
    print(pred_max)
    print('pred_max shape', pred_max.shape)
    print('test_Y shape', train_Y.shape)
    print('...MSE is the same as loss of course...')
    print(np.square(np.subtract(train_Y, pred_max)).mean())
    print('DONE')

    # serialize model to JSON
    model_json = model2.to_json()
    with open("/various/mtzortzi/from_nikela/ANWESHA/phase2/model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model2.save_weights("/various/mtzortzi/from_nikela/ANWESHA/phase2/saved_model.h5")
    print("Saved model to disk")



def lstm_model2(batch_size, rnn_size, train_X, train_Y, vocab_size_all, word_embeddings_phase2, embed_size, num_epochs):

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    print('Build LSTM model.')      
    model2 = Sequential() 
    model2.add(Embedding(input_dim=len(word_embeddings_phase2),
                    output_dim=embed_size,
                    weights=[word_embeddings_phase2], # the matrix holding the trained embeddings
                    input_length=train_X.shape[1],
                    trainable=False))
    model2.add(LSTM(rnn_size, kernel_regularizer=l2(1E-1), recurrent_regularizer=l2(1E-1), bias_regularizer=l2(1E-1), return_sequences=True))#, input_shape=(train_X.shape[1], train_X.shape[2])))       
    model2.add(Dropout(rate=0.2))#, training=True))
    model2.add(LSTM(rnn_size, return_sequences=True))
    model2.add(Dropout(rate=0.2))#, training=True))
    model2.add(Dense(vocab_size_all, name='dense_2')) 
    model2.add(Lambda(lambda x: x[:,-1:,:]))
    learning_rate = 0.01 # * (float(batch_size) / 16)#learning rate 0.01 * (float(batch_size) / 16) batch_size = 128
    optimizer = RMSprop(lr=learning_rate, rho=0.9)
   
    model2.compile(loss='mse', optimizer= optimizer, metrics=['accuracy', 'mae'])
    print("model built!")
    model2.summary()

        
    ex = '/various/mtzortzi/from_nikela/ANWESHA/phase2/models'
    model_dir = os.path.join('/various/mtzortzi/from_nikela/ANWESHA/phase2', ex)
    print(model_dir)
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)

    ex1 = '/various/mtzortzi/from_nikela/ANWESHA/phase2_tensorboard'
    logdir = os.path.join('/various/mtzortzi/from_nikela/ANWESHA', ex1)
    try:
        os.mkdir(logdir)
    except FileExistsError:
        pass    
    

    #callbacks
    csv_logger = CSVLogger('/various/mtzortzi/from_nikela/ANWESHA/phase2/training.log', separator=',', append=False)

    tb = TensorBoard(log_dir=logdir)

    #earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto')

    checkpoint = ModelCheckpoint(os.path.join(model_dir, 'model.{epoch:02d}-{val_loss:.4f}.hdf5'), 
                                 monitor='val_loss', verbose=1,
                                 save_best_only=True, save_weights_only=False, mode='auto', period=3)
    
    reduce_lr = ReduceLROnPlateau(monitor="val_mean_squared_error", patience=5, verbose=1)
    


    #class_weight
    y_ints2 = [y.argmax() for y in train_Y]
    unique_classes2 = np.unique(y_ints2) #classes
    class_weights2 = class_weight.compute_class_weight('balanced', unique_classes2, y_ints2)
    #class_weights_dict2 = { unique_classes2[i]: w for i,w in enumerate(class_weights2) }

    listd=[]
    for i, w in enumerate(class_weights2):
        if unique_classes2[i] ==14: 
            listd.append((unique_classes2[i], 100*w))
        elif unique_classes2[i] ==31: 
           listd.append((unique_classes2[i], 200*w))
        elif unique_classes2[i] ==11: 
            listd.append((unique_classes2[i], 1000*w))
        elif unique_classes2[i] ==34: 
            listd.append((unique_classes2[i], 100*w))
        elif unique_classes2[i] ==38: 
            listd.append((unique_classes2[i], 100*w))
        elif unique_classes2[i] ==40: 
            listd.append((unique_classes2[i], 10*w))
        elif unique_classes2[i] ==46: 
            listd.append((unique_classes2[i], 100*w))
        elif unique_classes2[i] ==49:
            listd.append((unique_classes2[i], 10*w))
        elif unique_classes2[i] ==51: 
            listd.append((unique_classes2[i], 10*w))
        else:
            listd.append((unique_classes2[i], w))
    class_weights_dict2 = dict(listd)


    weights2 = np.zeros((train_Y.shape[0], train_Y.shape[1], vocab_size_all))
    for i in class_weights_dict2:
        weights2[:, :, i] += class_weights_dict2[i]
    
 
     #fit the model
    history2 = model2.fit(train_X, train_Y, class_weight=weights2,
                     batch_size=batch_size,
                     shuffle=True,
                     epochs=num_epochs,
                     callbacks=[tb, checkpoint, reduce_lr, csv_logger],#, earlystop],
                     validation_split=0.1)


    #save the model to file
    model2.save('/various/mtzortzi/from_nikela/ANWESHA/phase2/multi_model.h5') 
    
    return model2, history2


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









