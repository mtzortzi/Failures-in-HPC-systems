import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
#os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = ""



import datetime
import pandas as pd
import csv
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
import fileinput
import random
import scipy.special
import math
import numpy as np
import scipy.stats
import pickle
from math import log
import numpy as np
import pickle
import io 
from tqdm import tqdm
from datetime import datetime
import time
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers.embeddings import Embedding
from keras.layers.core import Activation, Dropout, Dense
from keras.layers import Flatten, LSTM
from keras.layers import GlobalMaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from numpy import array
from keras.models import Model
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.layers import Input
from keras.layers.merge import Concatenate
import re
import matplotlib.pyplot as plt

from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

from keras.preprocessing import text
from keras.preprocessing.sequence import skipgrams
from keras.layers import Merge
from keras.layers.core import Dense, Reshape
from keras.layers.embeddings import Embedding
from keras.models import Sequential
# install Graphviz after download installer (https://www.graphviz.org/)
# insert in code this two lines:
import os
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.manifold import TSNE
from pickle import dump
from keras import initializers


# generate skip-grams
def skipgram_gen(vocab_size, wids, id2word):
#     skip_grams = [skipgrams(wids, vocabulary_size=vocab_size, window_size=3) for wid in wids]
    skip_grams = [skipgrams(wids, vocabulary_size=vocab_size, window_size=5)]
    # view sample skip-grams
    pairs, labels = skip_grams[0][0], skip_grams[0][1]
    '''
    for i in range(1):
        print("({:s} ({:d}), {:s} ({:d})) -> {:d}".format(
              id2word[pairs[i][0]], pairs[i][0],
              id2word[pairs[i][1]], pairs[i][1],
              labels[i]))
    '''
    return skip_grams


# build skip-gram architecture
def skip_gram_arch(vocab_size, embed_size):
    word_model = Sequential()
    word_model.add(Embedding(vocab_size, embed_size, 
                             embeddings_initializer=initializers.RandomNormal(mean=0.0, stddev=0.001, seed=None),
                             #embeddings_initializer="glorot_uniform",
                             input_length=1))
    word_model.add(Reshape((embed_size, )))

    context_model = Sequential()
    context_model.add(Embedding(vocab_size, embed_size,
                      embeddings_initializer=initializers.RandomNormal(mean=0.0, stddev=0.001, seed=None),
                      #embeddings_initializer="glorot_uniform",
                      input_length=1))
    context_model.add(Reshape((embed_size,)))

    model = Sequential()
    model.add(Merge([word_model, context_model], mode="dot"))
    model.add(Dense(1, kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.001, seed=None), 
              #kernel_initializer="glorot_uniform", 
              activation="sigmoid"))
    model.compile(loss="mean_squared_error", optimizer="rmsprop")

    # view model summary
    print(model.summary())   
    return model    


def train_model(skip_grams, model):
    #epochs = 5
    for epoch in range(1, 6):
        loss = 0
        for i, elem in enumerate(skip_grams):
            print('i from train_model', i)
            a = np.array(list(zip(*elem[0])), dtype='int32')
            if a.any() == False:
                print('I found a nan value')
                continue
            else:
                pair_first_elem = a[0] #np.array(list(zip(*elem[0]))[0], dtype='int32')
                pair_second_elem = a[1] #np.array(list(zip(*elem[0]))[1], dtype='int32')
                labels = np.array(elem[1], dtype='int32')
                X = [pair_first_elem, pair_second_elem]
                Y = labels
                if i % 10000 == 0:
                    #print('Χ Υ', X, Y )
                    print('Processed {} (skip_first, skip_second, relevance) pairs'.format(i))
                loss += model.train_on_batch(X,Y)  

        print('Epoch:', epoch, 'Loss:', loss)
    return X, Y

def get_word_embed(model, id2word):
    merge_layer = model.layers[0]
    word_model = merge_layer.layers[0]
    word_embed_layer = word_model.layers[0]
    weights = word_embed_layer.get_weights()[0][1:89] 
    #weights = word_embed_layer.get_weights()[0][1:87] #vocab_size 87 for DIVIDE60:40

    print('weights.shape', weights.shape)
    word_embeddings = pd.DataFrame(weights, index=id2word.values())
    return weights, word_embeddings


# I take the phase1_nodewise.csv from train30_test70_nodewise_ascending_order.py
def create_nodewise_ds():
	phase1_nodewise = pd.read_csv('/various/mtzortzi/from_nikela/ASCENDING_ORDER_METHOD/phase1_nodewise.csv')
	return phase1_nodewise


# Using dictionary comprehension + items() 
# Extracting specifix keys from dictionary 

def extract_word2id(word2id_all):
	return  {key: word2id_all[key] for key in word2id_all.keys() 
                               & {'thetransmittingnodedidnotalignwithlinkchipr22', 'alinkchipdidnotalignalongtheaportonswitch0', 'opticalmoduleenvironmentaldataisunavailable', 'l1pcorrectableerrorsummary', 'zero', 'serviceaction5516turnednodeboardr1e', 'linkfailuredetectedbetweennodesconnectedviacopperlinks', 'baddramwasdetected', 'ddr0ue', 'aconnectionistakingatocomplete', 'ddrarbitermachinecheck', 'ndreceivercorrectableerror', 'unabletoreadcardenvironmentaldata', 'ndcorrectableerror', 'trainingfailuredetected', 'mrkstdta', 'serviceaction6585startedtoservicer13', 'dcrarbitermachinecheck', 'abqllanewasspared', 'linkfailuredetectedbetweennodesconnectedviacopperandopticallinks', 'kernelunexpectedoperation', 'cfammachinecheck', 'unrecoverablemachinecheck', 'messageuniteccsummary', 'cable', 'badphywasdetected', 'serviceaction4961startedtoservicer20', 'serviceaction6585restartednodedcar13', 'sat1', 'alinkchipdidnotalignalongtheaportonswitch1', 'l2machinecheck', 'sincetheboardisunavailablewearefailingthisbootstep', 'ddrcorrectableerrorsummary', 'sat0', 'alinkchipdidnotbitalignalongthecport', 'abqlbiterrorthresholdwasexceededbutsparingisnotpossible', 'verificationofthekernelshutdownfailed', 'encounteredanexception', 'l2arraycorrectableerrorsummary', 'warningmc1rank0', 'cfamlivelockbusterfailure', 'thebqcclocksarenotinthecorrectstate', 'mailboxverificationfailedforthisnode', 'unabletoupdatethelcddisplay', 'warningmc1rank1', 'alinkchipdidnotalignalongtheaportonswitch2', 'kernelinternalassertionfailure', 'l1pmachinecheck', 'read', 'ddr0phywasrecalibrated0', 'addr', 'abqlbiterrorthresholdwasexceeded', 'a2processormachinecheck', 'plldidnotlock', 'devbusmachinecheck', 'successfullyresetthiscard', 'ndreceiverlinkerror', 'ndsenderretransmissioncorrectableerror', 'checkforsuccessofporsequencefailed', 'alinkchipdidnotalignalongtheaandbportsonswitch3', 'ddr1ue', 'ddr1phywasrecalibrated4', 'serdeslinkfailure', 'ddr1phywasrecalibrated1', 'l2directorycorrectableerrorsummary', 'memorycontrollerinitializationwarning', 'a2tlbparityerror', 'ddr1phywasrecalibrated0', 'sendofakernelshutdownmessagefailed', 'detectedthatthisboardhasbecomeunusable', 'alinkchipdidnotbitalignalongthereceivercport', 'thebroadcastinstallofakernelimagefailed', 'memorycontrollerinitializationerror', 'pllproblemonbqcchip', 'warningmc0rank1', 'rasstormwarning', 'messageunitrecoverableerror', 'detectedapowerrailwithanincorrectvoltage', 'cfamrecoverableerror', 'accessalert', 'thenodesentanunexpectedmailboxcommand', 'ndfatalerror', 'abqldoublebiterrorthresholdwasexceeded', 'warningmc0rank0', 'cnkunexpectedmuorndinterrupt', 'tvsensetemperatureisunavailable', 'rasstormerror', 'cnkdetectedanullipitargetfunctionpointer'}}


def extract_word2id_DIVIDE6040(word2id_all):
	return  {key: word2id_all[key] for key in word2id_all.keys() 
                               & {'warningmc0rank1', 'thebqcclocksarenotinthecorrectstate', 'detectedthatthisboardhasbecomeunusable', 'warningmc1rank0', 'aconnectionistakingatocomplete', 'kernelinternalassertionfailure', 'ddrcorrectableerrorsummary', 'pllproblemonbqcchip', 'ddr1ue', 'ndcorrectableerror', 'trainingfailuredetected', 'alinkchipdidnotbitalignalongthecport', 'serdeslinkfailure', 'alinkchipdidnotalignalongtheaportonswitch1', 'alinkchipdidnotalignalongtheaandbportsonswitch3', 'ddr1phywasrecalibrated0', 'addr', 'cnkunexpectedgeainterrupt', 'illegaldcraccess', 'ddr1phywasrecalibrated1', 'thebroadcastinstallofakernelimagefailed', 'linkfailuredetectedbetweennodesconnectedviacopperandopticallinks', 'mrkstdta', 'plldidnotlock', 'l2arraycorrectableerrorsummary', 'abqlbiterrorthresholdwasexceededbutsparingisnotpossible', 'accessalert', 'kernelunexpectedoperation', 'a2processormachinecheck', 'ddr0ue', 'serviceaction5516turnednodeboardr1e', 'warningmc0rank0', 'ndfatalerror', 'baddramwasdetected', 'abqllanewasspared', 'zero', 'unrecoverablemachinecheck', 'unabletoreadcardenvironmentaldata', 'alinkchipdidnotalignalongtheaportonswitch0', 'sincetheboardisunavailablewearefailingthisbootstep', 'successfullypoweredthisdevice', 'detectedapowerrailwithanincorrectvoltage', 'linkfailuredetectedbetweennodesconnectedviacopperlinks', 'read', 'abqlbiterrorthresholdwasexceeded', 'warningmc1rank1', 'devbusmachinecheck', 'dcrarbitermachinecheck', 'sendofakernelshutdownmessagefailed', 'ddr0phywasrecalibrated4', 'cnkdetectedanullipitargetfunctionpointer', 'sat1', 'encounteredanexception', 'alinkchipdidnotalignalongtheaportonswitch2', 'ndreceiverlinkerror', 'l1pcorrectableerrorsummary', 'ddr1phywasrecalibrated4', 'rasstormwarning', 'ddrarbitermachinecheck', 'messageuniteccsummary', 'l2machinecheck', 'cfammachinecheck', 'cable', 'cfamlivelockbusterfailure', 'a2tlbparityerror', 'memorycontrollerinitializationerror', 'theinstallofakernelimagefailed', 'checkforsuccessofporsequencefailed', 'alinkchipdidnotbitalignalongthereceivercport', 'badphywasdetected', 'opticalmoduleenvironmentaldataisunavailable', 'thetransmittingnodedidnotalignwithlinkchipr22', 'l1pmachinecheck', 'verificationofthekernelshutdownfailed', 'rasstormerror', 'successfullyresetthiscard', 'sat0', 'ndreceivercorrectableerror', 'messageunitrecoverableerror', 'l2directorycorrectableerrorsummary', 'ndsenderretransmissioncorrectableerror', 'cnkunexpectedmuorndinterrupt', 'abqldoublebiterrorthresholdwasexceeded', 'memorycontrollerinitializationwarning', 'cfamrecoverableerror', 'ddr0phywasrecalibrated0'}}


def extract_id2word(word2id):
	return {v: k for k, v in word2id.items()}
 
def find_wids(dataset, word2id_all):
	#dataset is phase1_nodewise
	wids = []
	for msg in dataset['MULTIWORDS']:
		wids.append(word2id_all[msg])
	return wids

def compute_vocab_size(word2id):
	return len(set(word2id.keys()))+1

def compute_vocab_size_all(word2id_all):
	word2id=extract_word2id(word2id_all)
	return compute_vocab_size(word2id)

##from skipgram import vocab_size, wids, word_embeddings

def compute_word_embeddings(vocab_size_all, wids, id2word, embed_size):
	skip_grams = skipgram_gen(vocab_size_all, wids, id2word)
	model = skip_gram_arch(vocab_size_all, embed_size)
	X, Y = train_model(skip_grams, model)
	weights, word_embeddings = get_word_embed(model, id2word)
	return weights, word_embeddings

def store_wids(dataset, wids, filename):
	dataset['WIDS'] = wids
	dataset.to_csv(filename, index = False, sep = ',', quoting=csv.QUOTE_ALL)
#I create a new column that I'll need later in phase2.
'''
phase1_nodewise['WIDS'] = wids
phase1_nodewise.to_csv('/various/mtzortzi/LogAider-master/RAS-JOB/phase1_nodewise_wids.csv', index=False, sep=',', quoting=csv.QUOTE_ALL)
'''

