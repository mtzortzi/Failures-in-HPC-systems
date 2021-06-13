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

from sklearn.utils import class_weight
from sklearn.metrics import f1_score
import collections

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


##Custom
##from skipgram import vocab_size, wids, id2word, word2id, word_embeddings
import skipgram
#import tensorflow as tf
#sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
#:print('sess', sess)


##faulthandler.enable(file=sys.stderr, all_threads=True)

import multiwords_train30_test70
import phase1_exp4
#import phase2prep_DTcalc_for_ph2_ph3
import skipgram
from metrics import *
import phase2_exp4


def main():

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    device_lib.list_local_devices()

    print('.....read phase3.csv...')
    print('...I had already reorder phase3.csv by descending order...')
    phase3 = pd.read_csv('/various/mtzortzi/from_nikela/ASCENDING_ORDER_METHOD/DESCENDING_ORDER_trains/phase3.csv') 
    print('phase3 shape', phase3.shape)
    print('DONE')

    print('....getting vocabulary sizes.....')
    merged_static = multiwords_train30_test70.create_merged('/various/mtzortzi/LogAider-master/RAS-JOB/merged_static.csv')
    corpus_all, vocab_size_all, wids_all, id2word_all, word2id_all, tokenizer_all = multiwords_train30_test70.build_corpus(merged_static['MULTIWORDS'])
    #vocab_size = skipgram.compute_vocab_size_all(word2id_all)
    print('vocab_size_all', vocab_size_all)
    word2id3 = extract_word2id3(word2id_all)
    id2word3 = skipgram.extract_id2word(word2id3)
    print('id2word_all', id2word_all)
    print('id2word3', id2word3)
    wids = skipgram.find_wids(phase3, word2id_all)
    vocab_size3 = skipgram.compute_vocab_size(word2id3)
    print('vocab_size3', vocab_size3)
    print('DONE')
    embed_size = 100
    
    print('....store phase3_wids....')
    skipgram.store_wids(phase3, wids, '/various/mtzortzi/from_nikela/ASCENDING_ORDER_METHOD/DESCENDING_ORDER_trains/phase3_wids.csv')
    print('DONE')

    print('.....read phase3_wids.csv...')
    phase3_wids = pd.read_csv('/various/mtzortzi/from_nikela/ASCENDING_ORDER_METHOD/DESCENDING_ORDER_trains/phase3_wids.csv') 
    print('phase3_wids shape', phase3_wids.shape)
    print('DONE')


    print('............load model from phase2..............')
    model2 = load_model('/various/mtzortzi/from_nikela/ASCENDING_ORDER_METHOD/DESCENDING_ORDER_trains/phase2_exp5/models/model.3879-0.0033.hdf5')#model.1873-0.0031.hdf5')
    history2 = pd.read_csv('/various/mtzortzi/from_nikela/ASCENDING_ORDER_METHOD/DESCENDING_ORDER_trains/phase2_exp5/training.log', sep=',', engine='python')
    print('DONE')

    TP_all = []
    FP_all = []
    FN_all = []
    TN_all = []
    TP_FC = 0
    TN_FC = 0
    FP_FC = 0
    FN_FC = 0
    for key, msg in phase3_wids.groupby(['LOCATION']):
        print("the group for Location node '{}' has {} rows".format(key,len(msg)))
        #print(msg)
        # 11 is the History Size
        if len(msg) > 10:
            #print(msg.LOCATION, msg.MULTIWORDS, multi_time_series(msg))
            #print('---------------')
            MSE, TP_all, FP_all, FN_all, TN_all = multi_time_series(msg, vocab_size_all, model2, TP_all, FP_all, FN_all, TN_all)
            #print('msg.WIDS', msg.WIDS)
            #print('msg.DT', msg.DT)
            print('MSE', MSE)
            if MSE >= 0.0147:
                print('Not suspicious outcomes to check for failures')
                if msg['DT'].isin([0]).any():
                    FN_FC += 1
                    print('I found a FN')
                else:
                    TN_FC += 1                
            else:
                print('msg.WIDS', msg.WIDS)
                print('msg.DT', msg.DT)
                if msg['DT'].isin([0]).any():
                    TP_FC += 1
                else:
                    FP_FC +=1 
                print('---------------')      

    print('...........calculate final metrics for all the nodes in MIRA-micro Average.........')
    final_metrics_for_MIRA(TP_all, FP_all, FN_all, TN_all)
    print('DONE')

    print('..........calculate metrics for failure chains in mira.....')
    final_metrics_for_FailureChains_Mira(TP_FC, FP_FC, TN_FC, FN_FC)
    print('DONE')


def final_metrics_for_FailureChains_Mira(TP_FC, FP_FC, TN_FC, FN_FC):
    print({'TP_FC':TP_FC, 'FP_FC': FP_FC, 'TN_FC': TN_FC, 'FN_FC': FN_FC})
    FP_Rate_FC = FP_FC/(FP_FC+TN_FC)
    FN_Rate_FC = FN_FC/(TP_FC+FN_FC)
    precisions_FC = precision_FC(TP_FC, FP_FC, FN_FC)
    recalls_FC = recall_FC(TP_FC, FP_FC, FN_FC)
    accuracy_FC = weighted_accuracy_FC(TP_FC, FP_FC, FN_FC, TN_FC)
    F1_FC = f1_measure_FC(precisions_FC, recalls_FC)
    print('recalls for failure chains', recalls_FC)
    print('accuracy for failure chains', accuracy_FC)
    print('precisions for failure chains', precisions_FC)
    print('F1 for failure chains', F1_FC)
    print('FP_Rate for failure chains', FP_Rate_FC)
    print('FN_Rate for failure chains', FN_Rate_FC)

    return



# Using dictionary comprehension + items() 
# Extracting specifix keys from dictionary 

def extract_word2id3(word2id_all):
	return  {key: word2id_all[key] for key in word2id_all.keys() 
                               & {'opticalmoduleenvironmentaldataisunavailable', 'messageunitrecoverableerror', 'cnkunexpectedgeainterrupt', 
                               'cnkdetectedanullipitargetfunctionpointer', 'detectedapowerrailwithanincorrectvoltage', 'ddr0phywasrecalibrated0',
                                'rasstormwarning', 'cable', "bumpedthisdca'sdomain1voltageupbydefaultdomain1voltagebumpmv", 'illegaldcraccess',
                                 'serviceaction6585completedonr13', 'theinstallofakernelimagefailed', 'rasstormerror', 'cfamrecoverableerror', 
                                 'ddrcorrectableerrorsummary', 'thenodesentanunexpectedmailboxcommand', 'warningmc0rank1', 'abqlbiterrorthresholdwasexceeded',
                                  'alinkchipdidnotalignalongtheaportonswitch0', 'ndreceiverlinkerror', 'ddr0ue', 'unabletoreadcardenvironmentaldata', 
                                  'sincetheboardisunavailablewearefailingthisbootstep', 'l2arraycorrectableerrorsummary', 'unabletoupdatethelcddisplay', 'sat0', 'addr',
                                   'cnkunexpectedmuorndinterrupt', 'ndcorrectableerror', 'pllproblemonbqcchip', 'aconnectionistakingatocomplete', 
                                   'a2processormachinecheck', 'mrkstdta', 'badphywasdetected', 'checkforsuccessofporsequencefailed', 
                                   'alinkchipdidnotalignalongtheaportonswitch2', 'successfullypoweredthisdevice', 'l1pcorrectableerrorsummary',
                                    'alinkchipdidnotbitalignalongthereceivercport', 'sat1', 'thetransmittingnodedidnotalignwithlinkchipr22',
                                     'l2directorycorrectableerrorsummary', 'ddr1ue', 'detectedthatthisboardhasbecomeunusable', 'ddr0phywasrecalibrated4',
                                      'abqldoublebiterrorthresholdwasexceeded', 'serviceaction4961startedtoservicer20', 'l1pmachinecheck',
                                       'alinkchipdidnotalignalongtheaandbportsonswitch3', 'read', 'thebqcclocksarenotinthecorrectstate',
                                        'mailboxverificationfailedforthisnode', 'kernelunexpectedoperation', 'ddr1phywasrecalibrated4', 'encounteredanexception',
                                         'dcrarbitermachinecheck', 'memorycontrollerinitializationerror', 'sendofakernelshutdownmessagefailed', 
                                         'cfamlivelockbusterfailure', 'successfullyresetthiscard', 'successfullyperformedthespecifiedoperationonthisdca',
                                          'ndsenderretransmissioncorrectableerror', 'disabledthisdca', 'baddramwasdetected', 'cfammachinecheck', 'enabledthisdca',
                                           'detectedthatoneofthedcasonthisboardhasexperiencedadomain1powerfailure', 'ddrarbitermachinecheck', 'warningmc1rank0', 
                                           'ddr1phywasrecalibrated0', 'a2tlbparityerror', 'warningmc0rank0', 'unrecoverablemachinecheck', 
                                           'linkfailuredetectedbetweennodesconnectedviacopperlinks', 'trainingfailuredetected', 'abqllanewasspared',
                                            'kernelinternalassertionfailure', 'tvsensetemperatureisunavailable', 
                                            'abqlbiterrorthresholdwasexceededbutsparingisnotpossible', 'messageuniteccsummary',
                                             'ndfatalerror', 'devbusmachinecheck', 'serviceaction6585turnednodedcar13', 'accessalert', 
                                             'ndreceivercorrectableerror', 'ddr1phywasrecalibrated1', 'serdeslinkfailure', 
                                             'verificationofthekernelshutdownfailed',
                                              'linkfailuredetectedbetweennodesconnectedviacopperandopticallinks', 'l2machinecheck', 'zero', 
                                              'plldidnotlock', 'memorycontrollerinitializationwarning', 
                                              'alinkchipdidnotalignalongtheaportonswitch1',
                                'warningmc1rank1', 'alinkchipdidnotbitalignalongthecport', 'thebroadcastinstallofakernelimagefailed'}}



def multi_time_series(phase, vocab_size_all, model2, TP_all, FP_all, FN_all, TN_all):
    phase_mt = phase.filter(['EVENT_TIME', 'WIDS', 'DT'])
    phase_mt = phase_mt.set_index('EVENT_TIME', drop=True, inplace=False)
    #print('phase_mt shape for each location', phase_mt.shape)
    #print('phase_mt head 3 for each location', phase_mt.head(3))
    values3 = phase_mt.values
    n_sequences3 = phase1_exp4.series_to_supervised(values3, 10, 1)
    #print(n_sequences3.head(2))
    n_features3 = 2 #DT and wids
    n_obs3 = 10 * n_features3 # 10 is the HS(History Size)    
    values33 = n_sequences3.values
    test_X, test_Y = values33[:, 0:n_obs3], values33[:, -2]
    #print('test_Y from values33', test_Y)
    test_Y = ku.to_categorical(test_Y, num_classes=vocab_size_all)
    test_X = test_X.reshape(test_X.shape[0], 20)# 10, 2) # reshape input to be 3D [samples, timesteps, features]
    print('test_X.shape', test_X.shape)
    print('len test_X', len(test_X))  
    #output timesteps 1
    test_Y = test_Y.reshape(test_Y.shape[0], 1, test_Y.shape[1])
    #print('3D test_Y shape',test_Y.shape)
     
    # make a prediction
    preds = model2.predict(test_X) #preds or yhat is the same -----> possibilities
    #print('preds', preds)
    clas = model2.predict_classes(test_X)
    p_max = ku.to_categorical(np.argmax(preds, axis=-1), num_classes = vocab_size_all)
    p_max = p_max.reshape(p_max.shape[0], 1, p_max.shape[1]) #predicted
    #p_thres = (preds > 0.5).astype(float)
    print('test_Y shape', test_Y.shape)
    #print('preds possibilities', preds.argmax(axis=-1))
    print('p_max shape', p_max.shape)
    MSE = np.square(np.subtract(test_Y,p_max)).mean()  

    # for each node
    tp, fp, fn, tn = confusion_matrix(test_Y, p_max)
    #print('tp, fp, fn, tn', tp, fp, fn, tn)
    TP = np.sum(tp)
    TN = np.sum(tn)
    FN = np.sum(fn)
    FP = np.sum(fp)
    # appending for all the nodes, regardless if they contain failures or not 
    TP_all.append(TP)
    TN_all.append(TN)
    FN_all.append(FN)
    FP_all.append(FP)
    results = {'TP': TP, 'FP': FP, 'TN': TN, 'FN': FN}
    #print('results', results)
    #print({'TP_all': TP_all, 'FP_all': FP_all, 'TN_all': TN_all, 'FN_all': FN_all})

    #if MSE <= 0.0152:
    print('p_max predicted', p_max.argmax(axis=-1)) #without argmax we have categorical values
    print('test_Y actual', test_Y.argmax(axis=-1))  #without argmax we have categorical values


    return MSE, TP_all, FP_all, FN_all, TN_all



def final_metrics_for_MIRA(TP_all, FP_all, FN_all, TN_all):
    TP_final = sum(TP_all)
    FP_final = sum(FP_all)
    TN_final = sum(TN_all)
    FN_final = sum(FN_all)
    print({'TP_final': TP_final, 'FP_final': FP_final, 'TN_final': TN_final, 'FN_final': FN_final})
    FP_Rate = FP_final/(FP_final+TN_final)
    FN_Rate = FN_final/(TP_final+FN_final)
    precisions = precision(TP_final, FP_final, FN_final)
    recalls = recall(TP_final, FP_final, FN_final)
    accuracy = weighted_accuracy(TP_final, FP_final, FN_final, TN_final)
    F1 = f1_measure(precisions, recalls)
    print('recalls', recalls)
    print('accuracy', accuracy)
    print('precisions', precisions)
    print('F1', F1)
    print('FP_Rate', FP_Rate)
    print('FN_Rate', FN_Rate)  

    return  


def precision_FC(TP, FP, FN):
    include = (TP + FP + FN)
    prec = (TP) / (TP + FP)
    return prec 

def recall_FC(TP, FP, FN):
    include = (TP + FP + FN)
    rec = (TP) / (TP + FN)
    return rec 

smooth = np.finfo (float).eps
def weighted_accuracy_FC(TP, FP, FN, TN):
    include = (TP + FP + FN)
    acc = (TP + TN) / (TP + FP + FN + TN)
    return acc 

def f1_measure_FC(prec, rec):
    return (2 * prec * rec + smooth) / (prec + rec + smooth)


if __name__=="__main__":
	main()
