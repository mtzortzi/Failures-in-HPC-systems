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
from numpy import array
from sklearn.model_selection import train_test_split
import re
import matplotlib.pyplot as plt


def phase1_nodewise(filename):
    #train
    train = pd.read_csv(filename) #pd.read_csv('/various/mtzortzi/from_nikela/DIVIDE6040/train60.csv')
    '''
    I create a dataframe which is node wise, meaning that logs from each node are concatenated and fed to the same LSTM. 
    Desh learns failure sequences from different nodes sequentially.

    Here I keep the nodes containing (>=3) 3 messages and up, this is because skip gram wasn't able to trained 
    for nodes containing 1 or 2 messages only, there was an Internal Error as far as gpu concerns.
    '''

    list1 = []
    for key, msg in train.groupby(['LOCATION']):
        print("the group for Location node '{}' has {} rows".format(key,len(msg))) 
        if len(msg)>2:
            list1.append(msg)
    train_nodewise = pd.concat(list1)

    '''
    Sort the EVENT_TIME column (timestamps) on descending order for each node group in train_nodewise dataframe. 
    '''
    list2 = []
    for key, msg in train_nodewise.groupby(['LOCATION']):
    #     print("the group for Location node '{}' has {} rows".format(key,len(msg))) 
    #     print(msg['EVENT_TIME'])
        msg.sort_values('EVENT_TIME', inplace=True, ascending=False)
        list2.append(msg)
    #     print(msg['EVENT_TIME'])
    phase1_nodewise = pd.concat(list2)

    from nltk import FreqDist

    freq_dist_pos = FreqDist(phase1_nodewise['MULTIWORDS'])
    print(freq_dist_pos.most_common(5))
    print(len(phase1_nodewise['MULTIWORDS']))

    phase1_nodewise.to_csv('/various/mtzortzi/from_nikela/DIVIDE6040/phase1_nodewise.csv', index=False, sep=',', quoting=csv.QUOTE_ALL)
    print('DONE WITH phase1_nodewise')
    return
    

def test_nodewise(filename):
    #test
    test = pd.read_csv(filename) #pd.read_csv('/various/mtzortzi/from_nikela/DIVIDE6040/test40.csv')

    list1 = []
    for key, msg in test.groupby(['LOCATION']):
    #     print("the group for Location node '{}' has {} rows".format(key,len(msg))) 
        if len(msg)>2:
            list1.append(msg)
    test_nodewise = pd.concat(list1)

    #Sort the EVENT_TIME column (timestamps) on descending order for each node group in test_nodewise dataframe. 
    list2 = []
    for key, msg in test_nodewise.groupby(['LOCATION']):
    #     print("the group for Location node '{}' has {} rows".format(key,len(msg))) 
    #     print(msg['EVENT_TIME'])
        msg.sort_values('EVENT_TIME', inplace=True, ascending=False)
        list2.append(msg)
    #     print(msg['EVENT_TIME'])
    test_nodewise = pd.concat(list2)

    from nltk import FreqDist

    freq_dist_pos = FreqDist(test_nodewise['MULTIWORDS'])
    print(freq_dist_pos.most_common(5))
    print(len(test_nodewise['MULTIWORDS']))
    print('DONE with test_nodewise')
    test_nodewise.to_csv('/various/mtzortzi/from_nikela/DIVIDE6040/test40_nodewise.csv', index=False, sep=',', quoting=csv.QUOTE_ALL)
    return