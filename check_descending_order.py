#!/home/users/nikela/tensorflow-env/bin/python
import os
import sys


import pandas as pd
import csv
import pickle
import numpy as np
import io
import fileinput
import random
import math
from math import log
import time
from datetime import datetime
import scipy.special
import scipy.stats

import faulthandler
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
from nltk import FreqDist
import fileinput
import io 
from tqdm import tqdm
import re
import tensorflow as tf 
from tensorflow.python.client import device_lib 


#import train30_test70_nodewise_ascending_order
#import phase2prep_DTcalc_for_ph2_ph3


def main():

    print('....read phase3........')
    phase_df = pd.read_csv('/various/mtzortzi/from_nikela/ASCENDING_ORDER_METHOD/phase3.csv')
    print('phase2 shape', phase_df.shape)
    print('DONE')

    #I reorder the dataset having descending timestamps
    list = []
    for key, msg in phase_df.groupby(['LOCATION']):
    #     print("the group for Location node '{}' has {} rows".format(key,len(msg)))
    #     print(msg['EVENT_TIME'])
        msg.sort_values('EVENT_TIME', inplace=True, ascending=False)
        list.append(msg)
    #     print(msg['EVENT_TIME'])
    phase_df = pd.concat(list)

    phase_df.to_csv('/various/mtzortzi/from_nikela/ASCENDING_ORDER_METHOD/DESCENDING_ORDER_trains/phase3.csv', index=False, sep=',', quoting=csv.QUOTE_ALL)

    return

if __name__=="__main__":
    main()
