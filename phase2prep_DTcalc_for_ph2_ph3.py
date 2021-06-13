#!/home/users/nikela/tensorflow-env/bin/python
import os
import sys

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
from nltk import FreqDist

def phase2_prep(filename):

    print('....read phase1_labeled........')
    phase1_labeled = pd.read_csv(filename) #pd.read_csv('/various/mtzortzi/from_nikela/phase1_labeled.csv')
    print('phase1_labeled shape', phase1_labeled.shape)
    print('DONE')

    # Remove the rows from the dataframe corresponding to Safe labels.
    indexNames = phase1_labeled[phase1_labeled['LABELS'] == "SAFE"].index
    phase1_nodewise_UE = phase1_labeled.drop(indexNames, inplace=False)
    print('phase1_nodewise_UE shape', phase1_nodewise_UE.shape)


    # Now if I'm guessing correct in the real dataset I have to find the Error Labels in each node, corresponding to node failure, 
    # such as node_shutdown, node_unavailable. After finding these phrases (of each node) create the failure chain. - 
    # How many failure chains in each node?
    # For the next phase I'm taking only phrases which are forming the failure chains I found above. 
    # I know also the timestamps of those phrases, so I compute the time differences between phrases in the failure chain to 
    # enable lead time prediction.
    # The manifested failure is indicated by the higher order time-series.
    # I remove nodes with 1 row of messages.

    '''
    list4 = []
    for key, msg in phase1_nodewise_UE.groupby(['LOCATION']):
        #print("the group for Location node '{}' has {} rows".format(key,len(msg))) 
        #print(msg['MULTIWORDS'])
        if len(msg)>2:
            #print(msg)
            list4.append(msg)
    print('--------------------------------------------------------------------')
    phase1_nodewise_UE = pd.concat(list4)
    '''

    # I remove the nodes which don't have Error messages (Error or Indicator Error), meaning they contain only Unknown messages.
    list5 = []
    for key, msg in phase1_nodewise_UE.groupby(['LOCATION']):
        #print("the group for Location node '{}' has {} rows".format(key,len(msg))) 
        #print(msg['MULTIWORDS'])
        if msg['LABELS'].str.contains('ERROR').any() or msg['LABELS'].str.contains('INDICATOR_ERROR').any():
            #print(msg['MULTIWORDS'])
            #print(msg['LABELS'])
            list5.append(msg)
    print('---------------------------------------------------------------------')
    phase1_nodewise_UE = pd.concat(list5)   

    # I remove nodes which don't have any Error messages INDICATING node failure. !!!!!!!!
    list6 = []
    for key, msg in phase1_nodewise_UE.groupby(['LOCATION']):
        #print("the group for Location node '{}' has {} rows".format(key,len(msg))) 
        #print(msg['MESSAGE'], msg['LABELS'])
        if msg['LABELS'].str.contains('INDICATOR_ERROR').any():
            list6.append(msg)
        else:
            continue

    # I have created a dataframe phase containing only nodes with failure chains.
    phase = pd.concat(list6)

    if filename == '/various/mtzortzi/from_nikela/ASCENDING_ORDER_METHOD/phase1_labeled.csv':
        phase.to_csv('/various/mtzortzi/from_nikela/ASCENDING_ORDER_METHOD/phase1_nodewise_UE_fc.csv', index=False, sep=',', quoting=csv.QUOTE_ALL)
        print('DONE WITH phase1_nodewise_UE_fc which contains only nodes with failure chains')
    elif filename == '/various/mtzortzi/from_nikela/ASCENDING_ORDER_METHOD/DESCENDING_ORDER_trains/anwesha_labeled.csv':
        phase.to_csv('/various/mtzortzi/from_nikela/ASCENDING_ORDER_METHOD/DESCENDING_ORDER_trains/anwesha_pre2.csv', index=False, sep=',', quoting=csv.QUOTE_ALL)
        print('DONE WITH anwesha_pre2 which contains only nodes with failure chains')
    
    return    


def DT_calculation(filename):

    print('....read phase1_nodewise_UE_fc(for phase2) or phase3........')
    phase_df = pd.read_csv(filename) #pd.read_csv('/various/mtzortzi/from_nikela/phase1_nodewise_UE_fc.csv') for phase2
    print('phase_df shape', phase_df.shape)
    print('DONE')

    # I want to create the DT column. 
    # For this I assign every row containing messages Indicators to node failure with 0 in the new column DT and 
    # all the other rows with NaN values.

    for idx, msg in phase_df.iterrows():
        if msg.LABELS == 'INDICATOR_ERROR':
            phase_df.loc[idx, 'DT'] = 0

    # I convert EVENT_TIME to seconds and I save it to a new column in my dataframe, called seconds.

    total_second_list = []
    for tmstmp in phase_df['EVENT_TIME']:
        #print(tmstmp)
        try:
            d = datetime.strptime(tmstmp, "%Y-%m-%d %H:%M:%S.%f")
        except:
            tmstmp = tmstmp + ".0"
            d = datetime.strptime(tmstmp, "%Y-%m-%d %H:%M:%S.%f")
        total_seconds = time.mktime(d.timetuple())
        total_seconds = int(total_seconds)
        total_second_list.append(total_seconds)
        #print(total_seconds)
    phase_df['seconds'] = total_second_list

    # At first I change the order from ascending to descending based on timestamps - it's easier to extract the DTs
    list = []
    for key, msg in phase_df.groupby(['LOCATION']):
    #     print("the group for Location node '{}' has {} rows".format(key,len(msg)))
    #     print(msg['EVENT_TIME'])
        msg.sort_values('EVENT_TIME', inplace=True, ascending=False)
        list.append(msg)
    #     print(msg['EVENT_TIME'])
    phase_df = pd.concat(list)

    # I compute the DTs subtracting timestamps of every phrase from the highest timestamped phrase (of course this is in a 
    # node with only one failure chain). Here I find the subtractions for every failure chain in each node.

    bigger_list = []
    for key, node_df in phase_df.groupby(['LOCATION']):
        keeper = 11111111111111111111111111111111111111111111111111111111111111
        #print(type(node_df))
        #print(node_df)
        temp_list = []
        for idx, msg in node_df.iterrows():
            #print(msg)
            #print('another one')
            if msg.DT == 0:
                temp_list.append(0)
                keeper = idx
                Time_keeper = msg.seconds
                print('keeper', keeper)
            elif keeper != 11111111111111111111111111111111111111111111111111111111111111 :
                print('Time_keeper - msg.seconds', Time_keeper - msg.seconds)
                #print('node_df.loc[idx]', node_df.loc[idx])
                temp_list.append(Time_keeper - msg.seconds)
                #msg.loc[idx, 'DT'] = Time_keeper - msg.seconds
            else:
                temp_list.append(11111111111)
                continue
        bigger_list.append(temp_list)
        print('---------------------')    


    # bigger_list is a list of lists containing for every node each DTs per every failure chain be contained there. 
    # I have marked some DTs with 11111111111, this is because some chains had NON-indicators (U/E) for node shutdown BEFORE the indicator. 
    # That of course would be "wrong", because we want the highest timestamped phrase in the sequence as the indicator. 
    # We assume or cut the sequences so as to be according to the order: IndicatorE, U/E, U/E, ...

    # I flatten the bigger_list so as to create the final DT column in my dataframe phase1_nodewise_UE_fc.

    flat_list = [item for sublist in bigger_list for item in sublist]
    #print(phase_df.head(2))
    phase_df['DT'] = flat_list
    #print(phase_df.head(2))

    print(set(phase_df['MULTIWORDS']))
    freq_dist_pos = FreqDist(phase_df['MULTIWORDS'])
    print(freq_dist_pos.most_common(5))
    print(len(phase_df['MULTIWORDS']))

    '''
    #I reorder the dataset again from descending order to ascending order - timestamps
    list = []
    for key, msg in phase_df.groupby(['LOCATION']):
    #     print("the group for Location node '{}' has {} rows".format(key,len(msg)))
    #     print(msg['EVENT_TIME'])
        msg.sort_values('EVENT_TIME', inplace=True, ascending=True)
        list.append(msg)
    #     print(msg['EVENT_TIME'])
    phase_df = pd.concat(list)
    '''

    if filename == '/various/mtzortzi/from_nikela/ASCENDING_ORDER_METHOD/phase1_nodewise_UE_fc.csv':
        phase_df.to_csv('/various/mtzortzi/from_nikela/ASCENDING_ORDER_METHOD/phase2.csv', index=False, sep=',', quoting=csv.QUOTE_ALL)
    elif filename == '/various/mtzortzi/from_nikela/ASCENDING_ORDER_METHOD/test70_ndw_labeled.csv':
        phase_df.to_csv('/various/mtzortzi/from_nikela/ASCENDING_ORDER_METHOD/phase3.csv', index=False, sep=',', quoting=csv.QUOTE_ALL)
    elif filename == '/various/mtzortzi/from_nikela/ASCENDING_ORDER_METHOD/DESCENDING_ORDER_trains/anwesha_pre2.csv':
        phase_df.to_csv('/various/mtzortzi/from_nikela/ASCENDING_ORDER_METHOD/DESCENDING_ORDER_trains/phase2_anwesha.csv', index=False, sep=',', quoting=csv.QUOTE_ALL)        

    return 
