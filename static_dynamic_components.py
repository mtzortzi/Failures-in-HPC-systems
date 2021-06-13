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


merged = pd.read_csv('/various/mtzortzi/LogAider-master/RAS-JOB/merged.csv')
print('merged shape', merged.shape)
print('merged head', merged.head(1))

uniques = merged.MESSAGE.unique()
print('uniques len', len(uniques))

def split_text(text):
    if 'Detected that one of the DCAs on this board has experienced a Domain 1 power failure.' in text:
        static, dynamic = text.split('.', 1)
    elif "Bumped this DCA's domain 1 voltage up by DefaultDomain1VoltageBump mV." in text:
        static, dynamic = text.split('.', 1)
    elif 'A BQL double bit error threshold was exceeded for' in text:
        static, dynamic = text.split('for', 1)
    elif 'bootBlock: boot failed for' in text:
        static, dynamic = text.split('for', 1)
    elif 'CFAM Machine Check.' in text:
        static, dynamic = text.split('.', 1)
    elif 'CFAM Live Lock Buster Failure.' in text:
        static, dynamic = text.split('.', 1)
    elif 'A connection is taking a   to complete.' in text:
        static, dynamic = text.split('.', 1)
    elif 'CFAM Recoverable Error' in text:
        static, dynamic = text.split('.', 1)
    elif 'Training failure detected by the Torus logical' in text:
        static, dynamic = text.split('by', 1)  
    elif 'Link failure detected between nodes connected via copper and optical links.' in text:
        static, dynamic = text.split('.', 1)
    elif 'Link failure detected between nodes connected via copper links.' in text:
        static, dynamic = text.split('.', 1)
    elif "Encountered an exception while servicing this compute's mailbox." in text:
        static, dynamic = text.split('while', 1)
    elif 'The broadcast install of a kernel image failed,  domain[0] rc' in text:
        static, dynamic = text.split(',', 1)
    elif 'Marking' in text:
        static, dynamic = text.split('Marking', 1)
    elif 'killing job' in text:
        static = 'killing job'
        dynamic = 'not important dynamic'
    elif 'Kernel unexpected operation.' in text:
        static, dynamic = text.split('.', 1)
    elif 'Kernel Internal assertion failure.' in text:
        static, dynamic = text.split('.', 1)
    elif 'Successfully reset this card (' in text:
        static, dynamic = text.split('(', 1)
    elif 'A BQL  bit error threshold was exceeded but sparing is not possible.' in text:
        static, dynamic = text.split('.', 1)
    elif 'A BQL  bit error threshold was exceeded.' in text:
        static, dynamic = text.split('.', 1)
    elif 'A BQL lane was spared' in text:
        static, dynamic = text.split('.',1 )
    elif 'Block failed to boot.' in text:
        static, dynamic = text.split('.',1 )
    elif 'Unrecoverable Machine Check.' in text:
        static, dynamic = text.split('.', 1)
    elif 'Verification of the kernel shutdown failed.' in text:
        static, dynamic = text.split('.', 1)
    elif 'Detected that this board has become unusable (due to invalid power rail voltages)' in text:
        static, dynamic = text.split('(', 1)
    elif 'Detected that this board has become unusable' in text:
        static = text
        dynamic = 'no dynamic'
    elif 'PLL did not lock' in text:
        static = text
        dynamic = 'no dynamic'
    elif 'CNK detected a NULL IPI target function pointer.' in text:
        static, dynamic = text.split('.', 1)
    elif 'Serdes link failure.' in text:
        static, dynamic = text.split('.', 1)
    elif 'CNK Unexpected GEA Interrupt' in text:
        static = text
        dynamic = 'no dynamic'
    elif 'Mailbox verification failed for this node' in text:
        static = text
        dynamic = 'no dynamic'
    elif 'The install of a kernel image failed,' in text:
        static, dynamic = text.split(',', 1)
    elif 'Send of a kernel shutdown message failed.  Return code' in text:
        static, dynamic = text.split('.', 1)
    elif 'send of a kernel shutdown message failed.  return code' in text:
        static, dynamic = text.split('.', 1)
    elif 'an attempt was made to execute a boot step on a board that is not initialized, since the board is unavailable we are failing this boot step.' in text:
        static = 'since the board is unavailable we are failing this boot step' 
        dynamic = 'no dynamic'
    elif 'An attempt was made to execute a boot step on a board that is not initialized, since the board is unavailable we are failing this boot step. ' in text:
        static =  'since the board is unavailable we are failing this boot step'
        dynamic = 'no dynamic'
    elif 'bootBlock: boot failed ' in text:
        static = 'boot failed'
        dynamic = 'no dynamic'
    elif 'ACCESS alert. Message' in text:
        static, dynamic = text.split('.', 1)
    elif "Unable to read card's environmental data" in text:
        static = 'Unable to read card environmental data'
        dynamic = 'no dynamic'
    elif 'Cable from' in text:
        static, dynamic = text.split('from', 1)
    elif "Encountered an exception while servicing this compute's mailbox" in text:
        static, dynamic = text.split('this', 1)
    elif 'The node sent an unexpected mailbox command, MbCmd' in text:
        static, dynamic = text.split(',', 1)
    elif 'CNK Unexpected MU or ND interrupt.  ND NFatal' in text:
        static, dynamic = text.split('.', 1)
    elif 'READ[' in text:
        static, dynamic = text.split('[', 1)
    elif 'ADDR[' in text:
        static, dynamic = text.split('[', 1)
    elif 'MRKSTDTA[' in text:
        static, dynamic = text.split('[', 1)
    elif ':' in text:
        static, dynamic = text.split(':', 1)
    elif '-' in text:
        static, dynamic = text.split('-', 1)
    elif '=' in text:
        static, dynamic = text.split('=', 1)
    elif '.' in text:
        static, dynamic = text.split('.', 1)
    else:
        static = text
        dynamic = 'no dynamic'
        print(text)
    return static
merged['STATIC'] = merged['MESSAGE'].apply(split_text)  
# logs['DYNAMIC'] = logs['MESSAGE'].apply(split_text)

un = merged['STATIC'].unique()
print('how many unique static components', len(un))
print('the unique static components', un)
print('merged new with the static component', merged)

merged.to_csv('/various/mtzortzi/LogAider-master/RAS-JOB/merged_static.csv', index=False, sep=',', quoting=csv.QUOTE_ALL)
