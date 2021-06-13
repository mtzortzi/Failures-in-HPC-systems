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

import phase2prep_DTcalc_for_ph2_ph3

def main():

    print('.........phase2 prep I keep only failure chains containing UE............')
    # it creates the phase1_nodewise_UE_fc.csv 
    phase2prep_DTcalc_for_ph2_ph3.phase2_prep('/various/mtzortzi/from_nikela/ASCENDING_ORDER_METHOD/DESCENDING_ORDER_trains/anwesha_labeled.csv')
    print('DONE')

    print('........DT_calculation for phase2........')
    # it creates the phase2.csv 
    phase2prep_DTcalc_for_ph2_ph3.DT_calculation('/various/mtzortzi/from_nikela/ASCENDING_ORDER_METHOD/DESCENDING_ORDER_trains/anwesha_pre2.csv')
    print('DONE')    

if __name__=="__main__":
	main()

