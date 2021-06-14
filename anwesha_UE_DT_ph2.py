
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

import anwesha_prep2_DT

def main():

    print('.........create anwesha_pre2.csv I keep only failure chains containing UE............')
    # it creates the anwesha_pre2.csv
    anwesha_prep2_DT.phase2_prep('/various/mtzortzi/from_nikela/anwesha_labeled.csv')
    print('DONE')

    print('........DT_calculation for anwesha phase2........')
    # it creates the phase2_anwesha.csv 
    anwesha_prep2_DT.DT_calculation('/various/mtzortzi/from_nikela/anwesha_pre2.csv')
    print('DONE')    

if __name__=="__main__":
	main()

