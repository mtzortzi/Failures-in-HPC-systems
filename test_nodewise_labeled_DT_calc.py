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


import train30_test70_nodewise_descending_order
import phase2prep_DTcalc_for_ph2_ph3

def main():
    def labeling3(text):
        # safe
        if 'read' in text: 
            label_list.append('SAFE')
        elif 'ddr0phywasrecalibrated0' in text:
            label_list.append('SAFE')
        elif 'ddr0phywasrecalibrated4' in text:
            label_list.append('SAFE')
        elif 'ddr1phywasrecalibrated0' in text:
            label_list.append('SAFE')
        elif 'ddr1phywasrecalibrated4' in text:
            label_list.append('SAFE')
        elif 'ddr1phywasrecalibrated1' in text:
            label_list.append('SAFE')
        elif 'ddrcorrectableerrorsummary' in text:
            label_list.append('SAFE')
        elif 'l1pcorrectableerrorsummary' in text:
            label_list.append('SAFE')
        elif 'l2arraycorrectableerrorsummary' in text:
            label_list.append('SAFE')
        elif 'l2directorycorrectableerrorsummary' in text:
            label_list.append('SAFE')
        elif 'messageuniteccsummary' in text:
            label_list.append('SAFE')
        elif 'ndreceivercorrectableerror' in text:
            label_list.append('SAFE')
        elif 'ndsenderretransmissioncorrectableerror' in text:
            label_list.append('SAFE')
        elif 'sat1' in text:
            label_list.append('SAFE')
        elif 'sat0' in text:
            label_list.append('SAFE')
        elif 'addr' in text:
            label_list.append('SAFE')
        elif 'ddr1ue' in text:
            label_list.append('SAFE')
        elif 'successfullyresetthiscard' in text:
            label_list.append('SAFE')
        elif 'mrkstdta' in text:
            label_list.append('SAFE')
        elif 'zero' in text:
            label_list.append('SAFE')
    
        #unknown
        elif 'a2tlbparityerror' in text:
            label_list.append('UNKNOWN')
        elif 'ddrarbitermachinecheckrecoverable' in text:
            label_list.append('UNKNOWN')
        elif 'detectedthatoneofthedcasonthisboardhasexperiencedadomain1powerfailure' in text:
            label_list.append('UNKNOWN')
        elif 'memorycontrollerinitializationwarning' in text:
            label_list.append('UNKNOWN')
        elif 'rasstormwarning' in text:
            label_list.append('UNKNOWN')
        elif 'ndcorrectableerror' in text:
            label_list.append('UNKNOWN')
        elif 'cfamrecoverableerror' in text:
            label_list.append('UNKNOWN')
        elif 'opticalmoduleenvironmentaldataisunavailable' in text:
            label_list.append('UNKNOWN')
        elif 'aconnectionistakingatocomplete' in text:
            label_list.append('UNKNOWN')
        elif 'verificationofthekernelshutdownfailed' in text:
            label_list.append('UNKNOWN')
        elif 'abqldoublebiterrorthresholdwasexceeded' in text:
            label_list.append('UNKNOWN')
        elif 'warningmc1rank1' in text:
            label_list.append('UNKNOWN')
        elif 'warningmc0rank1' in text:
            label_list.append('UNKNOWN')
        elif 'warningmc1rank0' in text:
            label_list.append('UNKNOWN')
        elif 'warningmc0rank0' in text:
            label_list.append('UNKNOWN')
        elif 'messageunitrecoverableerror' in text:
            label_list.append('UNKNOWN') 
        elif 'abqlbiterrorthresholdwasexceeded' in text:
            label_list.append('UNKNOWN') 
        elif 'plldidnotlock' in text:
            label_list.append('UNKNOWN')
        elif 'abqllanewasspared' in text: #A BQL lane was spared
            label_list.append('UNKNOWN')       
        elif 'abqlbiterrorthresholdwasexceededbutsparingisnotpossible' in text:
            label_list.append('UNKNOWN')

        #fatal_error
        elif 'accessalert' in text: #fatal_end_job
            label_list.append('INDICATOR_ERROR')
        elif 'a2processormachinecheck' in text: #fatal_end_job
            label_list.append('INDICATOR_ERROR')
        elif 'cfamlivelockbusterfailure' in text: #fatal_end_job
            label_list.append('INDICATOR_ERROR')
        elif 'cfammachinecheck' in text: #fatal_end_job
            label_list.append('INDICATOR_ERROR')
        elif 'cnkunexpectedmuorndinterrupt' in text: #fatal_end_job 
            label_list.append('INDICATOR_ERROR')
        elif 'ddrarbitermachinecheck' in text: #fatal_end_job 
            label_list.append('INDICATOR_ERROR')
        elif 'detectedapowerrailwithanincorrectvoltage' in text: #fatal_end_job 
            label_list.append('INDICATOR_ERROR')
        elif 'detectedthatthisboardhasbecomeunusable' in text: #fatal_end_job 
            label_list.append('INDICATOR_ERROR')
        elif 'devbusmachinecheck' in text: #fatal_end_job 
            label_list.append('INDICATOR_ERROR')
        elif 'kernelinternalassertionfailure' in text: #fatal_end_job 
            label_list.append('INDICATOR_ERROR')
        elif 'kernelunexpectedoperation' in text: #fatal_end_job
            label_list.append('INDICATOR_ERROR')
        elif 'l1pmachinecheck' in text: #fatal_end_job
            label_list.append('INDICATOR_ERROR')
        elif 'l2machinecheck' in text: #fatal_end_job
            label_list.append('INDICATOR_ERROR')
        elif 'linkfailuredetectedbetweennodesconnectedviacopperandopticallinks' in text: #fatal_end_job
            label_list.append('INDICATOR_ERROR')
        elif 'linkfailuredetectedbetweennodesconnectedviacopperlinks' in text: #fatal_end_job
            label_list.append('INDICATOR_ERROR')
        elif 'memorycontrollerinitializationerror' in text: #fatal_end_job
            label_list.append('INDICATOR_ERROR')
        elif 'rasstormerror' in text: #fatal_end_job
            label_list.append('INDICATOR_ERROR')
        elif 'trainingfailuredetected' in text: #fatal_end_job
            label_list.append('INDICATOR_ERROR')
        elif 'unrecoverablemachinecheck' in text: #fatal_end_job
            label_list.append('INDICATOR_ERROR')

        #error
        elif 'alinkchipdidnotbitalignalongthecport' in text:
            label_list.append('ERROR')
        elif 'alinkchipdidnotbitalignalongthereceivercport' in text:
            label_list.append('ERROR')
        elif 'cable' in text:
            label_list.append('ERROR')
        elif 'theinstallofakernelimagefailed' in text:
            label_list.append('ERROR')
        elif 'thetransmittingnodedidnotalignwithlinkchipr22' in text:
            label_list.append('ERROR')
        elif 'thebroadcastinstallofakernelimagefailed' in text:
            label_list.append('ERROR')
        elif 'sincetheboardisunavailablewearefailingthisbootstep' in text: #An attempt was made to execute a boot step on a board that is not initialized
            label_list.append('ERROR')
        elif 'sendofakernelshutdownmessagefailed' in text:
            label_list.append('ERROR')
        elif 'badphywasdetected' in text:
            label_list.append('ERROR')
        elif 'baddramwasdetected' in text:
            label_list.append('ERROR')
        elif 'unabletoreadcardenvironmentaldata' in text:
            label_list.append('ERROR') 
        elif 'ndreceiverlinkerror' in text:
            label_list.append('ERROR')

        #only in phase3
        elif 'encounteredanexception' in text:
            label_list.append('phase3')
        elif 'tvsensetemperatureisunavailable' in text:
            label_list.append('phase3')
        elif 'cnkdetectedanullipitargetfunctionpointer' in text:
            label_list.append('phase3')
        elif 'serdeslinkfailure' in text:
            label_list.append('phase3')
        elif 'cnkunexpectedgeainterrupt' in text:
            label_list.append('phase3')
        elif 'ddr0ue' in text:
            label_list.append('phase3')
        elif 'pllproblemonbqcchip' in text:
            label_list.append('phase3')
        elif 'thenodesentanunexpectedmailboxcommand' in text:
            label_list.append('phase3')
        elif 'unabletoupdatethelcddisplay' in text:
            label_list.append('phase3')
        elif 'serviceaction6585completedonr13' in text:
            label_list.append('phase3')
        elif 'successfullyperformedthespecifiedoperationonthisdca' in text:
            label_list.append('phase3')
        elif 'enabledthisdca' in text:
            label_list.append('phase3')
        elif 'disabledthisdca' in text:
            label_list.append('phase3')
        elif 'serviceaction6585restartednodedcar13' in text:
            label_list.append('phase3')
        elif 'serviceaction6585turnednodedcar13' in text:
            label_list.append('phase3')
        elif 'serviceaction6585startedtoservicer13' in text:
            label_list.append('phase3') 
        elif 'illegaldcraccess' in text:
            label_list.append('phase3')
        elif 'ndfatalerror' in text:
            label_list.append('phase3')
        elif 'mailboxverificationfailedforthisnode' in text:
            label_list.append('phase3')
        elif 'dcrarbitermachinecheck' in text:
            label_list.append('phase3')
        elif 'thebqcclocksarenotinthecorrectstate' in text:
            label_list.append('phase3')
        elif 'checkforsuccessofporsequencefailed' in text:
            label_list.append('phase3')
        elif 'alinkchipdidnotalignalongtheaportonswitch2' in text:
            label_list.append('phase3')
        elif 'alinkchipdidnotalignalongtheaportonswitch1' in text:
            label_list.append('phase3')
        elif 'alinkchipdidnotalignalongtheaportonswitch0' in text:
            label_list.append('phase3')
        return

    
    print('....create test70_nodewise with descending order....')
    #I have to load the outputs from train30_test70_nodewise_descending_order.py so as to have test70_nodewise work.
    train30_test70_nodewise_descending_order.test70_nodewise('/various/mtzortzi/from_nikela/test70.csv')
    print('DONE')
    
    print('....read test70_nodewise........')
    test70_nodewise = pd.read_csv('/various/mtzortzi/from_nikela/test70_nodewise.csv')
    print('test70 nodewise shape', test70_nodewise.shape)
    print('DONE')

    print('...........unique multiwords in test70_nodewise.....................')
    uniques = test70_nodewise.MULTIWORDS.unique()
    print('uniques', uniques)
    print('uniques len', len(uniques))
    print('DONE')

    print('.......start phrase labeling...........')
    label_list = []
    for msg in test70_nodewise['MULTIWORDS']:
        labeling3(msg)
    test70_nodewise['LABELS'] = label_list
    
    test70_nodewise.to_csv('/various/mtzortzi/from_nikela/test70_ndw_labeled.csv', index=False, sep=',', quoting=csv.QUOTE_ALL)
    print('DONE')

    print('......DT_calculation for phase3......')
    # it creates the phase3.csv 
    phase2prep_DTcalc_for_ph2_ph3.DT_calculation('/various/mtzortzi/from_nikela/test70_ndw_labeled.csv')
    print('DONE')


if __name__=="__main__":
    main()
