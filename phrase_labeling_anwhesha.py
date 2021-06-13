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


def labeling_text(text):
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
    elif 'ddr0ue' in text:
        label_list.append('SAFE')
    elif 'serviceaction6585restartednodedcar13' in text:
        label_list.append('SAFE')
    elif 'serviceaction6585startedtoservicer13' in text:
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
    elif 'encounteredanexception' in text:
        label_list.append('UNKNOWN')
    elif 'pllproblemonbqcchip' in text:
        label_list.append('UNKNOWN')
    elif 'thenodesentanunexpectedmailboxcommand' in text:
        label_list.append('UNKNOWN')
    elif 'thebqcclocksarenotinthecorrectstate' in text:
        label_list.append('UNKNOWN')
    elif 'serviceaction4961startedtoservicer20' in text:
        label_list.append('UNKNOWN')
    elif 'serviceaction5516turnednodeboardr1e' in text:
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
    elif 'tvsensetemperatureisunavailable' in text:
        label_list.append('INDICATOR_ERROR')
    elif 'cnkdetectedanullipitargetfunctionpointer' in text:
        label_list.append('INDICATOR_ERROR')
    elif 'serdeslinkfailure' in text:
        label_list.append('INDICATOR_ERROR')
    elif 'ndfatalerror' in text:
        label_list.append('INDICATOR_ERROR')
    elif 'dcrarbitermachinecheck' in text:
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
    elif 'unabletoupdatethelcddisplay' in text:
        label_list.append('ERROR')
    elif 'mailboxverificationfailedforthisnode' in text:
        label_list.append('ERROR')
    elif 'checkforsuccessofporsequencefailed' in text:
        label_list.append('ERROR')
    elif 'alinkchipdidnotalignalongtheaportonswitch2' in text:
        label_list.append('ERROR')
    elif 'alinkchipdidnotalignalongtheaportonswitch1' in text:
        label_list.append('ERROR')
    elif 'alinkchipdidnotalignalongtheaportonswitch0' in text:
        label_list.append('ERROR')
    elif 'alinkchipdidnotalignalongtheaandbportsonswitch3' in  text:
        label_list.append('ERROR')

    '''
    #Some confusing part I thought I should label them like this only in phase3 - but I changed that to normal labeling, of course I keep them all in phase3
    elif 'cnkunexpectedgeainterrupt' in text:
        label_list.append('INDICATOR_ERROR')
    elif 'serviceaction6585completedonr13' in text:
        label_list.append('SAFE')
    elif 'successfullyperformedthespecifiedoperationonthisdca' in text:
        label_list.append('SAFE')
    elif 'enabledthisdca' in text:
        label_list.append('SAFE')
    elif 'disabledthisdca' in text:
        label_list.append('ERROR')
    elif 'serviceaction6585turnednodedcar13' in text:
        label_list.append('SAFE')
    elif 'illegaldcraccess' in text:
        label_list.append('UNKNOWN')
    '''
    return



print('....read phase1_nodewise_wids........')
phase1_nodewise_wids = pd.read_csv('/various/mtzortzi/from_nikela/ASCENDING_ORDER_METHOD/DESCENDING_ORDER_trains/anwesha_df.csv')
print('phase1_nodewise_wids shape', phase1_nodewise_wids.shape)
print('DONE')

print('...........unique messages in phase1_nodewise_wids.....................')
uniques = phase1_nodewise_wids.MESSAGE.unique()
print('uniques', uniques)
print('uniques len', len(uniques))
print('DONE')


label_list = []
for msg in phase1_nodewise_wids['MULTIWORDS']:
    labeling_text(msg)
phase1_nodewise_wids['LABELS'] = label_list

phase1_nodewise_wids.to_csv('/various/mtzortzi/from_nikela/ASCENDING_ORDER_METHOD/DESCENDING_ORDER_trains/anwesha_labeled.csv', index=False, sep=',', quoting=csv.QUOTE_ALL)
