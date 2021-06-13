from keras import backend as K
import tensorflow as tf
import numpy as np

import pandas as pd
import csv
import pickle
import numpy as np
import io
import fileinput
import random
import math
import time

smooth = np.finfo (float).eps


def confusion_matrix(y_true, y_pred):
    #y_pred = (y_pred > 0.5).astype(float)
    TP = y_pred * y_true
    #print('inside confusion matrix TP', TP)
    FP = y_pred - TP
    FN = y_true - TP
    #print('inside confusion matrix FP', FP)
    TN = np.ones(y_true.shape) - TP - FP - FN
    return np.sum(TP, axis=0), np.sum(FP, axis=0), np.sum(FN, axis=0), np.sum(TN, axis=0)

def precision(TP, FP, FN):
    include = (TP + FP + FN).astype(bool)
    prec = (TP + smooth) / (TP + FP + smooth)
    return np.mean(prec * include)

def recall(TP, FP, FN):
    include = (TP + FP + FN).astype(bool)
    rec = (TP + smooth) / (TP + FN + smooth)
    return np.mean(rec * include)

def weighted_accuracy(TP, FP, FN, TN):
    include = (TP + FP + FN).astype(bool)
    acc = (TP + TN + smooth) / (TP + FP + FN + TN + smooth)
    return np.mean(acc * include)

def f1_measure(prec, rec):
    return (2 * prec * rec + smooth) / (prec + rec + smooth)


def jaccard_coef(y_true, y_pred):
    # __author__ = Vladimir Iglovikov
    intersection = K.sum(y_true * y_pred, axis=[0, -1, -2])
    sum_ = K.sum(y_true + y_pred, axis=[0, -1, -2])

    jac = (intersection + smooth) / (sum_ - intersection + smooth)

    return K.mean(jac)


def jaccard_coef_int(y_true, y_pred):
    # __author__ = Vladimir Iglovikov
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))

    intersection = K.sum(y_true * y_pred_pos, axis=[0, -1, -2])
    sum_ = K.sum(y_true + y_pred, axis=[0, -1, -2])
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return K.mean(jac)
