#region PARAMETERS
from settings import *

#region Argparser
import argparse

# OBS: Supress needed here! It stops value from being defaulted
parser = argparse.ArgumentParser(description='Keras based program that predicts Signal Peptides')
parser.add_argument('-i', dest='folder_destination', required=False,
                    help='Folder in query_data', default=argparse.SUPPRESS)
args = parser.parse_args()

print(vars(args))
for key in vars(args).keys():  # Creates dict
    vars()[key] = vars(args)[key]  # Overwrites config file


#region Logging start
import logging
import os
try:
    os.remove('results/query_logger.log')
except:
    pass  # First run, no logger to clear
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
handler = logging.FileHandler('results/query_logger.log')
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.info('-' * 20 + 'START OF RUN' + '-' * 20)
logger.info(str(locals()))

#region  MODULES
"""CONSIDER TO IMPORT WHEN NEEDED AND HAVE OPTIONAL LIBRARIES TO PREVENT BREAKAGE"""
"""Reason why not all modules are imported at start:
TF and Keras takes a long time to load, so I would rather prevent the script from loading at all
if the user has added an error"""
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # Stops tf allocating all memory
session = tf.Session(config=config)
print("Stopped TF from using all memory on gpu (only allocates what's needed now)")

"""I need to set CUDA before imports!!!"""
import os
if not use_gpu:
    """Use gpu if you have many parameters in your model"""
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    logger.info('Using cpu...')
else:
    logger.info('Using gpu...')

print('Importing modules...')
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, Concatenate, Input
from keras.layers import LSTM
from keras.layers import Convolution1D, Convolution2D
from keras.layers import MaxPooling1D, MaxPooling2D
from keras.layers import UpSampling1D, UpSampling2D
from keras.layers import Bidirectional
from keras.layers import LeakyReLU
from keras.layers import BatchNormalization
from keras.layers import Reshape
from keras.layers import GaussianNoise
from keras.layers import Permute  # Kinda like: I want this shape :)
from keras.callbacks import History, EarlyStopping, ModelCheckpoint
from keras import regularizers
import numpy as np
import time
import matplotlib
matplotlib.use('TkAgg')  # Needed to change plot position while calculating. NEEDS TO ADDED BEFORE pyplot import!!
import matplotlib.pyplot as plt
from keras.models import load_model
import pickle
from sklearn import metrics  # For ROC curve
from keras import backend as K  # for layer viz
import multiprocessing  # For future: make plotting in different processes


########################################################################################################################
#endregion

from scripts.load_data import load_training
X_test, Y_test = load_training(seq_length,
                     query_data_folder,
                     data_augmentation=False,
                     fix_samples=fix_samples,
                     equalize_data=False,
                     save_array=False,
                     vectorize=vectorize)

print('Preprocessing data...')
# Fix so that you can divide
if not vectorize:
    X_test.astype('float64')
    Y_test.astype('float64')

if vectorize:
    pass
elif use_ascii:
    X_test /= 90.
elif not use_ascii:
    X_test /= 20.

amount_negative = len(list(np.where(Y_test == 0.)[0]))
amount_positive = len(list(np.where(Y_test == 1.)[0]))
amount_unknown = len(list(np.where(Y_test == -1.)[0]))
if amount_unknown > 0:
    print('Unknown samples detected, starting prediction mode')
    known_sps = False
else:
    known_sps = True
logger.info(f'Test set: {amount_positive} positive samples, {amount_negative} negative samples and {amount_unknown} unknown samples')
########################################################################################################################
#endregion

#region LOADING MODEL

"""Consider splitting classes into separate files"""

#region Loading model
# Try to load model
try:
    logger.info('Loading model from file...')
    model = load_model('results/Signal_peptide_model.h5')  # Looks like you can load model and keep training
    logger.info('Loaded model: ')
    model.summary(print_fn=lambda txt: logger.info(txt))
except Exception as e:
    logger.info(e)
    logger.info('Cannot load model')
    quit()
#endregion

if not known_sps:
    predictions = model.predict(X_test)
    predictions = np.round(predictions)
    amount_negative = len(list(np.where(Y_test == 0.)[0]))
    amount_positive = len(list(np.where(Y_test == 1.)[0]))
    logger.info(f'Evaluation: {amount_positive} sequences are positive, {amount_negative} are negative')
else:
    #region Evaluation and saving
    logger.info('Evaluation of model processing...')
    scores = model.evaluate(X_test, Y_test)
    logger.info("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    #endregion
    ########################################################################################################################
    #endregion

    #region PLOTS
    """ADD PRINTS BEFORE PLOTTING"""

    #region Confusion matrix
    logger.info('Creating confusion matrix on test samples...')
    predictions = model.predict(X_test)
    predictions = np.round(predictions)

    TP=TN=FP=FN=0
    TN, FP, FN, TP = metrics.confusion_matrix(Y_test, np.round(predictions)).ravel()  # Could rewrite own cm function if I wanted
    confusion_matrix = np.array([[0, 0], [0, 0]])
    confusion_matrix[0][0] = TP
    confusion_matrix[1][1] = TN
    confusion_matrix[1][0] = FP
    confusion_matrix[0][1] = FN
    confusion_matrix_normalized = np.array([[0, 0], [0, 0]], dtype='float32')
    confusion_matrix_normalized[0][0] = TP / (TP + FN)
    confusion_matrix_normalized[1][1] = TN / (FP + TN)
    confusion_matrix_normalized[1][0] = FP / (FP + TN)
    confusion_matrix_normalized[0][1] = FN / (TP + FN)
    plt.figure()
    plt.axis('off')
    plt.title('Confusion matrix normalized')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    heatmap = plt.pcolor(confusion_matrix_normalized, cmap='Blues')
    cm_labels = [['True positive', 'False negative'], ['False positive', 'True negative']]
    '''THIS THING TOOK LIKE A MILLION YEARS TO FIGURE, DONT CHANGE THE DAMN COORDINATES'''
    for x in range(2):
        for y in range(2):
            plt.text(y + 0.5, 1.5 - x, "{0}\n{1:.4f}".format(cm_labels[x][y], confusion_matrix_normalized[x][y]),
                    ha='center', va='center', fontsize=16)
    plt.savefig('results/train_cm_normalized.png')
    if plot_performance:
        plt.show(block=False)
    else:
        plt.clf()

    plt.figure()
    plt.axis('off')
    plt.title('Confusion matrix')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    heatmap = plt.pcolor(confusion_matrix, cmap='Blues')
    for x in range(2):
        for y in range(2):
            plt.text(y + 0.5, 1.5 - x, "{0}\n{1}".format(cm_labels[x][y], confusion_matrix[x][y]),
                    ha='center', va='center', fontsize=16)
    plt.savefig('results/train_cm.png')
    if plot_performance:
        plt.show(block=False)
    else:
        plt.clf()
    #endregion

    #region ROC curve
    """Maybe write own ROC function to get rid of sklearn dependency? Should not be that hard"""
    logger.info('Creating ROC curve')
    plt.figure()
    plt.axis('on')
    plt.minorticks_on()
    plt.grid('on')
    plt.grid(b=True, which='major', color='k', linestyle='-', alpha=0.5)
    plt.grid(b=True, which='minor', color='r', linestyle='-', alpha=0.1)
    predictions = model.predict(X_test)
    fpr, tpr, thresholds = metrics.roc_curve(Y_test, predictions)
    auc_score = metrics.roc_auc_score(Y_test, predictions)
    if 0.95 < auc_score <= 1.0:
        classifier_grade = 'Excellent (S)'
    elif 0.9 < auc_score <= 0.95:
        classifier_grade = 'very good (A)'
    elif 0.8 < auc_score <= 0.9:
        classifier_grade = 'good (B)'
    elif 0.7 < auc_score <= 0.8:
        classifier_grade = 'not so good (C)'
    elif 0.6 < auc_score <= 0.7:
        classifier_grade = 'poor (D)'
    elif 0.5 <= auc_score <= 0.6:
        classifier_grade = 'fail (F)'
    elif auc_score < 0.5:
        classifier_grade = ' worse than a coin flip ()'
    else:
        classifier_grade = ' Error calculating AUC'
    plt.text(0.7, 0.2, 'AUC score: ' + str(round(auc_score,5)) + '\n' + classifier_grade,
             style='italic',
             bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 10})
    plt.title('ROC curve')
    plt.ylabel('False positive rate')
    plt.xlabel('True positive rate')
    plt.plot(fpr, tpr, 'b-', label='Model', )
    plt.plot([0, 1], [0, 1], 'r-', label='Baseline', linestyle='dashed')
    legend = plt.legend(loc=4, shadow=True)
    plt.savefig('results/train_roc.png')
    if plot_performance:
        plt.show(block=False)
    else:
        plt.clf()
    #endregion

    #region Metrics
    logger.info('Calculating metrics')
    accuracy_score = (TN+TP)/(TN+FP+FN+TP) # or use metrics.accuracy_score(Y_test, predictions)
    precision_score = TP/(FP+TP) # or use metrics.precision_score(Y_test, predictions)
    sensitivity_score = TP/(TP+FN) # or use metrics.recall_score(Y_test, predictions)
    specificity_score = TN/(TN+FP)
    logger.info('Accuracy: ' + str(round(accuracy_score, 4))
                + ', Precision: ' + str(round(precision_score, 4))
                + ', Sensitivity: ' + str(round(sensitivity_score, 4))
                + ', Specificity: ' + str(round(specificity_score, 4))) # Log this shit
    #endregions