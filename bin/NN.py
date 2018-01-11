#region PARAMETERS
from settings import *

########################################################################################################################
#endregion

#region Argparser
import argparse

# OBS: Supress needed here! It stops value from being defaulted
parser = argparse.ArgumentParser(description='Keras based program that predicts Signal Peptides')
parser.add_argument('--use_gpu', dest='use_gpu', required=False, action='store_true',
                    help='Makes your computer use the gpu', default=argparse.SUPPRESS)
parser.add_argument('--use_cpu', dest='use_gpu', required=False, action='store_false',
                    help='Makes your computer use the cpu', default=argparse.SUPPRESS)
args = parser.parse_args()

print(vars(args))
for key in vars(args).keys():  # Creates dict
    vars()[key] = vars(args)[key]  # Overwrites config file

########################################################################################################################
#endregion

#region Logging start
import logging
import os
try:
    os.remove('results/logger.log')
except:
    pass  # First run, no logger to clear
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
handler = logging.FileHandler('results/logger.log')
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.info('-' * 20 + 'START OF RUN' + '-' * 20)
logger.info(str(locals()))

########################################################################################################################
#endregion

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

#region CREATING MODEL

"""Discovered: output shape = seq_length-kernel+1"""
print('Creating model...')
model = Sequential()

if choosen_model == 0:
    # Model 0
    '''lr 0.02 is good for this one.
    Lower lr to 0.01 when getting 0.88 val acc
    Very fast learning'''
    model.add(Convolution1D(254, 20, input_shape=(seq_length, 1), activation='relu'))
    model.add(GaussianNoise(0.01))  # Maybe try with noise to reduce overfit?
    #model.add(Dropout(0.2))
    model.add(MaxPooling1D())
    #model.add(UpSampling1D(5))  # output = 5
    model.add(Convolution1D(128, 5, activation='relu'))
    model.add(Flatten())
    model.add(Dense(30, activation='relu'))
    model.add(Dense(10, activation='sigmoid'))
    model.add(Dense(1, activation='sigmoid'))
elif choosen_model == 1:
    # Model 1
    '''Give the LSTM time, at least 100 epochs before it starts to be good'''
    model.add(Convolution1D(254, 20, input_shape=(seq_length, 1), activation='relu'))
    #model.add(GaussianNoise(0.03))  # Maybe try with noise to reduce overfit?
    #model.add(Dropout(0.2))
    model.add(LSTM(512, return_sequences=True)) # [samples, time_steps, feautures], input_shape=(seq_length, 1)
    model.add(LSTM(512))
    model.add(Dense(1, activation='sigmoid'))
elif choosen_model == 2:
    # Model 2
    '''Needs vectorized data, pure LSTM.
    Give it about 100 epochs to improve
    It seems that dropout is actually not needed here!
    lr = 0.002
    lr_scheduler_plan = {0.88: 0.002, 0.92: 0.001, 0.93: 0.0005} '''
    model.add(Bidirectional(LSTM(64, return_sequences=True), input_shape=(seq_length, 21))) # [samples, time_steps, feautures], input_shape=(seq_length, 1)
    model.add(Dropout(0.5))
    model.add(Bidirectional(LSTM(64))) # [samples, time_steps, feautures], input_shape=(seq_length, 1)
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
elif choosen_model == 3:
    # Model 3
    '''Needs vectorized data combined LSTM and CNN
    lr = 0.001
    lr_scheduler_plan = {0.84: 0.0005, 0.88: 0.0002, 0.92: 0.0001, 0.93: 0.00005} '''
    model.add(Convolution1D(2048, 20, input_shape=(seq_length, 21), activation='relu'))
    model.add(LSTM(512, return_sequences=True)) # [samples, time_steps, feautures], input_shape=(seq_length, 1)
    model.add(LSTM(512))
    model.add(Dense(1, activation='sigmoid'))
else:
    pass  # CUSTOM MODEL


print('Compiling model...')
#Compile model
sgd = keras.optimizers.SGD(lr=lr, decay=decay, momentum=momentum, nesterov=True)
adam = keras.optimizers.Adam(lr=lr)
model.compile(metrics=['binary_accuracy'],
              loss='binary_crossentropy',
              optimizer=adam)

print('Showing model...')
model.summary(print_fn=lambda txt: logger.info(txt))
logger.info('Input shape: ' + str(model.input_shape))
logger.info('Output shape: ' + str(model.output_shape))

"""
Train acc and loss decreases, Test acc and loss increases = overfitting
Solutions: more data, change model, add dropout, add noise, 

Test acc and loss fluctuating = some samples are classified randomly
Solutions: more data, change hyperparameters, add dropout
"""
########################################################################################################################
#endregion

#region DATA
# Prints already in load script
if load_saved_array:
    try:
        from scripts.load_data import load_from_file
        X,Y = load_from_file('storage/loaded_array.npz')
    except:
        logger.info('Cannot load numpy file')
        load_saved_array = False  # If it cant find the file, load the data

if not load_saved_array:
    from scripts.load_data import load_training
    X, Y = load_training(seq_length,
                         data_folder,
                         data_augmentation=data_augmentation,
                         fix_samples=fix_samples,
                         equalize_data=equalize_data,
                         save_array=save_array,
                         vectorize=vectorize)

print('Shuffling data...')
np.random.seed(1) # fix random seed for reproducing results
# SHUFFLE, without it, ROC and AUC doesnt work, bad results etc
s = np.arange(X.shape[0])
np.random.shuffle(s)
X = X[s]
Y = Y[s]
#print(len(np.where(Y==0)[0]))
#print(len(np.where(Y==1)[0]))
# Limit samples
X = X[:limit_samples]
Y = Y[:limit_samples]

print('Preprocessing data...')
# Fix so that you can divide
if not vectorize:
    X.astype('float64')
    Y.astype('float64')

if vectorize:
    pass
elif use_ascii:
    X /= 90.
elif not use_ascii:
    X /= 20.

# Split train and test set
length_divide = round(X.shape[0]*train_test_split)
X_train, X_test = X[:length_divide], X[length_divide:] # X data set
Y_train, Y_test = Y[:length_divide], Y[length_divide:] # Y data set

# Split train to train and validation
length_divide = round(X_train.shape[0]*train_val_split)
X_train, X_val = X_train[:length_divide], X_train[length_divide:] # X data set
Y_train, Y_val = Y_train[:length_divide], Y_train[length_divide:] # Y data set

logger.info('Train set: {} positive samples and {} negative samples'.format(np.count_nonzero(Y_train),
                                                                            Y_train.shape[0] - np.count_nonzero(Y_train)))
logger.info('Validation set: {} positive samples and {} negative samples'.format(np.count_nonzero(Y_val),
                                                                            Y_val.shape[0] - np.count_nonzero(Y_val)))
logger.info('Test set: {} positive samples and {} negative samples'.format(np.count_nonzero(Y_test),
                                                                            Y_test.shape[0] - np.count_nonzero(Y_test)))
########################################################################################################################
#endregion

#region LOADING AND TRAINING MODEL
print('Training model...')

"""Consider splitting classes into separate files"""

#region History and realtime plotting callback
# Recording history for ending training early
class history_recorder(keras.callbacks.Callback):

    def __init__(self):
        self.config_plot_realtime = False
        self.history = {}
        self.record_initials = True
        self.history['loss'] = []
        self.history['val_loss'] = []
        self.history['binary_accuracy'] = []
        self.history['val_binary_accuracy'] = []
        self.loss_batch_history = []
        self.acc_batch_history = []
        self.lr_history = []

    def on_train_begin(self, logs={}):
        if self.config_plot_realtime:
            plt.ion()  # interactive mode for real time plotting. Add plt.ioff() when done
            self.fig, self.ax = plt.subplots(figsize=(10, 5))
            self.ax_acc = plt.subplot(1, 2, 1)  # Acc
            self.ax_acc.set_xlim([0, epochs])
            self.ax_acc.set_ylim([0, 1])
            self.ax_acc.set_autoscale_on(False)  # fix limits
            plt.minorticks_on()
            plt.grid('on')
            plt.grid(b=True, which='major', color='k', linestyle='-', alpha=0.5)
            plt.grid(b=True, which='minor', color='r', linestyle='-', alpha=0.1)
            plt.title('Real time model accuracy')
            plt.ylabel('accuracy')
            plt.xlabel('epoch')
            self.ax_loss = plt.subplot(1, 2, 2, sharex=self.ax_acc, sharey=self.ax_acc)  # Loss
            plt.minorticks_on()
            plt.grid('on')
            plt.grid(b=True, which='major', color='k', linestyle='-', alpha=0.5)
            plt.grid(b=True, which='minor', color='r', linestyle='-', alpha=0.1)
            plt.title('Real time model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

    def on_train_end(self, logs={}):
        if self.config_plot_realtime:
            plt.ioff() # Consider adding it if stopped early
            plt.close('all')

    def on_batch_end(self, batch, logs={}):  # This was the only way that I could record initial loss
        #lr = float(K.get_value(self.model.optimizer.lr))
        #print(lr)
        if self.record_initials:
            self.history['loss'].append(logs.get('loss'))
            self.history['val_loss'].append(logs.get('loss'))
            self.history['binary_accuracy'].append(logs.get('binary_accuracy'))  # Assume that val and train are same
            self.history['val_binary_accuracy'].append(logs.get('binary_accuracy'))
            self.record_initials = False

    def on_epoch_end(self, epoch, logs={}):
        self.history['loss'].append(logs.get('loss'))
        self.history['val_loss'].append(logs.get('val_loss'))
        self.history['binary_accuracy'].append(logs.get('binary_accuracy'))
        self.history['val_binary_accuracy'].append(logs.get('val_binary_accuracy'))
        if self.config_plot_realtime:
            if epoch%plot_realtime_interval == 0:
                self.ax_acc.plot(self.history['binary_accuracy'], 'r-')
                self.ax_acc.plot(self.history['val_binary_accuracy'], 'b-')
                self.ax_acc.legend(['train', 'test'], loc='upper left')
                self.ax_acc.set_ylim([0, 1])
                self.ax_loss.plot(self.history['loss'], 'r-')
                self.ax_loss.plot(self.history['val_loss'], 'b-')
                self.ax_loss.legend(['train', 'test'], loc='upper left')
                self.fig.canvas.draw()
                plt.pause(0.0000001)
#endregion

#region Learningrate callback
# Learning rate scheduler
class learning_rate_scheduler(keras.callbacks.Callback):

    def __init__(self):
        self.lr_scheduler_plan = {}

    def on_epoch_end(self, epoch, logs={}):
        if use_lr_scheduler:
            indices = [(x, y) for x, y in self.lr_scheduler_plan.items() if logs.get('binary_accuracy') >= x]
            if indices != []:
                K.set_value(model.optimizer.lr, indices[0][1]) # lr = y, value
                print(f'\nChanged learning rate to {indices[0][1]}')
                del self.lr_scheduler_plan[indices[0][0]]  # acc = x, key
#endregion

#region Fixing callbacks
# RUNNING IN ONEDRIVE OR DROPBOX SEEMS TO INDUCE PERMISSION DENIED SINCE IT WRITES SO FAST
checkpointer = ModelCheckpoint(filepath=os.getcwd() +'/' + 'storage/checkpoint.h5',  # I need full path, or permission error
                               verbose=0,
                               save_best_only=True)
history_callback = history_recorder()
learning_rate_callback = learning_rate_scheduler()
learning_rate_callback.lr_scheduler_plan = lr_scheduler_plan
history_callback.config_plot_realtime = config_plot_realtime
#endregion

#region Loading model
# Try to load model
if load_saved_model or train_loaded_model:
    try:
        logger.info('Loading model from file...')
        model = load_model('results/Signal_peptide_model.h5')  # Looks like you can load model and keep training
        logger.info('Loaded model: ')
        model.summary(print_fn=lambda txt: logger.info(txt))
        with open('storage/history.pkl', 'rb') as pickle_file:
            history = pickle.load(pickle_file)
        history_callback.history = history  # OBS: from above
    except Exception as e:
        logger.info(e)
        logger.info('Cannot load model')
        load_saved_model = False
#endregion

#region Check callback
# Check if program has crashed beforehand
if os.path.isfile('storage/checkpoint.h5') and os.path.isfile('storage/history.pkl'):
    logger.info('Detected an old checkpoint and history file in storage folder. Maybe your training crashed?')
    logger.info("Don't load checkpoint: 0")
    logger.info('Load checkpoint and evaluate: 1')
    logger.info("Load checkpoint and continue training: 2")
    while True:
        _choice = input('Choice: ')
        logger.info(_choice)
        if _choice == '0':
            train_loaded_model = False  # Debug this
            load_saved_model = False  # Debug this
            os.remove('storage/checkpoint.h5')
            break
        elif _choice == '1':
            train_loaded_model = False
            load_saved_model = True
            model = load_model('storage/checkpoint.h5')
            os.remove('storage/checkpoint.h5')
            with open('storage/history.pkl', 'rb') as pickle_file:
                history = pickle.load(pickle_file)
            break
        elif _choice == '2':
            train_loaded_model = True
            model = load_model('storage/checkpoint.h5')
            os.remove('storage/checkpoint.h5')
            with open('storage/history.pkl', 'rb') as pickle_file:
                history_callback.history = pickle.load(pickle_file)
            break
        else:
            print('Input not valid. Try again. Ctrl+c to quit')
#endregion

#region Training
t = time.time()
if (load_saved_model == False) or (train_loaded_model == True):
    try:
        logger.info('Press ctrl+c to stop training early')
        model.fit(X_train, Y_train,
                  validation_data=(X_val, Y_val),
                  epochs=epochs,
                  batch_size=batch_size,
                  shuffle=True,
                  verbose=1,
                  callbacks=[history_callback,
                             checkpointer,
                             learning_rate_callback])
        history = history_callback.history
    except KeyboardInterrupt:
        logger.info('\nStopped training...')
        model = load_model('storage/checkpoint.h5')
        os.remove('storage/checkpoint.h5')  # comment this for debugging
        history = history_callback.history

    with open('storage/history.pkl', 'wb') as pickle_file:
        pickle.dump(history, pickle_file, pickle.HIGHEST_PROTOCOL)
    if config_plot_realtime:
        plt.ioff()
        plt.close('all')
    logger.info('It took {0:.5f} seconds to train'.format(time.time()-t))
#endregion

#region Evaluation and saving
logger.info('Evaluation of model processing...')
scores = model.evaluate(X_test, Y_test)
logger.info("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

logger.info('Saving model...')
model.save('results/Signal_peptide_model.h5', overwrite=True)
#endregion
########################################################################################################################
#endregion

#region PLOTS
"""ADD PRINTS BEFORE PLOTTING"""

#region Accuracy
logger.info('Plotting accuracy...')
# Plotting training
plt.figure()
plt.minorticks_on()
plt.grid('on')
plt.grid(b=True, which='major', color='k', linestyle='-', alpha=0.5)
plt.grid(b=True, which='minor', color='r', linestyle='-', alpha=0.1)
axes = plt.gca()
axes.set_ylim([0,1])
plt.plot(history['binary_accuracy'], 'r-')
plt.plot(history['val_binary_accuracy'], 'b-')
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('results/train_acc.png')
if plot_performance:
    plt.show(block=False)
else:
    plt.clf()
#endregion

#region Loss
logger.info('Plotting loss...')
plt.figure()
plt.minorticks_on()
plt.grid('on')
plt.grid(b=True, which='major', color='k', linestyle='-', alpha=0.5)
plt.grid(b=True, which='minor', color='r', linestyle='-', alpha=0.1)
axes = plt.gca()
axes.set_ylim([0,1])
plt.plot(history['loss'], 'r-')
plt.plot(history['val_loss'], 'b-')
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('results/train_loss.png')
if plot_performance:
    plt.show(block=False)
else:
    plt.clf()
#endregion

#region Confusion matrix
logger.info('Creating confusion matrix on test samples...')
predictions = model.predict(X_test)
predictions = np.round(predictions)

TP=TN=FP=FN=0
TN, FP, FN, TP = metrics.confusion_matrix(Y_test, np.round(predictions)).ravel()  # Could rewrite own cm function if I wanted
"""
for i in range(Y_test.shape[0]): # All samples
    if predictions[i] == 1 and Y_test[i] == 1: # True positive
        TP += 1
    elif predictions[i] == 0 and Y_test[i] == 0: # True negative
        TN += 1
    elif predictions[i] == 1 and Y_test[i] == 0: # False positive
        FP += 1
    elif predictions[i] == 0 and Y_test[i] == 1: # False negative
        FN += 1
    else:
        print('ERROR')
"""

confusion_matrix = np.array([[0, 0], [0, 0]])
confusion_matrix[0][0] = TP
confusion_matrix[1][1] = TN
confusion_matrix[1][0] = FP
confusion_matrix[0][1] = FN
confusion_matrix_normalized = confusion_matrix/Y_test.shape[0]  # Total number of samples
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


########################################################################################################################
#endregion

#region VISUALIZE MODEL

#region Model overview
# https://graphviz.gitlab.io/download/ needs this
# and installing pydot-ng and graphviz
# NOTE: if you use linux, the path needs to be changed to correct path
from keras.utils import plot_model
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
plot_model(model, to_file='results/model.png')
plt.figure()
img=plt.imread('results/model.png')
imgplot = plt.imshow(img)
plt.tight_layout()
plt.axis('off')
if plot_performance:
    plt.show(block=False)
else:
    plt.clf()
#endregion

#region Filters
for i,j in enumerate(model.layers):
    layer = j.get_weights()
    if layer == []:
        continue
    else:
        layer = layer[0]
    plt.subplot(1,len(model.layers),i+1)
    logger.info(f'Bias of layer {j} ' + str(layer.shape))
    logger.info(f'Output of layer {j} ' + str(j.get_weights()[1].shape))
    if 'Conv1D' in str(j):  # batch, cropped axis, features
        try:
            layer = np.squeeze(layer, axis=1) # 20, 1, 254 --> 20, 254
            layer = np.transpose(layer)  # 20, 254 --> 254, 20
        except:
            layer = layer[0]  # if ex 25, 254, 128 --> take first slice
        plt.title(f'Conv1D_{i}')
    elif 'Dense' in str(j):
        plt.title(f'Dense_{i}')
    else:
        plt.title(f'Other_{i}')
    heatmap = plt.imshow(layer, cmap='bwr') # need to do list in list if 1d dim
    plt.colorbar(heatmap, aspect=50, orientation='vertical', fraction=0.046, pad=0.04)

plt.savefig('results/train_filters.png')
if plot_performance:
    plt.show(block=False)
else:
    plt.clf()
#endregion

########################################################################################################################
#endregion

if plot_performance:
    print('Close plots to continue')
    plt.show()

if train_GAN:
    #region GAN
    #region Creating generator
    logger.info('Creating generator...')
    # Remember: output = kernel - input + 1
    generator = Sequential()
    generator.add(Convolution1D(128, 20, input_shape=(30, 21)))
    generator.add(LSTM(64))
    generator.add(Dense(630, activation='sigmoid'))
    generator.add(Reshape((30,21)))
    logger.info(generator.summary(print_fn=lambda txt: logger.info(txt)))
    logger.info('Input shape: ' + str(generator.input_shape))
    logger.info('Output shape: ' + str(generator.output_shape))
    #endregion

    logger.info('Creating discriminator...')
    discriminator = keras.models.clone_model(model)
    logger.info(discriminator.summary(print_fn=lambda txt: logger.info(txt)))
    logger.info('Input shape: ' + str(discriminator.input_shape))
    logger.info('Output shape: ' + str(discriminator.output_shape))

    #region Compiling models
    adam = keras.optimizers.Adam(lr=GAN_lr)
    logger.info('Compiling models...')  # REMEMBER, CHANGE SO THAT THEY HAVE DIFFERENT LR
    generator.compile(metrics=['accuracy'],
                        loss='binary_crossentropy',  # USED WRONG LOSS
                        optimizer=adam)
    discriminator.compile(metrics=['binary_accuracy'],
                          loss='binary_crossentropy',
                          optimizer=adam)

    if load_trained_GAN:
        try:
            discriminator = load_model('results/Signal_peptide_discriminator.h5')
            generator = load_model('results/Signal_peptide_generator.h5')
        except:
            logger.info('GAN models not found...')

        logger.info("Setting up GAN (generator + discriminator)")
    GAN = Sequential()
    GAN.add(generator)
    discriminator.trainable=False
    GAN.add(discriminator)
    GAN.compile(loss='binary_crossentropy',
                optimizer=adam,
                metrics=['binary_accuracy'])
    logger.info(GAN.summary(print_fn=lambda txt: logger.info(txt)))
    logger.info('Input shape: ' + str(GAN.input_shape))
    logger.info('Output shape: ' + str(GAN.output_shape))
    #endregion

    #region Fixing data
    logger.info('Fixing real data...')
    negative_indices = list(np.where(Y == 0)[0])  #OBS
    X_positive = np.delete(X, list(negative_indices), axis=0)
    Y_positive = np.delete(Y, list(negative_indices), axis=0)
    Y_positive = Y_positive[:GAN_limit_samples]
    X_positive = X_positive[:GAN_limit_samples]
    false_labels = np.zeros((Y_positive.shape[0], 1))  # Y data (0 = false, 1 = true), for discriminator
    true_labels = np.ones((Y_positive.shape[0], 1))  # Y data (1 = true), for generator
    #endregion

    #region Training GAN
    _train_disc = True
    _train_gen = True
    from scripts.load_data import ascii_to_AAseq, non_ascii_to_AAseq, vectorize_to_AAseq

    for i in range(GAN_epochs):
        print(f'{i} out of {GAN_epochs} completed')
        logger.info('-'*10 + f'Iteration {i}' + '-'*10)
        # Shuffle data, this is actually bad for GANS, but I don't care atm
        # random_data = np.ones((Y_positive.shape[0], 30, 1))  # Y data (1 = true), for generator
        random_data = np.random.random((Y_positive.shape[0], 30, 21))  # Generate new data for generator
        predictions = generator.predict(random_data)  # Start data
        merged_dataX = np.concatenate((X_positive, predictions), axis=0)
        merged_dataY = np.concatenate((Y_positive, false_labels), axis=0)
        s = np.arange(merged_dataX.shape[0])
        np.random.shuffle(s)
        merged_dataX = merged_dataX[s]
        merged_dataY = merged_dataY[s]

        if _train_disc:
            # You should acually have only fake or only real in each batch
            print('Training discriminator to distinguish real and generated data...')
            discriminator.trainable = True
            history_disc = discriminator.fit(merged_dataX,
                                             merged_dataY,
                                             batch_size=GAN_batch_size,
                                             epochs=5)
            logger.info('Discriminator loss: ' + str(history_disc.history['loss'][-1]))

        if _train_gen:
            print('Training generator to fool discriminator...')
            discriminator.trainable = False
            history_gen = GAN.fit(random_data,
                                  true_labels,
                                  batch_size=GAN_batch_size,
                                  epochs=5) # I need to train the generator more, since it is weaker built than discriminator
            logger.info('Generator loss: ' + str(history_gen.history['loss'][-1]))
            predictions = generator.predict(random_data)  # Generator tries to create data
            
        if history_gen.history['binary_accuracy'][-1]<0.9:
            _train_disc = False
            _train_gen = True
        elif history_disc.history['binary_accuracy'][-1]<0.9:
            _train_disc = True
            _train_gen = False
        else:
            _train_disc = True
            _train_gen = True

        """
        history_gen = generator.fit(random_data,
                                    X_positive,
                                    batch_size=GAN_batch_size,
                                    epochs=5)  # I need to train the generator more, since it is weaker built than discriminator
        logger.info('Generator loss: ' + str(history_gen.history['loss'][-1]))
        predictions = generator.predict(random_data)  # Generator tries to create data
        """

        if vectorize:
            predictions_seq = vectorize_to_AAseq(predictions)
        elif use_ascii:
            predictions_seq = ascii_to_AAseq(predictions)
        else:
            predictions_seq = non_ascii_to_AAseq(predictions)

        #logger.info('First amino acids: ' + str(np.round(predictions[0][0], 4)))
        for i in range(0,10):
            logger.info('Generated sequence: ' + ''.join(next(predictions_seq))) # predictions_seq = generator object
        #scores = model.evaluate(predictions, Y_positive)
        #logger.info(f"Feeding output to model gives: {scores[1] * 100}%, should be close to 100%")
        discriminator.save('results/Signal_peptide_discriminator.h5', overwrite=True)
        generator.save('results/Signal_peptide_generator.h5', overwrite=True)

    #endregion

        logger.info('Saving generator...')
    discriminator.save('results/Signal_peptide_discriminator.h5', overwrite=True)
    generator.save('results/Signal_peptide_generator.h5', overwrite=True)

    plt.show()

    #endregion