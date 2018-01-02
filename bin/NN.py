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
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import LSTM
from keras.layers import Convolution1D
from keras.layers import MaxPooling1D
from keras.layers import UpSampling1D
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

"""Discovered: output shape = cutoff-kernel+1"""
print('Creating model...')
model = Sequential()

"""
# Model 1
'''lr 0.02 is good for this one.
Lower lr to 0.01 when getting 0.88 val acc
Very fast learning'''
model.add(Convolution1D(254, 20, input_shape=(cutoff, 1), activation='relu'))
model.add(GaussianNoise(0.01))  # Maybe try with noise to reduce overfit?
#model.add(Dropout(0.2))
model.add(MaxPooling1D())
#model.add(UpSampling1D(5))  # output = 5
model.add(Convolution1D(128, 5, activation='relu'))
model.add(Flatten())
model.add(Dense(30, activation='relu'))
model.add(Dense(10, activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))
"""

"""
# Model 2
'''Give the LSTM time, at least 100 epochs before it starts to be good'''
model.add(Convolution1D(254, 20, input_shape=(cutoff, 1), activation='relu'))
#model.add(GaussianNoise(0.03))  # Maybe try with noise to reduce overfit?
#model.add(Dropout(0.2))
model.add(LSTM(512, return_sequences=True)) # [samples, time_steps, feautures], input_shape=(cutoff, 1)
model.add(LSTM(512))
model.add(Dense(1, activation='sigmoid'))
"""

# Model 3
'''Needs vectorized data, pure LSTM.
Give it about 100 epochs to improve
It seems that dropout is actually not needed here!'''
model.add(LSTM(512, input_shape=(cutoff, 21), return_sequences=True)) # [samples, time_steps, feautures], input_shape=(cutoff, 1)
model.add(LSTM(512))
model.add(Dense(1, activation='sigmoid'))

"""
# Model 4
'''Needs vectorized data combined LSTM and CNN
Not tested yet'''
model.add(Convolution1D(254, 20, input_shape=(cutoff, 1), activation='relu'))
model.add(LSTM(512, input_shape=(cutoff, 21), return_sequences=True)) # [samples, time_steps, feautures], input_shape=(cutoff, 1)
model.add(LSTM(512))
model.add(Dense(1, activation='sigmoid'))
"""

print('Compiling model...')
#Compile model
sgd = keras.optimizers.SGD(lr=lr, decay=decay, momentum=momentum, nesterov=True)
model.compile(metrics=['binary_accuracy'],
              loss='binary_crossentropy',
              optimizer=sgd)

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
    X, Y = load_training(cutoff,
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

length_divide = round(X.shape[0]*val_split)
X_train, X_test = X[:length_divide], X[length_divide:] # X data set
Y_train, Y_test = Y[:length_divide], Y[length_divide:] # Y data set

if not vectorize:
    X_train.astype('float64')
    X_test.astype('float64')
    Y_train.astype('float64')
    Y_test.astype('float64')

logger.info('Train set: {} positive samples and {} negative samples'.format(np.count_nonzero(Y_train),
                                                                    Y_train.shape[0] - np.count_nonzero(Y_train)))
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

    def on_train_begin(self, logs={}):
        self.history = {}
        self.history['loss'] = []
        self.history['val_loss'] = []
        self.history['binary_accuracy'] = []
        self.history['val_binary_accuracy'] = []
        if self.config_plot_realtime:
            plt.ion()  # interactive mode for real time plotting. Add plt.ioff() when done
            self.fig, self.ax = plt.subplots(figsize=(10, 5))
            self.ax_acc = plt.subplot(1, 2, 1)  # Acc
            self.ax_acc.set_xlim([0, epochs])
            self.ax_acc.set_ylim([0, 1])
            self.ax_acc.set_autoscale_on(False)  # fix limits
            plt.grid('on')
            plt.title('Real time model accuracy')
            plt.ylabel('accuracy')
            plt.xlabel('epoch')
            self.ax_loss = plt.subplot(1, 2, 2, sharex=self.ax_acc, sharey=self.ax_acc)  # Loss
            plt.grid('on')
            plt.title('Real time model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

    def on_train_end(self, logs={}):
        if self.config_plot_realtime:
            plt.ioff() # Consider adding it if stopped early
            plt.close('all')

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
            indices = [(x, y) for x, y in self.lr_scheduler_plan.items() if logs.get('val_binary_accuracy') >= x]
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
        with open('storage/history.pkl', 'rb') as pickle_file:
            history = pickle.load(pickle_file)
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
            train_loaded_model = False
            load_saved_model = False
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
            break
        else:
            print('Input not valid. Try again. Ctrl+c to quit')
#endregion

#region Training
t = time.time()
if (load_saved_model == False) or (train_loaded_model == True):
    try:
        logger.info('Press ctrl+c to stop training early')
        history = model.fit(X_train, Y_train,
                        validation_data=(X_test, Y_test),
                        epochs=epochs,
                        batch_size=batch_size,
                        shuffle=True,
                        verbose=1,
                        callbacks=[history_callback,
                                   checkpointer,
                                   learning_rate_callback])
        history = history.history
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
print('Plotting accuracy...')
# Plotting training
plt.figure()
plt.grid(which='minor')
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
print('Plotting loss...')
plt.figure()
plt.grid()
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
print('Creating confusion matrix on test samples...')
predictions = model.predict(X_test)
predictions = np.round(predictions)
confusion_matrix = np.array([[0,0],[0,0]])
for i in range(Y_test.shape[0]): # All samples
    if predictions[i] == 1 and Y_test[i] == 1: # True positive
        confusion_matrix[0][0] += 1
    elif predictions[i] == 0 and Y_test[i] == 0: # True negative
        confusion_matrix[1][1] += 1
    elif predictions[i] == 1 and Y_test[i] == 0: # False positive
        confusion_matrix[1][0] += 1
    elif predictions[i] == 0 and Y_test[i] == 1: # False negative
        confusion_matrix[0][1] += 1
    else:
        print('ERROR')
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
print('Creating ROC curve')
plt.figure()
plt.axis('on')
plt.grid()
predictions = model.predict(X_test)
fpr, tpr, thresholds = metrics.roc_curve(Y_test, predictions)
auc_score = metrics.roc_auc_score(Y_test, predictions)
if 0.9 < auc_score <= 1.0:
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
print('Calculating metrics')
predictions = model.predict(X_test)
TN, FP, FN, TP = metrics.confusion_matrix(Y_test, np.round(predictions)).ravel()  # Could rewrite own cm function if I wanted
accuracy_score = (TN+TP)/(TN+FP+FN+TP) # or use metrics.accuracy_score(Y_test, predictions)
precision_score = TP/(FP+TP) # or use metrics.precision_score(Y_test, predictions)
sensitivity_score = TP/(TP+FN) # or use metrics.recall_score(Y_test, predictions)
specificity_score = TN/(TN+FP)
logger.info('Accuracy: ' + str(accuracy_score)
            + ', Precision: ' + str(precision_score)
            + ', Sensitivity: ' + str(sensitivity_score)
            + ', Specificity: ' + str(specificity_score)) # Log this shit
#endregions

#region Parameters
"""
print('Showing used parameters')
plt.figure()
plt.axis('off')
table = plt.table(cellText=[[x] for x in vars(args).values()], # Log this shit too
                 rowLabels=[x for x in vars(args).keys()],
                 rowLoc='right',
                 colWidths=[0.2]*len(vars(args).keys()),
                 loc='center')
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(2., 2.)
plt.savefig('results/train_param.png')
if plot_performance:
    plt.show(block=False)
else:
    plt.clf()
"""
#endregion

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
            layer = layer[0][:][:]  # if ex 25, 254, 128 --> take first slice
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

#region Connections
'''
import networkx as nx
G=nx.Graph()
labels={}
colors = []
pos={}
node_size = []
layers = []

def color_nodes(node_name, layer):
    if 'Conv' in str(layer):
        colors.append('r')
    elif 'Dense' in str(layer):
        colors.append('b')

"""Layers have two params: first [0] is the weight, second [1] is the bias"""
for layer_index, layer_contents in enumerate(model.layers): # Get all layers with weights, ignore rest
    if layer_contents.get_weights() == []:
        continue
    else:
        layers.append(layer_contents)

# http://www.shogun-toolbox.org/notebook/latest/neuralnets_digits.html inspiration
for layer_index, layer_contents in enumerate(layers): # Nodes
    nodes_in_layer = len(layer_contents.get_weights()[1])
    for node in range(nodes_in_layer):
        node_name = str(layer_index) + '_' + str(node)  # Node will have the name 00 first etc
        color_nodes(node_name, layer_contents)
        G.add_node(node_name)
        pos[node_name] = (node, len(layers) - layer_index) # I want first layr on top
        labels[node_name] = round(layer_contents.get_weights()[1][node], 3)  # Bias, only 3 decimals
        node_size.append(80)  # Can be used in the future to add different sizes of nodes based on parameters


for i in range(1,len(layers)-1): # Edges
    print(layers[i].get_weights()[0].shape[1])
    print(layers[i+1].get_weights()[0].shape[1])
    for j in range(layers[i].get_weights()[0].shape[1]):
        node_name1 = str(i)  + '_' + str(j)
        for k in range(layers[i+1].get_weights()[0].shape[1]):
            node_name2 = str(i+1) + '_' + str(k)
            print(node_name1, node_name2)
            G.add_edge(node_name1, node_name2)
plt.figure()
#pos = nx.get_node_attributes(G,'pos')  # if pos added in each node
nx.draw(G,
        labels=labels,
        font_size=10,
        pos=pos,
        node_color=colors,
        node_size=node_size,
        with_labels = True)
#nx.draw_random(G)
#nx.draw_circular(G)
#nx.draw_spectral(G)
#nx.draw_shell(G)
plt.savefig('results/train_network.png')
if plot_performance:
    plt.show(block=False)
else:
    plt.clf()
'''

########################################################################################################################
#endregion

########################################################################################################################
#endregion

if plot_performance:
    plt.show()
    print('Close plots to continue')

if train_GAN:
    #region GAN
    #region Creating generator
    print('Creating generator...')
    generator = Sequential()
    # Remember: output = kernel - input + 1
    generator.add(Convolution1D(2048, 20, input_shape=(30, 1), activation='relu'))
    generator.add(Convolution1D(2048, 5, activation='relu'))
    generator.add(Flatten())
    generator.add(Dense(630, activation='sigmoid')) # Softmax since classification
    generator.add(Reshape((30,21), input_shape=(630,)))
    print(generator.summary())
    print('Input shape: ' + str(generator.input_shape))
    print('Output shape: ' + str(generator.output_shape))
    #endregion

    print('Creating discriminator...')
    discriminator = keras.models.clone_model(model)

    #region Compiling models
    print('Compiling models...')  # REMEMBER, CHANGE SO THAT THEY HAVE DIFFERENT LR
    sgd = keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True)  # Too high lr --> output stuck on zero, no improvement
    generator.compile(metrics=['binary_accuracy'],
                        loss='binary_crossentropy',
                        optimizer=sgd)
    sgd = keras.optimizers.SGD(lr=0.02, momentum=0.5, nesterov=True)  # Too high lr --> output stuck on zero, no improvement
    discriminator.compile(metrics=['binary_accuracy'],
                          loss='binary_crossentropy',
                          optimizer=sgd)

    if load_trained_GAN:
        try:
            discriminator = load_model('results/Signal_peptide_discriminator.h5')
            generator = load_model('results/Signal_peptide_generator.h5')
        except:
            logger.info('GAN models not found...')

    print("Setting up GAN (generator + discriminator)")
    GAN = Sequential()
    GAN.add(generator)
    discriminator.trainable=False
    GAN.add(discriminator)
    GAN.compile(loss='binary_crossentropy',
                optimizer=sgd,
                metrics=['binary_accuracy'])
    print(GAN.summary())
    print('Input shape: ' + str(GAN.input_shape))
    print('Output shape: ' + str(GAN.output_shape))
    #endregion

    #region Fixing data
    print('Fixing real data...')
    negative_indices = np.where(Y == Y.argmin())
    X_positive = np.delete(X, list(negative_indices), axis=0)
    Y_positive = np.delete(Y, list(negative_indices), axis=0)
    false_labels = np.zeros((Y_positive.shape[0], 1))  # Y data (0 = false, 1 = true), for discriminator
    true_labels = np.ones((Y_positive.shape[0], 1))  # Y data (1 = true), for generator
    random_data = np.random.random((Y_positive.shape[0], 30, 1))  # Generate new data for generator
    #endregion

    #region Training GAN
    _train_disc = True
    from scripts.load_data import ascii_to_AAseq, non_ascii_to_AAseq, vectorize_to_AAseq

    for i in range(GAN_epochs):
        print(f'{i} out of {GAN_epochs} completed')

        # Shuffle data, this is actually bad for GANS, but I don't care atm
        # random_data = np.ones((Y_positive.shape[0], 30, 1))  # Y data (1 = true), for generator
        predictions = generator.predict(random_data)  # Start data
        merged_dataX = np.concatenate((X_positive, predictions), axis=0)
        merged_dataY = np.concatenate((Y_positive, false_labels), axis=0)
        s = np.arange(merged_dataX.shape[0])
        np.random.shuffle(s)
        merged_dataX = merged_dataX[s]
        merged_dataY = merged_dataY[s]

        if _train_disc:
            print('Training discriminator to distinguish real and generated data...')
            discriminator.trainable = True
            discriminator.fit(merged_dataX,
                              merged_dataY,
                              batch_size=GAN_batch_size,
                              epochs=5)
            # Should train with generated and true data seperatly, but whatever

        # Use this if you do NOT have a model that uses categorical data (vectorization)
        print('Training generator to fool discriminator...')  # VERY BAD WITH CATEGORICAL VALUES
        discriminator.trainable = False
        history = GAN.fit(random_data,
                          true_labels,
                          batch_size=GAN_batch_size,
                          epochs=10) # I need to train the generator more, since it is weaker built than discriminator

        predictions = generator.predict(random_data)  # Generator tries to create data

        if history.history['binary_accuracy'][-1]<0.9:
            _train_disc = False
        else:
            _train_disc = True

        if vectorize:
            predictions_seq = vectorize_to_AAseq(predictions)
        elif use_ascii:
            predictions_seq = ascii_to_AAseq(predictions)
        else:
            predictions_seq = non_ascii_to_AAseq(predictions)
        logger.info('-'*10 + f'Iteration {i}' + '-'*10)
        logger.info('History loss: ' + str(history.history['loss'][-1]))
        #logger.info('First amino acids: ' + str(np.round(predictions[0][0], 4)))
        for i in range(0,10):
            logger.info('Generated sequence: ' + ''.join(next(predictions_seq))) # predictions_seq = generator object
        scores = model.evaluate(predictions, Y_positive)
        logger.info(f"Feeding output to model gives: {scores[1] * 100}%, should be close to 100%")
        discriminator.save('results/Signal_peptide_discriminator.h5', overwrite=True)
        generator.save('results/Signal_peptide_generator.h5', overwrite=True)

    #endregion

    print('Saving generator...')
    discriminator.save('results/Signal_peptide_discriminator.h5', overwrite=True)
    generator.save('results/Signal_peptide_generator.h5', overwrite=True)

    plt.show()

    #endregion