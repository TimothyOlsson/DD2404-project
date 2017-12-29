#region PARAMETERS
########################################################################################################################
# CONSIDER USING JSON + ARGPARSER

# GPU
_use_gpu = True

# DATA
_load_array = True
_data_folder = '../../data/training_data'
_use_ascii = False
_equalize_data = True
_save_array = True
cutoff = 30
val_split = 0.7
data_augmentation = 'ALL'
fix_samples = 'NOISE'

# MODEL
_load_model = False
_train_load_model = False

# HYPERPARAMETERS
lr = 0.02  # More parameters --> change lr
decay = 1.1e-5
momentum = 0.5
epochs = 3000
batch_size = 524
_use_lr_scheduler = True
_lr_dict = {}  # REMEMBER TO CHANGE THIS WHEN CHANGING MODEL AND LR

# PLOTTINGS
_plot_performance = True
_config_plot_realtime = True
_plot_realtime_interval = 10

# GAN
_generate_GAN = False
_GAN_lr = 0.03
GAN_epochs = 1000

print(locals())


#region use gpu
"""I need to set CUDA before imports!!!"""
import os
if not _use_gpu:
    """Use gpu if you have many parameters in your model"""
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    print('Using cpu...')
else:
    print('Using gpu...')
#endregion

########################################################################################################################
#endregion

#region  MODULES
########################################################################################################################

"""CONSIDER TO IMPORT WHEN NEEDED AND HAVE OPTIONAL LIBRARIES TO PREVENT BREAKAGE"""

print('Importing modules...')
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import LSTM
from keras.layers import SimpleRNN
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
import pandas as pd
import tensorflow as tf
import time
import matplotlib
# Needed to change plot position while calculating. NEEDS TO ADDED BEFORE pyplot import
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
from keras.models import load_model
import pickle
from sklearn import metrics  # For ROC curve
from keras import backend as K  # for layer viz
import pdb
import multiprocessing

#endregion

#region CREATING MODEL
########################################################################################################################

"""Discovered: output shape = cutoff-kernel+1"""
print('Creating model...')
model = Sequential()
"""
# Model 1
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

# Model 2
look_back = 1  # number of timesteps to look back
model.add(Convolution1D(254, 20, input_shape=(cutoff, 1), activation='relu'))
#model.add(GaussianNoise(0.03))  # Maybe try with noise to reduce overfit?
model.add(Dropout(0.2))
model.add(LSTM(20, input_shape=(cutoff, look_back)))
model.add(Reshape((20,1)))
model.add(LSTM(20))
model.add(Dense(1, activation='sigmoid'))

print('Compiling model...')
#Compile model
sgd = keras.optimizers.SGD(lr=lr, decay=decay, momentum=momentum, nesterov=True)
model.compile(metrics=['binary_accuracy'],
              loss='binary_crossentropy',
              optimizer=sgd)

print('Showing model...')
print(model.summary())
print('Input shape: ' + str(model.input_shape))
print('Output shape: ' + str(model.output_shape))

"""
Train acc and loss decreases, Test acc and loss increases = overfitting
Solutions: more data, change model, add dropout, add noise, 

Test acc and loss fluctuating = some samples are classified randomly
Solutions: more data, change hyperparameters, add dropout
"""

#endregion

#region DATA
########################################################################################################################

# Prints already in load script
if _load_array:
    try:
        from load_data import load_from_file
        X,Y = load_from_file('storage/loaded_array.npz')
    except:
        print('Cannot load numpy file')
        _load_array = False  # If it cant find the file, load the data

if not _load_array:
    from load_data import load_training
    X, Y = load_training(cutoff,
                         _data_folder,
                         data_augmentation=data_augmentation,
                         fix_samples=fix_samples,
                         _equalize_data=_equalize_data,
                         save_array=_save_array)

print('Shuffling data...')
np.random.seed(1) # fix random seed for reproducing results
s = np.arange(X.shape[0])
np.random.shuffle(s)
X = X[s][:]
Y = Y[s]

# Reshape, need 3d for CNN
X = X.reshape(X.shape[0], X.shape[1], 1)

#time.sleep(2)  # If you want to see what data has loaded

# Fix so that you can divide
X.astype('float64')
Y.astype('float64')

print('Preprocessing data...')
# Preprocessing data

if _use_ascii:
    X /= 90.
elif not _use_ascii:
    X /= 20.

length_divide = round(X.shape[0]*val_split)
X_train, X_test = X[:length_divide], X[length_divide:] # X data set
Y_train, Y_test = Y[:length_divide], Y[length_divide:] # Y data set

X_train.astype('float64')
X_test.astype('float64')
Y_train.astype('float64')
Y_test.astype('float64')

print('Train set: {} positive samples and {} negative samples'.format(np.count_nonzero(Y_train),
                                                                    Y_train.shape[0] - np.count_nonzero(Y_train)))
print('Test set: {} positive samples and {} negative samples'.format(np.count_nonzero(Y_test),
                                                                    Y_test.shape[0] - np.count_nonzero(Y_test)))

#endregion

#region LOADING AND TRAINING MODEL
########################################################################################################################

print('Training model...')

"""Consider splitting classes into separate files"""

#region History and realtime plotting callback
# Recording history for ending training early
class history_recorder(keras.callbacks.Callback):

    def __init__(self):
        self._config_plot_realtime = False

    def on_train_begin(self, logs={}):
        self.history = {}
        self.history['loss'] = []
        self.history['val_loss'] = []
        self.history['binary_accuracy'] = []
        self.history['val_binary_accuracy'] = []
        if self._config_plot_realtime:
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
        if self._config_plot_realtime:
            plt.ioff() # Consider adding it if stopped early
            plt.close('all')

    def on_epoch_end(self, epoch, logs={}):
        self.history['loss'].append(logs.get('loss'))
        self.history['val_loss'].append(logs.get('val_loss'))
        self.history['binary_accuracy'].append(logs.get('binary_accuracy'))
        self.history['val_binary_accuracy'].append(logs.get('val_binary_accuracy'))
        if self._config_plot_realtime:
            if epoch%_plot_realtime_interval == 0:
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
        self._lr_dict = {}

    def on_epoch_end(self, epoch, logs={}):
        if _use_lr_scheduler:
            indices = [(x, y) for x, y in self._lr_dict.items() if logs.get('val_binary_accuracy') >= x]
            if indices != []:
                K.set_value(model.optimizer.lr, indices[0][1]) # lr = y, value
                print(f'\nChanged learning rate to {indices[0][1]}')
                del self._lr_dict[indices[0][0]]  # acc = x, key
#endregion

#region Fixing callbacks
# RUNNING IN ONEDRIVE OR DROPBOX SEEMS TO INDUCE PERMISSION DENIED SINCE IT WRITES SO FAST
checkpointer = ModelCheckpoint(filepath=os.getcwd() +'/' + 'storage/checkpoint.h5',  # I need full path, or permission error
                               verbose=0,
                               save_best_only=True)
history_callback = history_recorder()
learning_rate_callback = learning_rate_scheduler()
learning_rate_callback._lr_dict = _lr_dict
history_callback._config_plot_realtime = _config_plot_realtime
#endregion

#region Loading model
# Try to load model
if _load_model or _train_load_model:
    try:
        print('Loading model from file...')
        model = load_model('results/Signal_peptide_model.h5')  # Looks like you can load model and keep training
        with open('storage/history.pkl', 'rb') as pickle_file:
            history = pickle.load(pickle_file)
    except Exception as e:
        print(e)
        print('Cannot load model')
        _load_model = False
#endregion

#region Check callback
# Check if program has crashed beforehand
if os.path.isfile('storage/checkpoint.h5') and os.path.isfile('storage/history.pkl'):
    print('Detected an old checkpoint and history file in storage folder. Maybe your training crashed?')
    print("Don't load checkpoint: 0")
    print('Load checkpoint and evaluate: 1')
    print("Load checkpoint and continue training: 2")
    while True:
        _choice = input('Choice: ')
        if _choice == '0':
            _train_load_model = False
            _load_model = False
            os.remove('storage/checkpoint.h5')
            break
        elif _choice == '1':
            _train_load_model = False
            _load_model = True
            model = load_model('storage/checkpoint.h5')
            os.remove('storage/checkpoint.h5')
            with open('storage/history.pkl', 'rb') as pickle_file:
                history = pickle.load(pickle_file)
            break
        elif _choice == '2':
            _train_load_model = True
            model = load_model('storage/checkpoint.h5')
            os.remove('storage/checkpoint.h5')
            break
        else:
            print('Input not valid. Try again. Ctrl+c to quit')
#endregion

#region Training
t = time.time()
if (_load_model == False) or (_train_load_model == True):
    try:
        print('Press ctrl+c to stop training early')
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
        print('\nStopped training...')
        model = load_model('storage/checkpoint.h5')
        os.remove('storage/checkpoint.h5')  # comment this for debugging
        history = history_callback.history

    with open('storage/history.pkl', 'wb') as pickle_file:
        pickle.dump(history, pickle_file, pickle.HIGHEST_PROTOCOL)
    if _config_plot_realtime:
        plt.ioff()
        plt.close('all')
    print('It took {0:.5f} seconds to train'.format(time.time()-t))
#endregion

#region Evaluation and saving
print('Evaluation of model processing...')
scores = model.evaluate(X_test, Y_test)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

print('Saving model...')
model.save('results/Signal_peptide_model.h5', overwrite=True)
#endregion
########################################################################################################################
#endregion

#region PLOTS
########################################################################################################################
"""ADD PRINTS BEFORE PLOTTING"""

#region Accuracy
print('Plotting accuracy...')
# Plotting training
plt.figure()
plt.grid()
axes = plt.gca()
axes.set_ylim([0,1])
plt.plot(history['binary_accuracy'], 'r-')
plt.plot(history['val_binary_accuracy'], 'b-')
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('results/train_acc.png')
if _plot_performance:
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
if _plot_performance:
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
if _plot_performance:
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
if _plot_performance:
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
if _plot_performance:
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
print(accuracy_score, precision_score, sensitivity_score, specificity_score) # Log this shit
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
if _plot_performance:
    plt.show(block=False)
else:
    plt.clf()
"""
#endregion
########################################################################################################################
#endregion

#region VISUALIZE MODEL
########################################################################################################################

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
if _plot_performance:
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
    print(layer.shape)
    print(j.get_weights()[1].shape)
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
if _plot_performance:
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
if _plot_performance:
    plt.show(block=False)
else:
    plt.clf()
'''

plt.show()

########################################################################################################################
#endregion

########################################################################################################################
#endregion

if _generate_GAN:
    #region GAN
    ########################################################################################################################

    #region Creating generator
    print('Creating generator...')
    generator = Sequential()
    # Remember: output = kernel - input + 1
    generator.add(Convolution1D(128, 20, input_shape=(cutoff, 1), activation='sigmoid'))
    generator.add(MaxPooling1D())
    generator.add(Convolution1D(128, 5, activation='sigmoid'))
    generator.add(Convolution1D(254, 1, activation='sigmoid'))
    generator.add(Flatten())
    generator.add(BatchNormalization())  # This made it pretty good!
    generator.add(Dense(10, activation='sigmoid'))
    generator.add(Dense(300, activation='sigmoid'))
    generator.add(Dense(30, activation='sigmoid'))
    generator.add(Dense(30, activation='sigmoid'))
    generator.add(Reshape((30,1), input_shape=(30,)))
    # Input discriminator = (None, 30, 1), need to add dimension
    # Output layer, tanh is recommended but I need positive values
    # DONT FORGET TO CHANGE RANDOM DATA IF YOU CHANGE INPUT SHAPE
    print(generator.summary())
    print('Input shape: ' + str(generator.input_shape))
    print('Output shape: ' + str(generator.output_shape))
    #endregion

    #region Creating discriminator
    print('Creating discriminator...')  # USE SAME AS PREDICTOR, CHANGE WHEN YOU CHANGE MODEL
    discriminator = Sequential()
    discriminator.add(Convolution1D(254, 20, input_shape=(cutoff, 1), activation='relu'))
    discriminator.add(GaussianNoise(0.05))  # Maybe try with noise to reduce overfit?
    discriminator.add(MaxPooling1D())
    discriminator.add(Dropout(0.4))
    discriminator.add(Flatten())
    discriminator.add(Dense(10, activation='relu'))
    discriminator.add(Dense(5, activation='sigmoid'))
    discriminator.add(Dense(1, activation='sigmoid'))
    print(discriminator.summary())
    print('Input shape: ' + str(discriminator.input_shape))
    print('Output shape: ' + str(discriminator.output_shape))
    # endregion

    #region Compiling models
    print('Compiling models...')  # REMEMBER, CHANGE SO THAT THEY HAVE DIFFERENT LR
    sgd = keras.optimizers.SGD(lr=0.01, momentum=momentum, nesterov=True)  # Optimal = 0.03
    discriminator.compile(metrics=['binary_accuracy'],
                        loss='binary_crossentropy',
                        optimizer=sgd)
    sgd = keras.optimizers.SGD(lr=_GAN_lr, momentum=0.5, nesterov=True)  # Too high lr --> output stuck on zero, no improvement
    generator.compile(metrics=['binary_accuracy'],
                        loss='binary_crossentropy',
                        optimizer=sgd)
    print("Setting up GAN (generator + discriminator)")
    GAN = Sequential()
    GAN.add(generator)
    #discriminator = load_model('results/Signal_peptide_model.h5')  # Test
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
    random_data = np.random.random((Y_positive.shape[0], 30, 1))  # Generate data for generator
    #random_data = np.ones((Y_positive.shape[0], 30, 1))  # Y data (1 = true), for generator
    predictions = generator.predict(random_data)  # Generator tries to create data
    from query import predict_to_AAseq, predict_to_numbers
    #endregion

    #region Training GAN
    for i in range(GAN_epochs):
        print(f'{i} out of {GAN_epochs} completed')
        print('Generating data...')


        print('Training generator to trick discriminator...')
        discriminator.trainable = False
        history = GAN.fit(random_data,
                          true_labels,
                          batch_size=1000,
                          epochs=15) # I need to train the generator more, since it is weaker built than discriminator

        if history.history['binary_accuracy'][-1]<0.9:
            _train_disc = False
        else:
            _train_disc = True

        if _train_disc:
            print('Training discriminator with real data...')
            discriminator.trainable = True
            discriminator.fit(X_positive,
                              Y_positive,
                              batch_size=1000,
                              epochs=1)
            print('Training discriminator with generated data...')
            discriminator.fit(predictions,
                              false_labels,
                              batch_size=1000,
                              epochs=1)

        predictions_seq = predict_to_AAseq(predictions, _use_ascii)
        print('Generated sequence: ' + ''.join(next(predictions_seq))) # predictions_seq = generator object
        predictions_numbers = predict_to_numbers(predictions, _use_ascii)
        print('Corresponding to: ' + ','.join(next(predictions_numbers)))
    #endregion

    print('Saving generator...')
    model.save('storage/Signal_peptide_generator.h5', overwrite=True)

    plt.show()

    #endregion