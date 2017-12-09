#region ARGPARSE CLI
import argparse
parser = argparse.ArgumentParser(description='CNN')
parser.add_argument('-i', '--input', dest='data_folder', required=False, default='../../data/training_data',
                    help='')
parser.add_argument('--load_array', dest='_load_array', required=False, default=True, action='store_true',
                    help='')
parser.add_argument('--dont_load_array', dest='_load_array', required=False, default=True, action='store_false',
                    help='')
parser.add_argument('--load_model', dest='_load_model', required=False, default=False, action='store_true',
                    help='')
parser.add_argument('--train_load_model', dest='_train_load_model', required=False, default=False, action='store_true',
                    help='')
parser.add_argument('--use_gpu', dest='use_gpu', required=False, default=False, action='store_true',
                    help='')
parser.add_argument('--use_ascii', dest='use_ascii', required=False, default=False, action='store_true',
                    help='')
args = parser.parse_args()

"""CONSIDER USING A CONFIG FILE INSTEAD OF ARG PARSER (or us a combination??)"""

#endregion

#region PARAMETERS

print(args)
use_gpu = args.use_gpu
_load_model = args._load_model
_load_array = args._load_array
_train_load_model = args._train_load_model
data_folder = args.data_folder
use_ascii = args.use_ascii

cutoff = 30
val_split = 0.7
decay = 1.1e-5  # Good to stop oscillation
lr = 0.025  # 0.025 seems to be great
epochs = 3000
batch_size = 1000
momentum = 0.5
GAN_epochs = 100
resample_method = 'ALL'
fix_samples = 'NOISE'
equalize = True
_save_array = True
_plot_performance = True
_generate_GAN = False


"""I need to set CUDA before imports!!!"""
import os
if not use_gpu:
    """Use gpu if you have many parameters in your model"""
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    print('Using cpu...')
else:
    print('Using gpu...')

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
import numpy as np
import pandas as pd
from load_data import load_training
import tensorflow as tf
import time
import matplotlib.pyplot as plt
import matplotlib.transforms
import cv2
from keras.models import load_model
import pickle
from sklearn import metrics  # For ROC curve
from keras import backend as K  # for layer viz

#endregion

#region CREATING MODEL
########################################################################################################################

"""Discovered: output shape = cutoff-kernel+1"""
print('Creating model...')
model = Sequential()
model.add(Convolution1D(254, 20, input_shape=(cutoff, 1), activation='relu'))
model.add(GaussianNoise(0.03))  # Maybe try with noise to reduce overfit?
model.add(MaxPooling1D())
model.add(Dropout(0.2))
model.add(UpSampling1D(5))
model.add(Convolution1D(128, 25, activation='relu'))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(10, activation='relu'))
model.add(Dense(5, activation='sigmoid'))
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
        X,Y = load_from_file('loaded_array.npz')
    except:
        print('Cannot load numpy file')
        _load_array = False

if not _load_array:
    X, Y = load_training(cutoff,
                         data_folder,
                         resample_method=resample_method,
                         fix_samples=fix_samples,
                         equalize=equalize,
                         save_array=_save_array)

#time.sleep(2)  # I want to see what data has loaded
X.astype('float64')
Y.astype('float64')

print('Shuffling data...')
np.random.seed(1) # fix random seed for reproducing results
s = np.arange(X.shape[0])
np.random.shuffle(s)
X = X[s][:]
Y = Y[s]

print('Preprocessing data...')
# Preprocessing data
X = X.reshape(X.shape[0], X.shape[1], 1)
if use_ascii:
    X /= 90
elif not use_ascii:
    X /= 20
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

#region TRAINING MODEL
########################################################################################################################

print('Training model...')
# Fit the model
t = time.time()

# Recording history for ending training early
class history_recorder(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.history = {}
        self.history['loss'] = []
        self.history['binary_accuracy'] = []
        self.history['val_loss'] = []
        self.history['val_binary_accuracy'] = []

    def on_epoch_end(self, epoch, logs={}):
        self.history['loss'].append(logs.get('loss'))
        self.history['binary_accuracy'].append(logs.get('binary_accuracy'))
        self.history['val_loss'].append(logs.get('val_loss'))
        self.history['val_binary_accuracy'].append(logs.get('val_binary_accuracy'))

# Learning rate scheduler
class learning_rate_scheduler(keras.callbacks.Callback):

    def on_train_begin(self, logs={}):
        self.checkpoint1 = True
        self.checkpoint2 = True
        self.checkpoint3 = True
        self.checkpoint4 = True

    def on_epoch_end(self, epoch, logs={}):

        if logs.get('val_binary_accuracy')>0.88 and self.checkpoint1:
            K.set_value(model.optimizer.lr, 0.02)
            print('\nChanged learning rate to 0.02')
            self.checkpoint1 = False
        elif logs.get('val_binary_accuracy')>0.92 and self.checkpoint2:
            K.set_value(model.optimizer.lr, 0.015)
            print('\nChanged learning rate to 0.015')
            self.checkpoint2 = False
        elif logs.get('val_binary_accuracy') > 0.95 and self.checkpoint3:
            K.set_value(model.optimizer.lr, 0.01)
            print('\nChanged learning rate to 0.01')
            self.checkpoint3 = False
        elif logs.get('val_binary_accuracy') > 0.96 and self.checkpoint4:
            K.set_value(model.optimizer.lr, 0.05)
            print('\nChanged learning rate to 0.05')
            self.checkpoint4 = False

history_callback = history_recorder()
learning_rate_callback = learning_rate_scheduler()
checkpointer = ModelCheckpoint(filepath='checkpoint.h5', verbose=0, save_best_only=True)


if _load_model or _train_load_model:
    try:
        print('Loading model from file...')
        model = load_model('Signal_peptide_model.h5')  # Looks like you can load model and keep training
        with open('history.pkl', 'rb') as pickle_file:
            history = pickle.load(pickle_file)
    except Exception as e:
        print(e)
        print('Cannot load model')
        _load_model = False

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
        model = load_model('checkpoint.h5')  # Remove model and check if model exist later!
        history = history_callback.history

    with open('history.pkl', 'wb') as pickle_file:
        pickle.dump(history, pickle_file, pickle.HIGHEST_PROTOCOL)
    print('It took {0:.5f} seconds to train'.format(time.time()-t))

print('Evaluation of model processing...')
scores = model.evaluate(X_test, Y_test)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

print('Saving model...')
model.save('Signal_peptide_model.h5', overwrite=True)

#endregion

#region PLOTS
########################################################################################################################
"""ADD PRINTS BEFORE PLOTTING"""

#region Accuracy
print('Plotting accuracy...')
# Plotting training
plt.figure()
plt.grid()
plt.plot(history['binary_accuracy'])
plt.plot(history['val_binary_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('train_acc.png')
plt.show(block=False)
#endregion

#region Loss
print('Plotting loss...')
plt.figure()
plt.grid()
plt.plot(history['loss'])
plt.plot(history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('train_loss.png')
plt.show(block=False)
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
heatmap = plt.pcolor(confusion_matrix_normalized, cmap='Blues')  # CAN PLOT WITHOUT NORMALIZATION
cm_labels = [['True positive', 'False negative'], ['False positive', 'True negative']]
'''THIS THING TOOK LIKE A MILLION YEARS TO FIGURE, DONT CHANGE THE DAMN COORDINATES'''
for x in range(2):
    for y in range(2):
        plt.text(y + 0.5, 1.5 - x, "{0}\n{1:.4f}".format(cm_labels[x][y], confusion_matrix_normalized[x][y]),
                ha='center', va='center', fontsize=16)
plt.savefig('train_cm.png')
plt.show(block=False)
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
plt.plot(fpr, tpr, label='Model')
plt.plot([0, 1], [0, 1], label='Baseline', linestyle='dashed')
legend = plt.legend(loc=4, shadow=True)
plt.savefig('train_roc.png')
plt.show(block=False)
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
plt.savefig('train_param.png')
plt.show(block=False)
#endregion

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
plot_model(model, to_file='model.png')
plt.figure()
img=plt.imread('model.png')
imgplot = plt.imshow(img)
plt.tight_layout()
plt.axis('off')
plt.show(block=False)
#endregion

#region Filters
W_1 = model.layers[0].get_weights()
W_1 = np.squeeze(W_1[0],axis=1) # 20, 1, 254 --> 20, 254
W_1 = np.transpose(W_1) # 20, 254 --> 254, 20
plt.figure()
heatmap = plt.imshow(W_1, cmap='bwr') # need to do list in list if 1d dim
plt.colorbar(heatmap, orientation='vertical')
plt.show(block=False)
#endregion

#region Connections
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


"""
for i in range(1,len(layers)-1): # Edges
    print(layers[i].get_weights()[0].shape[1])
    print(layers[i+1].get_weights()[0].shape[1])
    for j in range(layers[i].get_weights()[0].shape[1]):
        node_name1 = str(i)  + '_' + str(j)
        for k in range(layers[i+1].get_weights()[0].shape[1]):
            node_name2 = str(i+1) + '_' + str(k)
            print(node_name1, node_name2)
            G.add_edge(node_name1, node_name2)
"""

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
plt.show(block=False)
#endregion

#endregion

plt.show()
if _generate_GAN:
    #region GAN
    ########################################################################################################################

    print('Creating generator...')
    generator = Sequential()
    generator.add(Convolution1D(254, 20, input_shape=(cutoff, 1), activation='relu'))  # Remember: output = kernel - input
    generator.add(Convolution1D(254, 11, input_shape=(cutoff, 1), activation='relu'))
    generator.add(Flatten())
    generator.add(Dense(30, activation='sigmoid'))
    generator.add(Reshape((30,1), input_shape=(30,)))
    # Input discriminator = (None, 30, 1), need to add dimension
    # Output layer, tanh is recommended but I need positive values
    # DONT FORGET TO CHANGE RANDOM DATA IF YOU CHANGE INPUT SHAPE
    print(generator.summary())
    print('Input shape: ' + str(generator.input_shape))
    print('Output shape: ' + str(generator.output_shape))

    print('Creating discriminator...')  # USE SAME AS PREDICTOR, CHANGE WHEN YOU CHANGE MODEL
    discriminator = Sequential()
    discriminator.add(Convolution1D(254, 20, input_shape=(cutoff, 1), activation='relu'))
    discriminator.add(MaxPooling1D())
    discriminator.add(Dropout(0.2))
    discriminator.add(UpSampling1D(5))
    discriminator.add(Convolution1D(128, 25, activation='relu'))
    discriminator.add(Dropout(0.2))
    discriminator.add(Flatten())
    discriminator.add(Dense(10, activation='relu'))
    discriminator.add(Dense(5, activation='sigmoid'))
    discriminator.add(Dense(1, activation='sigmoid'))

    print('Compiling models...')  # REMEMBER, CHANGE SO THAT THEY HAVE DIFFERENT LR
    sgd = keras.optimizers.SGD(lr=0.03, momentum=momentum, nesterov=True)  # Optimal = 0.03
    discriminator.compile(metrics=['binary_accuracy'],
                        loss='binary_crossentropy',
                        optimizer=sgd)
    sgd = keras.optimizers.SGD(lr=0.4, momentum=momentum, nesterov=True)
    generator.compile(metrics=['binary_accuracy'],
                        loss='binary_crossentropy',
                        optimizer=sgd)
    print("Setting up GAN (generator + discriminator)")
    GAN = Sequential()
    GAN.add(generator)
    discriminator.trainable=False
    GAN.add(discriminator)
    sgd = keras.optimizers.SGD(lr=0.4, momentum=momentum, nesterov=True)
    GAN.compile(loss='binary_crossentropy',
                optimizer=sgd,
                metrics=['binary_accuracy'])
    print(GAN.summary())
    print('Input shape: ' + str(GAN.input_shape))
    print('Output shape: ' + str(GAN.output_shape))

    print('Fixing real data...')
    negative_indices = np.where(Y == Y.argmin())
    X_positive = np.delete(X, list(negative_indices), axis=0)
    Y_positive = np.delete(Y, list(negative_indices), axis=0)

    from query import predict_to_AAseq, predict_to_numbers

    for i in range(GAN_epochs):
        print(f'{i} out of {GAN_epochs} completed')
        print('Generating data...')
        random_data = np.random.random((Y_positive.shape[0], 30, 1))  # Generate data for generator
        false_labels = np.zeros((Y_positive.shape[0], 1))  # Y data (0 = false, 1 = true), for discriminator
        true_labels = np.ones((Y_positive.shape[0], 1))  # Y data (1 = true), for generator
        predictions = generator.predict(random_data)  # Generator tries to create data

        print('Training generator to trick discriminator...')
        discriminator.trainable = False
        GAN.fit(random_data,
                true_labels,
                batch_size=1000,
                epochs=10) # I need to train the generator more, since it is weaker built than discriminator

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

        predictions_seq = predict_to_AAseq(predictions, use_ascii)
        print('Generated sequence: ' + ''.join(next(predictions_seq))) # predictions_seq = generator object
        predictions_numbers = predict_to_numbers(predictions, use_ascii)
        print('Corresponding to: ' + ','.join(next(predictions_numbers)))

    print('Saving generator...')
    model.save('Signal_peptide_generator.h5', overwrite=True)

    plt.show()

    #endregion