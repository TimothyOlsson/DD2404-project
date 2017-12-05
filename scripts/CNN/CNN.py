# CLI
import argparse
parser = argparse.ArgumentParser(description='CNN')
parser.add_argument('-i', '--input', dest='data_folder', required=False, default='../../data/training_data',
                    help='')
parser.add_argument('--load_array', dest='_preload_array', required=False, default=True, action='store_true',
                    help='')
parser.add_argument('--dont-load_array', dest='_preload_array', required=False, default=True, action='store_false',
                    help='')
parser.add_argument('--load_model', dest='_preload_model', required=False, default=False, action='store_true',
                    help='')
parser.add_argument('--train-load_model', dest='_train_preload_model', required=False, default=False, action='store_true',
                    help='')
parser.add_argument('--use_gpu', dest='use_gpu', required=False, default=False, action='store_true',
                    help='')
args = parser.parse_args()

# PARAMETERS
cutoff = 30
val_split = 0.7
decay = 1e-5
lr = 0.025
epochs = 1000
batch_size = 1000
momentum = 0.5
resample_method='ALL'
fix_samples = 'NOISE'
equalize = True
_save_array = True

print(args)

use_gpu = args.use_gpu
_preload_model = args._preload_model
_preload_array = args._preload_array
_train_preload_model = args._train_preload_model
data_folder = args.data_folder

# kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
#results = cross_val_score(estimator, X, dummy_y, cv=kfold)
#print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

parameters = {'cutoff': cutoff, 'val_split': val_split, 'decay': decay, 'lr': lr,
              'epochs': epochs, 'batch_size': batch_size, 'momentum': momentum, 'use_gpu': use_gpu,
              'resample_method': resample_method, 'fix_samples': fix_samples, 'equalize': equalize,
              'save_array': _save_array}

"""I need to set CUDA before imports!!!"""
import os
if not use_gpu:
    """Use gpu if you have many parameters in your model"""
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    print('Using cpu...')
else:
    print('Using gpu...')


########################################################################################################################
# MODULES

print('Importing modules...')
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import LSTM
from keras.layers import SimpleRNN
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Convolution1D, MaxPooling1D
from keras.layers import UpSampling1D
from keras.layers import Reshape
from keras.callbacks import History, EarlyStopping
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

########################################################################################################################
# MODEL

print('Creating model...')
# Create model
model = Sequential()
model.add(Convolution1D(128, 10, input_shape=(cutoff, 1), activation='relu'))
model.add(Dropout(0.50))
#model.add(UpSampling1D(3))
#model.add(Convolution1D(30, cutoff, activation='relu'))
"""
model.add(Dropout(0.50))
model.add(MaxPooling1D())
model.add(MaxPooling1D(pool_size=2)) # Downsample by 2x
model.add(Dropout(0.50))
model.add(Convolution1D(32, 5, activation='relu'))
model.add(MaxPooling1D())
model.add(Dropout(0.50))
"""
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
# Print model
print(model.summary())
print(model.input_shape)
print(model.output_shape)


########################################################################################################################
# DATA

# Load training, prints already in script
if _preload_array:
    try:
        from load_data import load_from_file
        X,Y = load_from_file('loaded_array.npz')
    except:
        print('Cannot load numpy file')
        _preload_array = False

if not _preload_array:
    X, Y = load_training(cutoff,
                         data_folder,
                         resample_method=resample_method,
                         fix_samples=fix_samples,
                         equalize=equalize,
                         save_array=_save_array)

#time.sleep(2)  # I want to see what data has loaded
X.astype('float32')
Y.astype('float32')

print('Shuffling data...')
# Shuffles and divides data
np.random.seed(1) # fix random seed for reproducing results
s = np.arange(X.shape[0])
np.random.shuffle(s)
X = X[s][:]
Y = Y[s]

print('Preprocessing data...')
# Preprocessing data
X = X.reshape(X.shape[0], X.shape[1], 1)
X /= 90
length_divide = round(X.shape[0]*val_split)
X_train, X_test = X[:length_divide], X[length_divide:] # X data set
Y_train, Y_test = Y[:length_divide], Y[length_divide:] # Y data set
print('Train set: {} positive samples and {} negative samples'.format(np.count_nonzero(Y_train),
                                                                    Y_train.shape[0] - np.count_nonzero(Y_train)))
print('Test set: {} positive samples and {} negative samples'.format(np.count_nonzero(Y_test),
                                                                    Y_test.shape[0] - np.count_nonzero(Y_test)))

########################################################################################################################
# TRAIN

print('Training model...')
# Fit the model
t = time.time()

if _preload_model or _train_preload_model:
    try:
        print('Loading model from file...')
        model = load_model('Signal_peptide_model.h5')  # Looks like you can load model and keep training
        with open('history.pkl', 'rb') as pickle_file:
            history = pickle.load(pickle_file)
    except Exception as e:
        print(e)
        print('Cannot load model')
        _preload_model = False

if (_preload_model == False) or (_train_preload_model == True):
    history = model.fit(X_train, Y_train,
                        validation_data=(X_test, Y_test),
                        epochs=epochs,
                        batch_size=batch_size,
                        shuffle=True,
                        verbose=1)
    history = history.history
    with open('history.pkl', 'wb') as pickle_file:
        pickle.dump(history, pickle_file, pickle.HIGHEST_PROTOCOL)

print('It took {0:.5f} seconds to train'.format(time.time()-t))

print('Evaluation of model processing...')
# evaluate the model
scores = model.evaluate(X_test, Y_test)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

print('Saving model...')
# Saving model
model.save('Signal_peptide_model.h5', overwrite=True)


########################################################################################################################
# PLOTS

print('Plotting results...')
# Plotting training
plt.figure()
plt.plot(history['binary_accuracy'])
plt.plot(history['val_binary_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('train_acc.png')
plt.show(block=False)

# summarize history for loss
plt.figure()
plt.plot(history['loss'])
plt.plot(history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('train_loss.png')
plt.show(block=False)

"""
# Plot parameters
plt.figure()
table = plt.table(cellText=[[x] for x in parameters.values()],
                 rowLabels=[x for x in parameters.keys()],
                 rowLoc='right',
                 colWidths=[0.2]*len(parameters.keys()),
                 loc='center')
table.scale(1, 1.1)
plt.axis('off')
plt.savefig('train_param.png')
plt.show(block=False)
"""

print('Creating confusion matrix on test samples...')
# calculate predictions
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
plt.title('Confusion matrix')
plt.ylabel('True label')
plt.xlabel('Predicted label')
print(confusion_matrix_normalized)
heatmap = plt.pcolor(confusion_matrix_normalized, cmap='Blues')
cm_labels = [['True positive', 'False negative'], ['False positive', 'True negative']]
'''THIS THING TOOK LIKE A MILLION YEARS TO FIGURE, DONT CHANGE. DAMN COORDINATE'''
for x in range(2):
    for y in range(2):
        plt.text(y + 0.5, 1.5 - x, "{0}\n{1:.4f}".format(cm_labels[x][y], confusion_matrix_normalized[x][y]),
                ha='center', va='center', fontsize=16)
plt.savefig('train_cm.png')
plt.show(block=False)

########################################################################################################################
# GAN

print('Creating GAN...')
generator = Sequential()
generator.add(Dense(1, input_dim=1, activation='softmax'))
generator.add(Dense(5, activation='softmax'))
generator.add(Dense(cutoff, activation='softmax'))
generator.add(Reshape((cutoff, 1)))
generator.compile(loss='binary_crossentropy', optimizer='adam')
print(generator.summary())
print(generator.input_shape)
print(generator.output_shape)

print("Setting up combined net (GAN + predictor)")
gen_pred = Sequential()
gen_pred.add(generator)
model.trainable=False
gen_pred.add(model)
sgd = keras.optimizers.SGD(lr=0.1, decay=decay, momentum=momentum, nesterov=True)
gen_pred.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])

print('Generating train data...')
a = np.random.random((100000,1))
b = np.round(a)

print('Training GAN...')
gen_pred.fit(a,b, batch_size=100, epochs=15)

print('Showing predictions in terminal...')
for random in np.arange(0.0, 1.0, 0.1):
    predictions = generator.predict(np.array([random]))
    predictions = [np.round(x*90)+65 for x in predictions]
    predictions = predictions[0]
    predictions = [chr(int(x)) for x in predictions]
    print(predictions)
    print(random, end='\r')
    if round(random) == 0.0:
        print('\r Generated Negative')
    else:
        print('\r Generated Positive')

plt.show()
W_Input_Hidden = model.layers[0].get_weights()
W_Output_Hidden = model.layers[1].get_weights()