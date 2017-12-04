# PARAMETERS
cutoff = 30
val_split = 0.2
decay = 1e-6
lr = 0.05
epochs = 100
batch_size = 1000
use_cpu = True
resample_method='FIRST'
fix_samples = 'NOISE'

parameters = {'cutoff': cutoff, 'val_split': val_split, 'decay': decay,
              'lr': lr, 'epochs': epochs, 'batch_size': batch_size, 'use_cpu': use_cpu}

"""I need to set CUDA before imports!!!"""
import os
if use_cpu:
    """Use gpu if you have many parameters in your model"""
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    print('Using cpu...')
else:
    print('Using gpu...')

print('Importing modules...')
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import LSTM
from keras.layers import SimpleRNN
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Convolution1D, MaxPooling1D
#from keras.callbacks import History
import numpy as np
import pandas as pd
from load_data import load_training
import tensorflow as tf
import time
import matplotlib.pyplot as plt
import matplotlib.transforms

print('Creating model...')
# Create model
model = Sequential()
model.add(Convolution1D(20, 5, input_shape=(cutoff, 1), activation='relu'))
model.add(MaxPooling1D()) # Downsample by 2x
model.add(Convolution1D(15, 5, activation='relu'))
#model.add(MaxPooling1D()) # Downsample by 2x
model.add(Convolution1D(10, 3, activation='relu'))
model.add(MaxPooling1D())
model.add(Dropout(0.50))
model.add(Flatten())
model.add(Dense(5, activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))

print('Compiling model...')
#Compile model
sgd = keras.optimizers.SGD(lr=lr, decay=decay)
model.compile(metrics=['binary_accuracy'],
              loss='binary_crossentropy',
              optimizer=sgd)

print('Showing model...')
# Print model
print(model.summary())

# Load training, prints already in script
X, Y = load_training(cutoff, resample_method=resample_method, fix_samples=fix_samples)
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

print('Training model...')
# Fit the model
t = time.time()
history = model.fit(X, Y, validation_split=val_split, epochs=epochs, batch_size=batch_size, shuffle=True, verbose=1)
print('It took {0:.5f} seconds to train'.format(time.time()-t))

print('Evaluation of model processing...')
# evaluate the model
scores = model.evaluate(X, Y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

print('Saving model...')
# Saving model
model.save('model.h5')

print('Plotting results...')
# Plotting training
# list all data in history
#print(history.history.keys())
# summarize history for accuracy
plt.figure()
plt.plot(history.history['binary_accuracy'])
plt.plot(history.history['val_binary_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show(block=False)

# summarize history for loss
plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show(block=False)

# Plot parameters
plt.figure()
table = plt.table(cellText=[[x] for x in parameters.values()],
                 rowLabels=[x for x in parameters.keys()],
                 rowLoc='right',
                 colWidths=[0.2]*len(parameters.keys()),
                 loc='center')
table.scale(1, 1.1)
plt.axis('off')
plt.show(block=False)

print('Creating confusion matrix...')
# calculate predictions
predictions = model.predict(X)
predictions = np.round(predictions)
confusion_matrix = np.array([[0,0],[0,0]])
for i in range(Y.shape[0]): # All samples
    if predictions[i] == 1 and Y[i] == 1: # True positive
        confusion_matrix[0][0] += 1
    elif predictions[i] == 0 and Y[i] == 0: # True negative
        confusion_matrix[1][1] += 1
    elif predictions[i] == 1 and Y[i] == 0: # False positive
        confusion_matrix[1][0] += 1
    elif predictions[i] == 0 and Y[i] == 1: # False negative
        confusion_matrix[0][1] += 1
    else:
        print('ERROR')

confusion_matrix_normalized = confusion_matrix/Y.shape[0]  # Total number of samples
plt.title('Confusion matrix')
plt.ylabel('True label')
plt.xlabel('Predicted label')
print(confusion_matrix_normalized)
heatmap = plt.pcolor(confusion_matrix_normalized, cmap='Blues')
cm_labels = [['True positive', 'False negative'], ['False positive', 'True negative']]

"""THIS THING TOOK LIKE A MILLION YEARS TO FIGURE, DONT CHANGE. DAMN COORDINATES"""
for x in range(2):
    for y in range(2):
        plt.text(y + 0.5, 1.5 - x, "{0}\n{1:.4f}".format(cm_labels[x][y], confusion_matrix_normalized[x][y]),
                ha='center', va='center', fontsize=16)
plt.show()

with open('predictions.txt', 'w') as outfile:
    for i in predictions.flatten().tolist():
        outfile.write(str(int(i)))
    outfile.write('\n\n')
    for i in Y.flatten().tolist():
        outfile.write(str(int(i)))

"""
summary = model.summary()
W_Input_Hidden = model.layers[0].get_weights()
W_Output_Hidden = model.layers[1].get_weights()
print(summary)
print('INPUT-HIDDEN LAYER WEIGHTS:')
print(W_Input_Hidden)
print('HIDDEN-OUTPUT LAYER WEIGHTS:')
print(W_Output_Hidden)
"""



