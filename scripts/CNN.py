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

dim = 50
set_division = 0.7

# Load training
X, Y = load_training(dim, verbose=True)
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

# X data set
X = X.reshape(X.shape[0], X.shape[1], 1)
X /= 90
length_divide = round(X.shape[0]*set_division)
X_train, X_test = X[:length_divide], X[length_divide:]

# Y data set
Y_train, Y_test = Y[:length_divide], Y[length_divide:]

print('Creating model...')
# Create model
model = Sequential()
model.add(Convolution1D(128, 10, input_shape=(dim, 1), activation='relu'))
model.add(MaxPooling1D()) # Downsample by 2x
model.add(Convolution1D(128, 5, activation='relu'))
model.add(MaxPooling1D())
model.add(Convolution1D(128, 3, activation='relu'))
model.add(MaxPooling1D())
model.add(Dropout(0.50))
model.add(Flatten())
model.add(Dense(5, activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))

print('Compiling model...')
#Compile model
sgd = keras.optimizers.SGD(lr=0.03, decay=1e-6, nesterov=True, momentum=0.9)
model.compile(metrics=['binary_accuracy'],
              loss='binary_crossentropy',
              optimizer=sgd)

# Print model
print(model.summary())
print(model.input_shape)
print(model.output_shape)

# Fit the model
try:
    history = model.fit(X, Y, validation_split=0.3, epochs=300, batch_size=20, shuffle=True, verbose=1)
except KeyboardInterrupt:
    print('\nStopped training...')

print('Evaluation of model processing...')
# evaluate the model
scores = model.evaluate(X_test, Y_test)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# calculate predictions
predictions = model.predict(X_test)

print('Saving model...')
# Saving model
model.save('model.h5')


"""NOTE:
Increasing loss in validation, while increasing validation accuracy indicates
overfitting!!!
"""

print('Plotting data...')
# Plotting training
import matplotlib.pyplot as plt
# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['binary_accuracy'])
plt.plot(history.history['val_binary_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


