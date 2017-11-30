from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import LSTM
from keras.layers import SimpleRNN
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Convolution1D, MaxPooling1D
import numpy as np
import pandas as pd
from load_data_test import load_training

dim = 500
set_division = 0.7

# Load training
X, Y = load_training(dim)

# Preprocess data, shuffles and divides
np.random.seed(1) # fix random seed for reproducing results
length_divide = round(X.shape[0]*set_division)
s = np.arange(X.shape[0])
np.random.shuffle(s)
X = X[s][:]
Y = Y[s]

# X data set
X_train, X_test = X[:length_divide], X[length_divide:]
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
X_train.astype('float32')
X_test.astype('float32')
X_train /= 255
X_test /= 255

# Y data set
Y_train, Y_test = Y[:length_divide], Y[length_divide:]
Y_train.astype('float32')
Y_test.astype('float32')
Y_train /= 255
Y_test /= 255

# Create model
model = Sequential()
model.add(Convolution1D(32, 10, input_shape=(dim, 1), activation='relu'))
model.add(Convolution1D(32, 10, activation='relu'))
model.add(Dropout(0.1))
model.add(Flatten())
model.add(Dense(100, activation='softmax'))
model.add(Dense(1, activation='relu'))

#Compile model
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Print model
print(model.summary())

# Fit the model
model.fit(X_train, Y_train, epochs=20, batch_size=10, shuffle=True, verbose=1)

# evaluate the model
scores = model.evaluate(X_test, Y_test)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# calculate predictions
predictions = model.predict(X_train)

# Save model
model.save('model.h5')
