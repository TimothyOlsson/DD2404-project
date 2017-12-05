# Create first network with Keras
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import pandas as pd
# fix random seed for reproducibility
np.random.seed(1)

#Loading data
dataset = np.genfromtxt('Cancer_data.csv', delimiter = ',')

# split into input (X) and output (Y) variables
X = dataset[:,2:]
Y = dataset[:,1]

# create model
model = Sequential()
model.add(Dense(30, input_dim=30, activation='relu'))
model.add(Dense(10, activation='sigmoid'))
model.add(Dense(5, activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))

#Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model
model.fit(X, Y, epochs=1600, batch_size=10)

# evaluate the model
scores = model.evaluate(X, Y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# calculate predictions
predictions = model.predict(X)
# round predictions
rounded = [round(x[0]) for x in predictions]
print(rounded)
input('')
