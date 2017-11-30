from keras.datasets import mnist

mndata = mnist.load_data()

(x_train, y_train), (x_test, y_test) = mnist.load_data()

import matplotlib.pyplot as plt

plt.imshow(x_test[1], cmap='hot', interpolation='nearest')
plt.show()

