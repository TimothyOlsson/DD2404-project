from load_data_test import load_training

x,y = load_training(50)

import matplotlib.pyplot as plt
seq = x[0][:]
print(seq)
seq = list(seq)
seq = [seq]
plt.imshow(seq, cmap='plasma', interpolation='nearest') # Needs list in list
plt.show()

