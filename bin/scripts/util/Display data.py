import sys
sys.path.append('..')
from load_data import load_training
import os
import numpy as np
print(os.getcwd())

x, y = load_training(30,
                     '../../../data/query_data/non_tm',
                     data_augmentation=False,
                     fix_samples='ZERO',
                     equalize_data=False,
                     save_array=False,
                     vectorize=True)
data_set = 1334
import matplotlib.pyplot as plt
import time

for i in range(100):
    seq = x[data_set]
    val = y[data_set]
    print(seq.shape)
    print(val.shape)
    plt.imshow(seq, cmap='binary', interpolation='nearest', aspect='auto') # Needs list in list
    # Fix ticks
    AA_to_int = {'X': 0, 'R': 1, 'H': 2, 'K': 3, 'D': 4, 'E': 5, 'S': 6, 'T': 7, 'N': 8, 'Q': 9, 'C': 10,
                 'G': 11, 'P': 12, 'A': 13, 'V': 14, 'I': 15, 'L': 16, 'M': 17, 'F': 18, 'Y': 19, 'W': 20}
    labels = list(AA_to_int.keys())
    #locs, labels = plt.xticks()
    plt.xticks(np.arange(21))
    plt.xticks(np.arange(20), labels)
    plt.show(block=False)
    plt.savefig('results/sample.png')
    data_set += i
    time.sleep(2)
    plt.close()
    #input('Press enter for next entry')
    
