from load_data import load_training

x,y = load_training(50, verbose=True, resample_method='FIRST')
data_set = 1334
import matplotlib.pyplot as plt
import time

for i in range(100):
    seq = x[data_set][:]
    val = y[data_set][:]

    seq /= 90

    print(seq)
    print(val)

    seq = list(seq)
    seq = [seq]
    plt.imshow(seq, cmap='binary', interpolation='nearest', aspect='auto') # Needs list in list
    plt.show(block=False)

    data_set += i
    time.sleep(2)
    plt.close()
    #input('Press enter for next entry')
    
