import os
import tarfile
from Bio import SeqIO
import numpy as np
import math
import time
import random
import sys
import logging

#Finds folder location
cwd = os.getcwd()

#saves root cwd
rootcwd = cwd

print('Opening GUI to choose folder')
try:
    import tkinter
    from tkinter import filedialog
    cwd = os.getcwd()
    root = tkinter.Tk()
    root.withdraw()
    print('Gui opened')
    file_path = filedialog.askopenfilename(parent=root,initialdir='../../../data',title='Please select a directory')
    os.chdir(os.path.dirname(file_path))
except (RuntimeError, TypeError, NameError):
    print('Error. You do not have Tkinter to choose folder. Put script in correct folder to proceed. There will be erros')
    input('Press enter to quit')
    quit()

def progress(sample_counter):
    print(f"""{sample_counter} samples loaded.""",
          end='\r')

print('Counting samples')
records = SeqIO.parse(file_path, 'fasta')
for sample_counter, record in enumerate(records):  # count samples
    progress(sample_counter)
total_samples = sample_counter

print('Fixing samples')
print('')
file_counter = 0
division = 0.02
cur_div = division
id_list = []
record_list = []
records = SeqIO.parse(file_path, 'fasta')
for sample_counter, record in enumerate(records):  # count samples
    id_list.append(str(record.id))
    record_list.append(str(record.seq))
    if total_samples*cur_div < sample_counter or sample_counter == total_samples:
        cur_div += division
        with open(f'{os.path.splitext(os.path.basename(file_path))[0]}_split_{file_counter}.fasta', 'w') as file:
            for i,j in zip(id_list, record_list):
                file.write('>' + i + '\n' + j + '\n')
        # Reset lists
        id_list = []
        record_list = []
        file_counter += 1
        print(f'Written file {file_counter}', end='\r')
