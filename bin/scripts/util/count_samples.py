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
rootcwd=cwd

print('Opening GUI to choose folder')
try:
    import tkinter
    from tkinter import filedialog
    cwd = os.getcwd()
    root = tkinter.Tk()
    root.withdraw()
    print('Gui opened')
    dirname = filedialog.askdirectory(parent=root,initialdir=cwd,title='Please select a directory')
    os.chdir(dirname)
except (RuntimeError, TypeError, NameError):
    print('Error. You do not have Tkinter to choose folder. Put script in correct folder to proceed. There will be errros')
    input('Press enter to quit')
    quit()

def progress(file_counter, total_file_count, sample_counter):
    print(f"""{file_counter} out of {total_file_count} files loaded, {sample_counter} samples loaded.""",
          end='\r')

print('Loading data...')
t = time.time()
cur_dir = os.getcwd()  # Needed to reset working directory
os.chdir(dirname)  # Go to data folder
sample_counter = 0  # Just to count amount of data
file_counter = 0
total_file_count = 0  # Count total amount of files
for dirpath, dirnames, files in os.walk(os.getcwd(), topdown=True):
    dirnames[:] = [x for x in dirnames if not x.startswith('.')]
    total_file_count += len([x for x in files if not x.startswith('.')])

progress(file_counter, total_file_count, sample_counter)
big_seq_list = []  # FASTER
big_label_list = []  # FASTER
for (dirpath, dirnames, filenames) in os.walk(os.getcwd(), topdown=True):  # Walks through all files and dirs
    dirnames[:] = [x for x in dirnames if not x.startswith('.')]
    for filename in filenames:
        if filename.startswith('.'):  # Ignore config files
            continue
        records = SeqIO.parse(dirpath + '/' + filename, 'fasta')
        for record in records:
            sample_counter +=1
        file_counter += 1
        progress(file_counter, total_file_count, sample_counter)
print('')
input('Enter to end')
