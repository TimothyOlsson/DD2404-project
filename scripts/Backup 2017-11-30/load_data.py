import os
import tarfile
from Bio import SeqIO
import numpy as np
import math

def load_training(train_length, verbose=False):
    """Loads traning data
    Ignores files that starts with . since they are config files in ubuntu.
    """
    if verbose:
        print('Loading data...')
    
    x = np.empty((0, train_length))
    y = np.empty((0, 1))
    y_pos = np.array([1])
    y_neg = np.array([0])
    cur_dir = os.getcwd()
    os.chdir('../data/training_data')
    data_counter = 0

    for (dirpath, dirnames, filenames) in os.walk(os.getcwd()):
        for filename in filenames:
            if filename.startswith('.'):
                continue
            records = SeqIO.parse(dirpath + '/' + filename, 'fasta')
            for record in records:
                record = str(record.seq)
                record = record.split('#')
                seq = record[0]
                region = record[1]

                # ord('A') = 65
                seq = [ord(x)-65 for x in seq]
                
                """#FILL WITH ZEROS, NOTE: RIGHT NOW, YOU JUST TAKE FIRST X AND FILLS REST WITH 0
                if len(seq) < train_length:
                    for i in range(train_length - len(seq)):
                        seq.append(0)
                """
                
                #LOOP SEQ
                if len(seq) < train_length:
                    seq = seq + seq*(math.floor(train_length - len(seq))) # Double list if possible
                    fill = train_length - len(seq) # Find how many elements left to fill
                    seq = seq + seq[:fill]
                
                        
                seq = np.array(seq[:train_length])
                x = np.vstack((x, seq))
                if os.path.dirname(dirpath).endswith('positive_examples'):
                    y = np.vstack((y, y_pos))
                else:
                    y = np.vstack((y, y_neg))
                data_counter += 1
                """Can be used in future to find which data was tm or not"""
                #print(os.path.basename(dirpath))

                
    os.chdir(cur_dir)
    if verbose:
        print(f'Loaded {data_counter} data sets')
        print('Pos values starts at: ' + str(np.argmax(y)))
    return x, y

def load_tar(): # WIP
    """Loads a tar file and creates a numpy array
    Ignores files that starts with . since they are config files in ubuntu.
    Note: this is probably slower than loading from extracted files,
    since the tar function needs to extract each file
    """
    os.chdir('../data')
    tar = tarfile.open("spdata.tar.gz", "r:gz")
    for tar_member in tar.getmembers():
        if os.path.split(tar_member.name)[1].startswith('.'):
            continue
        f = tar.extractfile(tar_member)
        print(f)
        print(tar_member.name)
        if f is not None:
            read_seq(f)
            data = np.loadtxt(content)
