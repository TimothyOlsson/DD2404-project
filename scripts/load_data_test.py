import os
import tarfile
from Bio import SeqIO
import numpy as np

def load_training(train_length):
    """Loads traning data
    Ignores files that starts with . since they are config files in ubuntu.
    """
    x_train = np.empty((0, train_length))
    y_train = np.empty((0, 1))
    y_pos = np.array([1])
    y_neg = np.array([0])
    cur_dir = os.getcwd()
    os.chdir('../data/training_data')
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
                #NOTE: RIGHT NOW, YOU JUST TAKE FIRST X AND SKIPS THE REST
                seq = [ord(x) for x in seq]
                if len(seq) < train_length:
                    for i in range(train_length - len(seq)):
                        seq.append(0)
                seq = np.array(seq[:train_length])
                x_train = np.vstack((x_train, seq))
                if os.path.dirname(dirpath).endswith('positive_examples'):
                    y_train = np.vstack((y_train, y_pos))
                else:
                    y_train = np.vstack((y_train, y_neg))
                """Can be used in future to find which data was tm or not"""
                #print(os.path.basename(dirpath))
    os.chdir(cur_dir)
    return x_train, y_train

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
