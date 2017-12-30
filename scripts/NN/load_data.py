import os
import tarfile
from Bio import SeqIO
import numpy as np
import math
import time
import random
import sys

AA_list = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
AA_to_int = {'X': 0, 'R': 1, 'H': 2, 'K': 3, 'D': 4, 'E': 5, 'S': 6, 'T': 7, 'N': 8, 'Q': 9, 'C': 10,
             'G': 11, 'P': 12, 'A': 13, 'V': 14, 'I': 15, 'L': 16, 'M': 17, 'F': 18, 'Y': 19, 'W': 20}

def check_region(region):
    if any(x in region for x in ['c','n','h','C']):
        return np.array([1])
    else:
        return np.array([0])

def progress(file_counter, total_file_count, sample_counter):
    print(f"""{file_counter} out of {total_file_count} files loaded, {sample_counter} samples loaded.""",
          end='\r')

def one_hot_endcoding(vector):
    # Stupid keras....
    # TO FIGURE OUT THIS BULLSHIT TOOK SO LONG TIME, I FIRST THOUGHT IT WAS NUMPY BUT NOOOO....
    for i,j in enumerate(vector):
        _hot = [0]*len(AA_to_int.keys())
        _hot[AA_to_int[j]] = 1. # Add 1. at correct index
        vector[i] = _hot
    return vector

def load_training(cutoff, data_folder, data_augmentation=False,
                  fix_samples='IGNORE', _equalize_data=False, save_array=True,
                  use_ascii=False, vectorize=True):
    """Loads traning data into a numpy array.
    Ignores files that starts with . since they are config files in ubuntu.
    """
    print('Loading data...')
    t = time.time()
    cur_dir = os.getcwd()  # Needed to reset working directory
    os.chdir(data_folder)  # Go to data folder
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
                record = str(record.seq)
                record = record.split('#')
                full_seq = list(record[0])
                # Discard bad data
                if len(full_seq) < 2:
                    continue

                # The first amino acid is usually M or not in signal peptide. Ignore it
                full_seq = full_seq[1:]

                # seqs = list in list
                if not data_augmentation:
                    seqs = [full_seq[:cutoff]]
                elif data_augmentation:
                    # Divide into smaller pieces
                    seqs = [full_seq[x:x + cutoff] for x in range(0, len(full_seq), cutoff)]
                else:
                    print('No resample method has been choosen')
                    quit()

                if fix_samples == 'LOOP_SEQ':
                    seqs = [list(x) + (full_seq*(math.ceil(cutoff/(len(full_seq)))))[:cutoff-len(x)]
                            if len(x) < cutoff
                            else x for x in seqs]
                elif fix_samples == 'ZERO':
                    seqs = [list(x) + [0]*(cutoff-len(x))
                            if len(x) < cutoff
                            else x for x in seqs]
                elif fix_samples == 'IGNORE':
                    seqs = [x for x in seqs
                            if len(x) == cutoff]
                    if seqs == []: # Check for empty lists
                        continue
                elif fix_samples == 'NOISE':
                    seqs = [x + random.choices(AA_list, k=(cutoff-len(x)))
                            if len(x) < cutoff
                            else x for x in seqs]

                # Fix Y
                if 'positive' in dirpath:
                    """No region, assume the first bases are the signal peptide"""
                    for i in range(len(seqs)):
                        if i == 0:
                            big_label_list.append([1])
                        else:  # When doing data augmentation, this is needed
                            big_label_list.append([0])
                elif 'negative' in dirpath:
                    for i in range(len(seqs)):
                        big_label_list.append([0])
                else:
                    # ADD MORE THINGS HERE
                    print('ERROR, not negative or positive list')
                    print(dirpath)
                    quit()

                # Fix X
                if vectorize:
                    for i,j in enumerate(seqs):
                        seqs[i] = one_hot_endcoding(j)
                elif use_ascii:
                    # Using ascii numbers, ord('A') = 65
                    """Doing this sped up the process by 20 fold!"""
                    for i,j in enumerate(seqs):
                        seqs[i] = [float(ord(x)) - 65 for x in j]
                elif not use_ascii:
                    # Using ascii numbers, ord('A') = 65
                    """Doing this sped up the process by 20 fold!"""
                    for i,j in enumerate(seqs):
                        seqs[i] = [float(AA_to_int[x])
                                   if x in AA_to_int.keys()
                                   else 0  # Fix unknown amino acids
                                   for x in j]

                big_seq_list.append(seqs)
                sample_counter += 1
                # Slows performance, but I still like it here
                #progress(file_counter, total_file_count, sample_counter)

            file_counter += 1
            progress(file_counter, total_file_count, sample_counter)

            """Can be used in future to find which data was tm or not"""
            #print(os.path.basename(dirpath))
            """For neg or pos"""
            #print(os.path.basename(dirpath))

    # Needs to flatten big_seq_list, since it is now a 3 matrix
    print('')
    print(f'Loaded {sample_counter} samples')
    big_seq_list = sum(big_seq_list, [])  # Flattens list, needed since the code needs list in lists for data aug
    X = np.array(big_seq_list, dtype=np.float32)   # THIS DOES NOT WORK FOR VECTORIZATION, NEEDS MORE PROCESSING
    Y = np.array(big_label_list, dtype=np.float32)
    if not vectorize:
        X = X.reshape(X.shape[0], X.shape[1], 1) # Reshape, need 3d for CNN
    os.chdir(cur_dir)
    print('{} positive samples and {} negative samples'.format(np.count_nonzero(Y), Y.shape[0]-np.count_nonzero(Y)))
    print('It took {0:.5f} seconds to load'.format(time.time()-t))
    #print('Positive values starts at: ' + str(np.argmax(Y)))
    t = time.time()
    if _equalize_data:
        amount_positive = np.count_nonzero(Y)
        amount_negative = Y.shape[0] - amount_positive
        removed_samples = 0
        amount_to_remove = 0
        indices = []
        if amount_positive > amount_negative:
            amount_to_remove = amount_positive - amount_negative
            indices = np.where(Y == Y.argmax())[0][:amount_to_remove]
            print(f'More positive than negative samples. Removing {amount_to_remove} positive samples')
        elif amount_positive <= amount_negative:
            amount_to_remove = amount_negative - amount_positive
            indices = np.where(Y == Y.argmin())[0][:amount_to_remove]
            print(f'More negative than positive samples. Removing {amount_to_remove} negative samples')
        X = np.delete(X, list(indices), axis=0)
        Y = np.delete(Y, list(indices), axis=0)
        removed_samples=len(indices)
        print(f'Equalized, removed {removed_samples} samples')
        print('{} positive samples and {} negative samples'.format(np.count_nonzero(Y),
                                                                    Y.shape[0] - np.count_nonzero(Y)))
        print('It took {0:.5f} seconds to load'.format(time.time() - t))

    if save_array:
        print('Saving array to file...')
        np.savez('storage/loaded_array.npz', X, Y)

    return X, Y

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

def load_from_file(file_name):
    t = time.time()
    npz_file = np.load(file_name)
    data = dict(zip(("X", "Y"), (npz_file[x] for x in npz_file)))
    X = data['X']
    Y = data['Y']
    print('Loaded arrays from file')
    print(f'Loaded {Y.shape[0]} samples')
    print('{} positive samples and {} negative samples'.format(np.count_nonzero(Y), Y.shape[0] - np.count_nonzero(Y)))
    print('It took {0:.5f} seconds to load'.format(time.time() - t))
    return X, Y