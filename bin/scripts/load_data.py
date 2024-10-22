import os
import tarfile
from Bio import SeqIO
import numpy as np
import math
import time
import random
import sys
import logging

#region Logging start
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
handler = logging.FileHandler('results/logger.log')
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
#endregion

AA_list = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']

# Custom dictionary, amino acids with similar properties are "close" to one another
AA_to_int = {'X': 0, 'R': 1, 'H': 2, 'K': 3, 'D': 4, 'E': 5, 'S': 6, 'T': 7, 'N': 8, 'Q': 9, 'C': 10,
             'G': 11, 'P': 12, 'A': 13, 'V': 14, 'I': 15, 'L': 16, 'M': 17, 'F': 18, 'Y': 19, 'W': 20}
int_to_AA = {x: y for y, x in AA_to_int.items()}


def check_region(region):
    if any(x in region for x in ['c','n','h','C']):
        return np.array([1])
    else:
        return np.array([0])

def progress(file_counter, total_file_count, sample_counter):
    s = (f"{file_counter} out of {total_file_count} files loaded, "
        f"{sample_counter} samples loaded")
    print(s, end='\r')

def one_hot_endcoding(vector):
    # Stupid keras....
    # TO FIGURE OUT THIS BULLSHIT TOOK SO LONG TIME, I FIRST THOUGHT IT WAS NUMPY BUT NOOOO....
    for i,j in enumerate(vector):
        _hot = [0]*len(AA_to_int.keys())
        if j in AA_to_int.keys():
            _hot[AA_to_int[j]] = 1. # Add 1. at correct index
        else:
            _hot[0] = 1.  # Add X if unknown AA
        vector[i] = _hot
    return vector

def load_training(seq_length, data_folder, data_augmentation=False,
                  fix_samples='IGNORE', equalize_data=False, save_array=True,
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
                    seqs = [full_seq[:seq_length]]
                elif data_augmentation:
                    # Divide into smaller pieces
                    seqs = [full_seq[x:x + seq_length] for x in range(0, len(full_seq), seq_length)]
                else:
                    print('No resample method has been choosen')
                    quit()

                if fix_samples == 'LOOP_SEQ':
                    seqs = [list(x) + (full_seq*(math.ceil(seq_length/(len(full_seq)))))[:seq_length-len(x)]
                            if len(x) < seq_length
                            else x for x in seqs]
                elif fix_samples == 'ZERO':
                    seqs = [list(x) + ['X']*(seq_length-len(x))
                            if len(x) < seq_length
                            else x for x in seqs]
                elif fix_samples == 'IGNORE':
                    seqs = [x for x in seqs
                            if len(x) == seq_length]
                    if seqs == []: # Check for empty lists
                        continue
                elif fix_samples == 'NOISE':
                    seqs = [x + random.choices(AA_list, k=(seq_length-len(x)))
                            if len(x) < seq_length
                            else x for x in seqs]

                # Fix Y
                if 'positive' in dirpath:
                    """No region, assume the first bases are the signal peptide"""
                    for i in range(len(seqs)):
                        if i == 0:
                            big_label_list.append([1.])
                        else:  # When doing data augmentation, this is needed
                            big_label_list.append([0.])
                elif 'negative' in dirpath:
                    for i in range(len(seqs)):
                        big_label_list.append([0.])
                else:
                    # Unknown
                    big_label_list.append([-1.])

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
                for seq in seqs:
                    big_seq_list.append(seq)  # Needed, since data aug breaks
                sample_counter += len(seqs)
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
    logger.info(f'Loaded {sample_counter} samples')
    #print('Flattening...')
    #big_seq_list = sum(big_seq_list, [])  # Flattens list, needed since the code needs list in lists for data aug
    print('Converting to numpy array...')
    X = np.array(big_seq_list, dtype=np.float32)   # THIS DOES NOT WORK FOR VECTORIZATION, NEEDS MORE PROCESSING
    Y = np.array(big_label_list, dtype=np.float32)
    print('Flattening...')
    X = np.squeeze(X)  # WAY faster than using the sum flattening
    if not vectorize:
        X = X.reshape(X.shape[0], X.shape[1], 1) # Reshape, need 3d for CNN
    os.chdir(cur_dir)
    logger.info('Dataset is ' + str(X.nbytes / 1e6) + ' mb in memory')  # X is [samples, time steps, features]
    logger.info('{} positive samples and {} negative samples'.format(np.count_nonzero(Y), Y.shape[0]-np.count_nonzero(Y)))
    logger.info('It took {0:.5f} seconds to load'.format(time.time()-t))
    #print('Positive values starts at: ' + str(np.argmax(Y)))
    t = time.time()
    if equalize_data:
        amount_positive = np.count_nonzero(Y)
        amount_negative = Y.shape[0] - amount_positive
        removed_samples = 0
        amount_to_remove = 0
        indices = []
        if amount_positive > amount_negative:
            amount_to_remove = amount_positive - amount_negative
            # Removes random samples, to prevent bias. (it is read in order)
            indices = random.sample(list(np.nonzero(Y)[0]), amount_to_remove)  # np.where(Y == Y.argmax())[0] DID NOT WORK!!
            logger.info(f'More positive than negative samples. Removing {amount_to_remove} positive samples')
        elif amount_positive <= amount_negative:
            amount_to_remove = amount_negative - amount_positive
            indices = random.sample(list(np.where(Y == 0)[0]), amount_to_remove)
            logger.info(f'More negative than positive samples. Removing {amount_to_remove} negative samples')
        X = np.delete(X, list(indices), axis=0)
        Y = np.delete(Y, list(indices), axis=0)
        removed_samples = len(indices)
        logger.info(f'Equalized, removed {removed_samples} samples')
        logger.info('{} positive samples and {} negative samples'.format(np.count_nonzero(Y),
                                                                         Y.shape[0] - np.count_nonzero(Y)))
        logger.info('It took {0:.5f} seconds to equalize'.format(time.time() - t))

    if save_array:
        logger.info('Saving array to file...')
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

# GAN
def ascii_to_AAseq(predictions):
    predictions = [np.round(x * 90) + 65 for x in predictions]
    for i in range(len(predictions)):
        predictions_seq = predictions[i]
        predictions_seq = [chr(int(x)) for x in predictions_seq]
        yield predictions_seq

def non_ascii_to_AAseq(predictions):
    predictions = [np.round(x * 19) for x in predictions]
    for i in range(len(predictions)):
        predictions_seq = predictions[i]
        predictions_seq = [AA_to_int[int(x)] if 0. <= int(x) <= 19.
                           else 'X'
                           for x in predictions_seq]
        yield predictions_seq

# From text generation, include variation in output
def sample(preds, temperature=0.1):
    """From what I have gathered, the code makes the output more "variable", since if for an example if you get
	Ha, the model might always assume it's Harry while the true word is Hagrid.
	Low temp = conservative, High temp = more creative
	helper function to sample an index from a probability array"""
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds + 1e-8) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def vectorize_to_AAseq(predictions, temperature=0.2):
    for i in range(len(predictions)):  # Note: many vectors
        predictions_seq = predictions[i]
        generated_AA = []
        print(np.round(predictions_seq[0], 3))
        #predictions_seq = np.round(predictions_seq)
        for i,j in enumerate(predictions_seq):
            j = sample(j)  # Randomizes which 1 is taken from the list
            generated_AA.append(j)

            """
            if not 1. in j:  # If no value = 1
                generated_AA.append(0)  # 0 = X
            else:
                j[0] = 0  # Ignore first one, since index just takes first best index
                if not 1. in j: # If only first value is 1
                    generated_AA.append(0)
                else:
                    j = sample(j)  # Randomizes which 1 is taken from the list
                    generated_AA.append(j)
            """

        predictions_seq = [int_to_AA[int(x)] for x in generated_AA]
        yield predictions_seq