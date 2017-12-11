import os
import tarfile
from Bio import SeqIO
import numpy as np
import math
import time
import random
import sys

def check_region(region):
    if any(x in region for x in ['c','n','h','C']):
        return np.array([1])
    else:
        return np.array([0])


def progress(file_counter, total_file_count, sample_counter):
    print(f"""{file_counter} out of {total_file_count} files loaded, {sample_counter} samples loaded.""",
          end='\r')

def load_training(cutoff, data_folder, resample_method='ALL',
                  fix_samples='IGNORE', _equalize_data=False, save_array=True,
                  use_ascii=True):
    """Loads traning data into a numpy array.
    Ignores files that starts with . since they are config files in ubuntu.
    """
    print('Loading data...')
    t = time.time()
    X = np.empty((0, cutoff))  # Creates empty 2d matrix, where columns are
    Y = np.empty((0, 1))  # Empty vector
    AA_list = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
    AA_dict = {'R': 0, 'H': 1, 'K': 2, 'D': 3, 'E': 4, 'S': 5, 'T': 6, 'N': 7, 'Q': 8, 'C': 9,
               'G': 10, 'P': 11, 'A': 12, 'V': 13, 'I': 14, 'L': 15, 'M': 16, 'F': 17, 'Y': 18, 'W': 19}
    cur_dir = os.getcwd()  # Needed to reset working directory
    os.chdir(data_folder)  # Go to data folder
    sample_counter = 0  # Just to count amount of data
    file_counter = 0
    total_file_count = 0  # Count total amount of files
    for dirpath, dirnames, files in os.walk(os.getcwd(), topdown=True):
        dirnames[:] = [x for x in dirnames if not x.startswith('.')]
        total_file_count += len([x for x in files if not x.startswith('.')])

    progress(file_counter, total_file_count, sample_counter)
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
                try:
                    full_region = list(record[1])
                except IndexError:
                    if 'positive' in dirpath:
                        """No region, assume the first bases are signal peptide"""
                        full_region = ['h']*cutoff + ['-']*(cutoff-len(full_seq))
                        if len(full_region) > len(full_seq):
                            full_region = full_region[:len(full_seq)]
                    else:
                        full_region = ['-']*cutoff + ['-']*(cutoff-len(full_seq))
                        if len(full_region) > len(full_seq):
                            full_region = full_region[:len(full_seq)]

                # Divide into smaller pieces
                seqs = [full_seq[x:x + cutoff] for x in range(0, len(full_seq), cutoff)]
                regions = [full_region[x:x + cutoff] for x in range(0, len(full_region), cutoff)]

                if resample_method == 'FIRST':
                    seqs = [seqs[0]]
                    regions = [regions[0]]
                elif resample_method == 'ALL':
                    pass
                else:
                    print('No resample method has been choosen')
                    quit()

                if fix_samples == 'LOOP_SEQ':
                    seqs = [list(x) + (full_seq*(math.ceil(cutoff/(len(full_seq)))))[:cutoff-len(x)]
                            if len(x) < cutoff
                            else x for x in seqs]
                    regions = [list(x) + (full_region*(math.ceil(cutoff/(len(full_region)))))[:cutoff-len(x)]
                               if len(x)<cutoff
                               else x for x in regions]
                elif fix_samples == 'ZERO':
                    seqs = [list(x) + [0]*(cutoff-len(x))
                            if len(x) < cutoff
                            else x for x in seqs]
                    regions = [list(x) + [0]*(cutoff-len(x))
                               if len(x) < cutoff
                               else x for x in regions]
                elif fix_samples == 'IGNORE':
                    seqs = [x for x in seqs
                            if len(x) == cutoff]
                    regions = [x for x in regions
                               if len(x) == cutoff]
                elif fix_samples == 'NOISE':
                    seqs = [x + random.choices(AA_list, k=(cutoff-len(x)))
                            if len(x) < cutoff
                            else x for x in seqs]
                    regions = [x + ['-']*(cutoff-len(x))
                               if len(x) < cutoff
                               else x for x in regions]

                if use_ascii:
                    # Using ascii numbers, ord('A') = 65
                    """Doing this sped up the process by 20 fold!"""
                    for i,j in enumerate(seqs):
                        seqs[i] = [ord(x) - 65 for x in j]
                elif not use_ascii:
                    # Using ascii numbers, ord('A') = 65
                    """Doing this sped up the process by 20 fold!"""
                    for i,j in enumerate(seqs):
                        seqs[i] = [AA_dict[x] for x in j]

                for seq, region in zip(seqs, regions):
                    X = np.vstack((X, np.array(seq)))
                    Y = np.vstack((Y, check_region(region)))
                    sample_counter += 1
                    # Slows performance, but I still like it here
                    progress(file_counter, total_file_count, sample_counter)

            file_counter += 1
            progress(file_counter, total_file_count, sample_counter)

            """Can be used in future to find which data was tm or not"""
            #print(os.path.basename(dirpath))
            """For neg or pos"""
            #print(os.path.basename(dirpath))

    os.chdir(cur_dir)
    print('')
    print(f'Loaded {sample_counter} samples')
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
        os.remove('storage/loaded_array.npz')
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