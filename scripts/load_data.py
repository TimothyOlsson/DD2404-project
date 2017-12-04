import os
import tarfile
from Bio import SeqIO
import numpy as np
import math
import time
import random

def check_region(region):
    if any(x in region for x in ['c','n','h','C']):
        return np.array([1])
    else:
        return np.array([0])

def load_training(cutoff, verbose=True, resample_method='ALL', fix_samples='IGNORE'):
    """Loads traning data into a numpy array.
    Ignores files that starts with . since they are config files in ubuntu.
    """
    if verbose:
        print('Loading data...')
        t = time.time()
    X = np.empty((0, cutoff))  # Creates empty 2d matrix, where columns are
    Y = np.empty((0, 1))  # Empty vector
    AA_list = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
    cur_dir = os.getcwd()  # Needed to reset working directory
    os.chdir('../data/training_data')  # Go to data folder
    sample_counter = 0  # Just to count amount of data
    file_counter = 0
    total_file_count = 0  # Count total amount of files
    for root, dirs, files in os.walk(os.getcwd()):
        total_file_count += len(files)

    for (dirpath, dirnames, filenames) in os.walk(os.getcwd()):  # Walks through all files and dirs
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
                        """No region, assume 40 bases first are signal peptide"""
                        full_region = ['h']*40 + ['-']*(cutoff-len(full_seq))
                        if len(full_region) > len(full_seq):
                            full_region = full_region[:len(full_seq)]
                    else:
                        full_region = ['-']*40 + ['-']*(cutoff-len(full_seq))
                        if len(full_region) > len(full_seq):
                            full_region = full_region[:len(full_seq)]

                # ord('A') = 65
                full_seq = [ord(x)-65 for x in full_seq]
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
                            if len(x)<cutoff
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
                    seqs = [x + random.sample(AA_list, cutoff-len(x))
                            if len(x) < cutoff
                            else x for x in seqs]
                    regions = [x + ['-']*(cutoff-len(x))
                               if len(x) < cutoff
                               else x for x in regions]


                for seq, region in zip(seqs, regions):
                    X = np.vstack((X, np.array(seq)))
                    Y = np.vstack((Y, check_region(region)))
                    sample_counter += 1

            file_counter += 1
            if verbose:
                print(f'{file_counter} out of {total_file_count} files loaded', end='\r')

                """Can be used in future to find which data was tm or not"""
                #print(os.path.basename(dirpath))
                """For neg or pos"""
                #print(os.path.basename(dirpath))

    os.chdir(cur_dir)
    if verbose:
        print(f'Loaded {sample_counter} samples')
        print('{} positive samples and {} negative samples'.format(np.count_nonzero(Y), Y.shape[0]-np.count_nonzero(Y)))
        print('It took {0:.5f} seconds to load'.format(time.time()-t))
        print('Positive values starts at: ' + str(np.argmax(Y)))
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
