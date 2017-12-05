import os
import gzip
from Bio import SeqIO
import glob
import time
import shutil
from read_fasta import read_fasta

def reformat(verbose=True):
    """Loads traning data into a numpy array.
    Ignores files that starts with . since they are config files in ubuntu.
    """
    if verbose:
        print('Loading data...')
        t = time.time()
    cur_dir = os.getcwd()  # Needed to reset working directory
    os.chdir('F:/NCBI proteins')  # Go to data folder
    file_counter = 0  # Just to count amount of data
    sample_counter = 0  # Just to count amount of data
    if os.path.exists("reformatted"):
        print('reformatted folder exists')
        shutil.rmtree('reformatted')
        os.makedirs('reformatted')
        print("Removed folder with it's contents")
    else:
        print('reformatted folder does not exists')
        os.makedirs('reformatted')
        print('Created folder')
    amount_of_files = len(glob.glob("*.gz"))
    print(f'Loading {amount_of_files} files')
    for file in glob.glob("*.gz"):
        file_handle = gzip.open(file, mode="r")
        names, sequences = read_fasta(file_handle.read().decode('utf-8'))
        found_peptide=False
        for name, seq in zip(names, sequences):
            if 'signal peptide' not in name:
                continue
            with open('reformatted/'+str(os.path.splitext(file)[0])+'_formatted.fasta', 'a') as outfile:
                outfile.write(name + '\n')
                outfile.write(str(seq) + '\n')
            sample_counter += 1
        file_counter += 1
        print(f'{file_counter} out of {amount_of_files} done', end='\r')
    os.chdir(cur_dir)
    if verbose:
        print(f'Loaded {file_counter} files')
        print(f'Loaded {sample_counter} samples')
        print('It took {0:.5f} seconds to load'.format(time.time()-t))

reformat()