"""After quite some fiddling with creating config files and fiddling with different modules, why not just
have all the settings in a python file and import it? That is such a simple solution :)"""

#region Train
# GPU
use_gpu = True  # Only works if you have tensorflow-gpu installed

# DATA
load_saved_array = True  # Loads array from file
data_folder = '../data/training_data'  # Folders are choosen relative from NN.py
equalize_data = True  # Makes amount of negative and positive data equal
save_array = True  # Save array to file
seq_length = 30  # Length of sequence that will be analysed
data_augmentation = False  # Will divide
fix_samples = 'IGNORE'
vectorize = True
use_ascii = False

# MODEL
load_saved_model = True
train_loaded_model = False
choosen_model = 2  #

# TRAINING DATA
train_test_split = 0.8  # Splits full data set into train and test set, percentage train:test
train_val_split = 0.8  # Splits train data set into train and validation set, percentage train:validation
limit_samples = None # None if you don't want to limit the amount of samples

# TRAINING
lr = 0.0001  # More parameters --> change lr
decay = 1.1e-5
momentum = 0.5
epochs = 3000
batch_size = 4096
use_lr_scheduler = True
lr_scheduler_plan = {}  # REMEMBER TO CHANGE THIS WHEN CHANGING MODEL AND LR
config_plot_realtime = True
plot_realtime_interval = 1

# PLOTTING
plot_performance = True

# GAN
train_GAN = True
load_trained_GAN = True
GAN_lr = 0.0001
GAN_epochs = 3000
GAN_batch_size = 512
GAN_limit_samples = None  # OBS: Depends on first limit samples
#endregion

#region Query
query_data_folder = '../data/query_data/'
#endregion