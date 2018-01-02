"""After quite some fiddling with creating config files and fiddling with different modules, why not just
have all the settings in a python file and import it? That is such a simple solution :)"""

# GPU
use_gpu = True

# DATA
load_saved_array = False
data_folder = '../data/training_data'
equalize_data = True
save_array = True
cutoff = 30
data_augmentation = False
fix_samples = 'IGNORE'
vectorize = True
use_ascii = False

# MODEL
load_saved_model = False
train_loaded_model = False

# TRAINING
lr = 0.005  # More parameters --> change lr
decay = 1.1e-5
momentum = 0.5
val_split = 0.7
epochs = 3000
batch_size = 258
use_lr_scheduler = True
lr_scheduler_plan = {0.88: 0.002, 0.92: 0.001, 0.93: 0.0005}  # REMEMBER TO CHANGE THIS WHEN CHANGING MODEL AND LR
config_plot_realtime = True
plot_realtime_interval = 10

# PLOTTING
plot_performance = False

# GAN
train_GAN = True
load_trained_GAN = True
GAN_lr = 0.03
GAN_epochs = 3000
GAN_batch_size = 256