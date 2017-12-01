# Notebook - Timothy

## Diary

### 2017-11-28
Created repo and choose which project to work on

### 2017-11-29
Created milestones for the project and structured up what to do
Added a notebook.md file where I will write what I do.

### 2017-11-30
#### 11:00
Partially finished a script that loads all the data into a numpy array
Made a prototype for a 1D CNN algorithm with Keras.

I've been thinking how you can load sequences that are not as long as the input length. Right now, the script just fills the remaining with zeros, but I think there is a better way.

#### 13:00
Since the CNN are just looking for patterns with CNN filters, it's only local nodes besides each other that are effected. Filling the remaining with zeros will work, but the problem is with overfitting. By chance, the training data that are positive might be shorter and therefore have a higher number of zeros. The script will then train to count the number of zeros instead of finding patterns. Therefore, "looping" the sequence again to fill out the rest might work, since if the sequence does not have the same pattern that makes it negative, another set wont have that too!

I successfully made the script loop the data. I also finished a script that displays the sequence in a 1 dimensional heatmap to visualize the sequences.

#### 14:00
I can't get the CNN to improve! The loss is abysmal and nothing is working. I fixed a lot of bugs, such as the Y data (if it has a signal peptide or not) was normalized. I also changed how the data is returned from the loading, for better normalization.

#### 15:30
I finally got it to improve! At first, the network was worse than a coinflip (45% accuracy), but after a lot of debugging, it seems that the learning rate was either too high or too low. You have to find the sweet spot to make it improve. I can now get as much as 80% accuracy! As I thought, the network was not the problem, it was one of the parameters.

#### 16:00
It seems that I have the learning rate still too high, since the model cannot reach higher than 0.98 accuracy (it keeps jumping up and down).

The evaluation shows only 57% accuracy, which is the result of overfitting the training data. I need to either restructure the network, find more data or do bootstrapping of the data, to get a larger data set.

#### 16:20
I managed to check and get data of training history. I divided the whole set into a training and validation set. When comparing the loss between the validation and the training, it shown that it is overfitting by a lot. I saved two graphs where it is shown.

#### 19:30
I managed to get the program to calculate on my GPU! This will speed up the training process by an order of magnitude, allowing me to run more complex models and even longer training sessions.

#### 19:40
I have reached a limit what I can do with my data. There are many paths that I can take now.

I need more data to work with. The current data set is not sufficient for the type of training that I want to do.

I can try to find more data on the web and add it to my data sets. I can resample my data and use as much as possible (by using smaller parts of the whole protein and not just the first part of it, where most signal peptides are). Resampling means that I need to change my code to work with partial sets.

There are many problems with resampling and dividing the proteins into smaller subsets. One is that problems may occur if I cut the protein right on an important region of the protein, causing the NN to give bad results. Another problem is how I can decide if a protein has a signal peptide if only one region (ex c region) is present in the sample. The h region is a better indicator, but does this mean that the output data should have a "percentage" based value if a peptide is present or not? Currently, there is only 0 (no signal peptide) or 1 (signal peptide present).

I can tackle the first problem by introducing a LSTM region in the model. The LSTM will remember the last sample and store the information for a couple of samples. The difficult part is how this should be implemented or if it's even possible with the program that I'm using.

I should also try to visualize the filters used in the CNN and checking where most of the attention is on the sequence, to find the cause for overfitting.


### 2017-12-01
#### 11:30
It seems that the gpu-version of tensorflow is actually slower than the cpu version.

However, this might be that using a small data set with a simple model will yield that the cpu version is better. I set a parameter so you can use gpu or cpu.

I recreated the scripts that loads training data. Yesterday, I was thinking about a way to get or create more data and this is the result. The new method looks at the region and determines if the amino acid sequence is a signal peptide or not for training. (input = sequence, output = 1 if positive, 0 if negative). Instead of using only the first part of the genome where the signal peptide belongs, I can choose if I want to use the full sequence for training or not.

The script cuts up the sequences into smaller parts (parameter "cutoff"). The tricky part is what I will do with samples that are shorter than the cutoff. As I mentioned in a previous entry, there are multiple ways to fix this. I have established three models to fix bad samples:

IGNORE the samples, i.e. discard them
ZERO the samples, i.e. fill the sequence with zeros until it is sufficiently long
LOOP the samples, i.e. stitch up the partial sequence with the start of the full sequence. If the full sequence is shorter than cutoff, add multiple samples until its sufficiently long. 

####12:00
Found a database with signal peptides. I now have about 1500 more positive results, but I will need to add code to the loading script for using them too.

If I use the resampling method, I can get as much as 95 000 samples instead of 2500. The problem is that most of the samples will then be negative, causing the model to fail. I will need a good distribution of positive and negative samples to improve the model further.

It is also possible that my model and my way to predict the signal peptides are not sufficient enough. I will keep trying until I get good enough to see what patterns work and does not work until I determine if this method works or not.










































