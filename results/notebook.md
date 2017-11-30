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
Since the CNN are just looking for patterns with CNN filters, it's only local nodes besides each other that are effected. Filling the remaning with zeros will work, but the problem is with overfitting. By chance, the training data that are positive might be shorter and therefore have a higher number of zeros. The script will then train to count the number of zeros instead of finding patterns. Therefore, "looping" the sequence again to fill out the rest might work, since if the sequence does not have the same pattern that makes it negative, another set wont have that too!

I successfully made the script loop the data. I also finished a script that displays the sequence in a 1 dimensional heatmap to visualize the sequences.

#### 14:00
I can't get the CNN to improve! The loss is abysmal and nothing is working. I fixed a lot of bugs, such as the Y data (if it has a signal peptide or not) was normalized. I also changed how the data is returned from the loading, for better normalization.

#### 15:30
I finally got it to improve! At first, the network was worse than a coinflip (45% accuracy), but after a lot of debugging, it seems that the learning rate was either too high or too low. You have to find the sweet spot to make it improve. I can now get as much as 80% accuracy! As I thought, the network was not the problem, it was one of the parameters.

#### 16:00
It seems that I have the learning rate still too high, since the model cannot reach higher than 0.98 accuracy (it keeps jumping up and down).

The evaluation shows only 57% accuracy, which is the result of overfitting the training data. I need to either restructure the network, find more data or do boostrapping of the data, to get a larger data set.

#### 16:20
I managed to check and get data of training history. I divided the whole set into a training and validation set. When comparing the loss between the validation and the training, it show that it is overfitting by a lot. I saved two graphs where it is shown.

#### 19:30
I managed to get the program to calculate on my GPU! This will speed up the training process by an order of magnitude, allowing me to run more complex models and even longer training sessions.

#### 19:40
I have reached a limit what I can do with my data. There are many paths that I can take now.

I need more data to work with. The current data set is not sufficient for the type of training that I want to do.

I can try to find more data on the web and add it to my data sets. I can resample my data and use as much as possible (by using smaller parts of the whole protein and not just the first part of it, where most signal pepties are). Resampling means that I need to change my code to work with partial sets.

There are many problems with resampling and dividing the proteins into smaller subsets. One is that problems may occur if I cut the protein right on an important region of the protein, causing the NN to give bad results. Another problem is how I can decide if a protein has a signal peptide if only one region (ex c region) is present in the sample. The h region is a better indicator, but does this mean that the output data should have a "percentage" based value if a peptide is present or not? At the moment, there is only 0 (no signal peptide) or 1 (signal peptide present).

I can tackle the first problem by introducing a LSTM region in the model. The LSTM will remember the last sample and store the information for a couple of samples. The difficult part is how this should be implemented or if it's even possible with the program that I'm using.

I should also try to visualize the filters used in the CNN and checking where most of the attention is on the sequence, to find the cause for overfitting.












































