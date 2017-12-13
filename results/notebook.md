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

#### 12:00
Found a database with signal peptides. I now have about 1500 more positive results, but I will need to add code to the loading script for using them too.

If I use the resampling method, I can get as much as 95 000 samples instead of 2500. The problem is that most of the samples will then be negative, causing the model to fail. I will need a good distribution of positive and negative samples to improve the model further.

It is also possible that my model and my way to predict the signal peptides are not sufficient enough. I will keep trying until I get good enough to see what patterns work and does not work until I determine if this method works or not.

#### 13:30
I made major improvements to the CNN.

A confusion matrix can now be shown (but not plotted yet)

Predictions are saved to a file

Added more prints

Parameters are saved and plotted

#### 13:50
I discovered that there are bacterial signal peptides in the database provided. This means that I can use bacterial signal peptides too. I have found another database with 850 000 (!) proteins. This might be just what I need to train the model.

After training, I have concluded that there are some limitations:

Amount of positive data (I can generate lots of negative data easily with the resampling)

The model (since the acc converges at 95% for training and 70% for validation).

I will try to do a more complicated model when I have all the data I need.

#### 14:15
New problem:
How to download a lot of data at once
How to deal with data where signal peptide location is not disclosed


Solutions:

Download every data that has the word "signal peptide".
Signal peptide length is max 40 AA.
http://www.cbs.dtu.dk/services/SignalP-1.1/sp_lengths.html, where most of them are 20 AA.

Almost all proteins have the signal peptide in the start of the sequence. Use that data to fix more data.

Cutoff should therefore be around 40 to 50 AA.


#### 14:40
I figured out how to download a lot of files from a FTP server. 

Commands:
$webclient = New-Object System.Net.WebClient 
$ftp = ftp://ftp.ncbi.nlm.nih.gov/ncbi-asn1/protein_fasta/README.asn1.protein_fasta
$path = "C:\" # NOTE: If permission denied, choose another folder
$webclient.DownloadFile($ftp,$path)

if multiple files, use *

Might be more difficult than I predicted...

### 2017-12-03
#### 19:40
It was not that hard. Paste in the ftp web link into a explorer and its opened!

I now have 12 Gb of data that I can use for training. However, maybe only 0.5 Gb will be useful, but that is still more than the current amount of data that I have (2 mb). The problem that I have right now is that the computer needs 3 hours to download the files. The data needs to be reformatted to get out the needed files and I consider a script in C++ to be the best way to extract the sequences, due to the large data size.

I have also been thinking in adding a way to add noise to the samples, to prevent overfitting. 


#### 21:40
I have downloaded all the files and extracted all sequences that have signal peptides

#### 22:00
Finished the confusion matrix plotting and have experimented with visualizing the weights and filters in the model. Parameters are also shown in a table

### 2017-12-06
I have missed adding entries to the notebook.

Lot's of things have changed for the model. I have found a good model that I can use for predictions.

The extra data has been proven useful. It has reduced the overfittness and made the model more powerful.

Lots of bugfixed and better plots.


### 2017-12-09
#### 14:40
I have found a model that gives great results. The model has a lot of parameters and takes abouut 50 seconds per epochs with my cpu (i7-2600k oc 4.5 GHz), but less than 0.5 seconds with the gpu. Using the gpu results in about 100 times faster calculations! To train 1000 epochs, gpu would need 500 seconds (about 8 minutes) and the cpu about 50 000 seconds (about 14 hours). It was a great idea to use the gpu for the calculations :) The gpu scales very well with the model, which means that if I double the amount of parameters, the gpu training might take 12 minutes while the cpu might take 30 hours.

#### 20:00
I have made quite some progress. I noticed that adding small amounts of noise made the network less overfit. I have also added a function for checkpoints and early stoppings.

### 2017-12-10
#### 13:30
The real time plotting is a success! However, there are some performance issues.

50 epochs, plot = 37.7 seconds
150 epochs, plot = 130.3 seconds

50 epochs, no plot = 29.6 seconds 
150 epochs, no plot = 83.2 seconds

Maybe spawining a new process that plots will help.

#### 14:20
After some tinkering with multiprocessing, I could not get it to work. The problem lies within multiprocessing and matplotlib. Matplotlib is not thread safe and multiprocessing on windows does not have forking like linux, which means that all processes needs to be added at the bottom of the code (if __name__="main":). This means that I have to divide the code in modules, which I am not ready to do yet.

I need another solution...

#### 14:50
I have found a sufficient solution to my problem.

Instead of drawing the plot multiple times, I will just add points to the plot and extend it. This keeps the performance loss constant throughout the training.

50 epochs, plot = 38.6 seconds
150 epochs, plot = 119.4 seconds

50 epochs, no plot = 29.6 seconds 
150 epochs, no plot = 83.2 seconds

The performance loss is still huge, but the loss will not increase for longer trainings.


#### 15:00
Setting shared axes did not improve performance.

#### 15:20
Fixing axes did not improve performance

#### 15:40
The easy solution is simply to not have the graph plot in real time, but to update it at intervals that you choose. Every 100 epochs works great. The real time plotting has given me some insight how to improve the model and training. I can now reach 95% for the validation set.

#### 17:10
I have been thinking of using a .config file instead of argparser, due to the amount of variables that can be changed. Combining the two seems to be trivial, so I will focus on using one.

#### 17:50
When I implemented the .config file, I noticed how useless it was in this case. Having all parameters in the file is much mor intuitive. The argparser was good, but I believe that it yields too much confusion when using the program.

#### 18:45
The discriminator is way to powerful for the generator. When training, the generator can't fool the discriminator and loss increases rapidly. I believe that the discriminator memorizes the data (due to a large amount of parameters available) and stops generator from improving (i.e its a bad teacher). Ill try doing another model for the discriminator.


### 2017-12-11
I have discovered a fault in my data. By visualizing the filters shows that the CNN mostly keeps it's attention to the first amino acid. The problem is that when I'm doing the data augmentation (mentioned as "resampling" in previous entries), the N terminal always starts M, making the model only look for that property (a CNN that's detecting if it is at the N terminal is pretty useless and even more so if it just have 95% accuracy). This means that I need more negative data.

However, I have 12 Gb of protein data where the majority is negative proteins (i.e proteins without signal peptides). Extracting the data yields way more than 20 Gb of data, so I had to add a function that compresses the data (i.e ignore protein name and cut out the first 40 base pairs).

I constantly hunt for more data to use and I have discovered a way for me to get almost unlimited amounts of positive data! The Signal Peptide website has over 200 000 signal peptides and I wrote a simple scraper in two hours that can parse the whole the whole website and scrape every protein into fasta files. However, I need to be careful with the scraping since the script that I created is too powerful for the website to handle. The script creates 50 workers that work asynchronously that scrapes the data, but it slows down the website by a lot. Maybe I should add delays to the workers to slow down the scraping?

I have also considered changing the way I rank amino acids. At the moment, there is no order in the classification of the amino acids. By setting similar amino acids and their corresponding numbers close to one another, I might be able to improve training. As you need to know for machine learning, bad data in, bad data out.

It takes way too long time to load the data now with more data (around 12 minutes). I need to rewrite the function to improve performance.

I improved performance slightly, reducing one minute.

WOW! I collected everything to one big list and converted to a numpy array instead of appending to numpy array. Now all the data loads in 0.02 seconds (increase by 36 000 times!)


### 2017-12-12
A bunch of things have been completed:

Found an error when loading the data that resulted in lower performance. The script loaded empty sequences (very noisy) and could not handle X as an amino acid.
It resulted in a really bad performance for the network.








