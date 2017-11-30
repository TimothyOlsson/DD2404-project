2017-11-28
Created repo and choose which project to work on

2017-11-29
Created milestones for the project and structured up what to do
Added a notebook.md file where I will write what I do.

2017-11-30
11:00
Partially finished a script that loads all the data into a numpy array
Made a prototype for a 1D CNN algorithm with Keras.

I've been thinking how you can load sequences that are not as long as the input length. Right now, the script just fills the remaining with zeros, but I think there is a better way.

13:00
Since the CNN are just looking for patterns with CNN filters, it's only local nodes besides each other that are effected. Filling the remaning with zeros will work, but the problem is with overfitting. By chance, the training data that are positive might be shorter and therefore have a higher number of zeros. The script will then train to count the number of zeros instead of finding patterns. Therefore, "looping" the sequence again to fill out the rest might work, since if the sequence does not have the same pattern that makes it negative, another set wont have that too!

I successfully made the script loop the data. I also finished a script that displays the sequence in a 1 dimensional heatmap to visualize the sequences.

14:00
I can't get the CNN to improve! The loss is abysmal and nothing is working. I fixed a lot of bugs, such as the Y data (if it has a signal peptide or not) was normalized. I also changed how the data is returned from the loading, for better normalization.

15:30
I finally got it to improve! At first, the network was worse than a coinflip (45% accuracy), but after a lot of debugging, it seems that the learning rate was either too high or too low. You have to find the sweet spot to make it improve. I can now get as much as 80% accuracy! As I thought, the network was not the problem, it was one of the parameters.

16:00
It seems that I have the learning rate still too high, since the model cannot reach higher than 0.98 accuracy (it keeps jumping up and down).

The evaluation shows only 57% accuracy, which is the result of overfitting the training data. I need to either restructure the network, find more data or do boostrapping of the data, to get a larger data set.

16:20
I managed to check and get data of training history. I divided the whole set into a training and validation set. When comparing the loss between the validation and the training, it show that it is overfitting by a lot. I saved two graphs where it is shown


