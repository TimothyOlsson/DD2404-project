2017-11-28
Created repo and choose which project to work on

2017-11-29
Created milestones for the project and structured up what to do
Added a notebook.md file where I will write what I do.

2017-11-30
Partially finished a script that loads all the data into a numpy array
Made a prototype for a 1D CNN algorithm with Keras.

I've been thinking how you can load sequences that are not as long as the input length. Right now, the script just fills the remaining with zeros, but I think there is a better way.

Since the CNN are just looking for patterns with CNN filters, it's only local nodes besides each other that are effected. Filling the remaning with zeros will work, but the problem is with overfitting. By chance, the training data that are positive might be shorter and therefore have a higher number of zeros. The script will then train to count the number of zeros instead of finding patterns. Therefore, "looping" the sequence again to fill out the rest might work, since if the sequence does not have the same pattern that makes it negative, another set wont have that too!