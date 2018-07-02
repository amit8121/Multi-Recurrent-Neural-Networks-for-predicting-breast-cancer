# Multi-Recurrent-Neural-Networks-for-predicting-breast-cancer
The Multi_RNN.py contains a class that classifies on Breast Cancer dataset using Multiple Recurrent Neural Network. 

Dataset here: https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data.

The goal was to predict the 'severity' of the cancer by feeding to a Recurrent Neural Network (ex: RI index, Na index, Mg index etc) variables. I've decided not to perform Exploratory Data Analysis because, the main priority was to build a decent Recurrent Neural Network Classifier to predict the severity of the cancer.

For better readability of the code I've used the name_scope convention (one of tensor flow's function) to differentiate levels of neural network and visualize each stage on the tensorboard later. 

 

Hyper Parameters: There are many hyper parameters to be tuned in a neural network. I've tried my best to experiment with as many hyper parameters as I could and choosen the ones which are better suited for this data set:

1)Initialization: I've tested a proven Initialization strategy which is better than the default implementation of Xavier Initialization, the 'He Initialization'. It turned out (after testing on 1000 epochs and several runs) that Xavier method performed much better than the He Initialization on this data set.

Activation Function: I've used the most suitable activation function for an RNN, the tanh activation function. Other activation functions were note tested.

Normalization: I've Implemented the standard StandardScalar normalization.

Regularization: I've tried to Implement one of the best regularization techniques "DROPOUT". I've started from a reasonable drop out rate of 0.2 (probability of dropping a neuron while training) it turned out that the model was underfitting the data. A drop out probability of 0.15 worked best for this model.

Optimization: I've tried a variety of optimization techniques ranging from the standard GradientDescentOptimizer to Momentum, NesterovGradientDescent and Adam. The AdamOptimizer with a learning rate of 0.001 worked fine.



