# Multi-Recurrent-Neural-Networks-for-predicting-breast-cancer
The Multi_RNN.py contains a class that classifies on Breast Cancer dataset using Multiple Recurrent Neural Network. 

Dataset here: https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data.

The goal was to predict the 'severity' of the cancer by feeding to a Recurrent Neural Network 30 important characteristics(ex: radius_mean, texture_mean, perimeter_mean) variables. The are 30 independent variables (excluding the 'id' column) and one categorical dependent variable 'diagnosis'. I've decided not to perform Exploratory Data Analysis because, the main priority was to build a decent Recurrent Neural Network Classifier to predict the severity of the cancer.

For better readability of the code I've used the name_scope convention (one of tensor flow's function) to differentiate levels of neural network and visualize each stage on the tensorboard later. 

 
 Multiple cells(3) of  Recurrent Neural Networks with 100 neurons each were used for prediction. A "Dynamic_RNN" function of the tensorflow API was used to unroll multiple recurrent neural network cells through time.
 
 
Hyper Parameters: There are many hyper parameters to be tuned in a neural network. I've tried my best to experiment with as many hyper parameters as I could and choosen the ones which are better suited for this data set:

1)Initialization: I've tested a proven Initialization strategy which is better than the default implementation of Xavier Initialization, the 'He Initialization'. Although the accuracy didn't change much the 'He Initialization' considerably increased the training speed of the algorithm. So, we've settled for He Initialization.

2)Activation Function: I've used the most suitable activation function for an RNN, the tanh activation function. Other activation functions were note tested.

3)Normalization:Implemented a standard StandardScalar normalization.

4)Regularization:I've used "DROPOUT" as the regularization parameter. I've tested on both cell wise (dropping each recurrent neural network layer randomly) by using "DROPOUT WRAPPER" of tensor flow and neuron wise dropout(dropping neurons in individual cells). The neuron level dropout strategy was performing far better than cell wise dropout strategy.I've started from a high drop out rate of 0.5 (probability of dropping a neuron while training) it turned out that the model was underfitting the data. A drop out probability of 0.12 worked best for this model.

5)Optimization: I've tried a variety of optimization techniques ranging from the standard GradientDescentOptimizer to Momentum, NesterovGradientDescent and Adam. The AdamOptimizer with a learning rate of 0.001 worked the best of all.



