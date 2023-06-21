The goal of this Assignment is to design ANNs that can classify objects in the CIFAR10 dataset-https://www.cs.toronto.edu/~kriz/cifar.html

The main goal of this assignment is to design neural networks that can successfully (to some degree of accuracy) classify the object
classes in the CIFAR10 data-set.


To activate the venv type <source ./venv/bin/activate>
All ANNs must be trainable within 10 minutes or 15 epochs
To install the required environment and packages type "make"

------------------------------------------------------------------------------
The first network required is a Multi Layer Perceptron saved in the file MLP.py
Basically a standard feed forward MLP with no restrictions as to number of hidden layers network can use.
The challenge was basically to explore a variety of MLP designs.
This is an implementation of a multilayer perceptron (MLP) model using PyTorch on the CIFAR-10 dataset. 
The MLP has two hidden layers with 512 and 256 neurons respectively. 
The model is trained for 15 epochs with a learning rate of 0.015 and a weight decay of 0.0013. 
The optimizer used is Stochastic Gradient Descent (SGD) with momentum 0.9. 
The learning rate is decreased by a factor of 0.15 after every 5 epochs using a step learning rate scheduler. 
The code includes functions to train and test the model, and has options to save and load model parameters. 
The training and testing datasets are loaded using PyTorch's torchvision.datasets package and normalized. 
The MLP uses the ReLU activation function in the hidden layers and the log-softmax activation function in the output layer for multi-class classification.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
The second ANN is a CNN:
This Python script implements a convolutional neural network (CNN) to classify images from the CIFAR-10 dataset. 
The script first imports the required packages including PyTorch, then sets up command-line arguments to save and load the trained model.
The hyperparameters for the model including learning rate, number of classes in the dataset, number of epochs, batch size, and weight decay variable are defined next.

The CIFAR-10 dataset is then loaded using torchvision and data loaders for the training and testing datasets are created.
The script then defines the LeNet5 model using PyTorch and creates an instance of the model.
The loss function and optimizer are also defined.

The script then trains the model, prints the loss at every 100th step, and uses a scheduler to adjust the learning rate after every 5 epochs.
The script then evaluates the trained model on the testing dataset, prints the test accuracy, and saves the model parameters to a file if the user chooses to save them.

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Third is A RESNET:
This is a PyTorch implementation of a Residual Neural Network (ResNet) that classifies images from the CIFAR10 dataset. 
The code reads in command line arguments to either save or load the model parameters. 
The ResNet model consists of several layers of convolutional and batch normalization layers, as well as a fully connected layer for classification. 
The code uses stochastic gradient descent (SGD) as an optimizer, cross-entropy loss as the loss function, and a learning rate scheduler.
The training is performed using a loop that iterates over a set number of epochs, and a timer is used to measure the training time. 
A function is defined for training the model and another one is defined for testing the model. 
Finally, the code saves or loads the model parameters based on the input arguments.