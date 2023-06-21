#This file containts code to the MLP implementation

#Importing necessary packages
import argparse
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn.functional as T 
import time
import math
import io

parser = argparse.ArgumentParser(description='Train a neural network')
parser.add_argument('-save', action='store_true', help='Save model parameters')
parser.add_argument('-load', action='store_true', help='Load model parameters')
args = parser.parse_args()

# Defining the device to be used for training the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
numEpochs = 15  # Number of epochs to train the MLP going through the dataset
learningRate = 0.015  # Learning rate for the optimizer
weightDecay = 0.0013  # Weight decay for L2 regularization
numClasses = 10  # Number of classes in the dataset CIFAR10
hiddenNeurons1 = 512  # Number of neurons in the first hidden layer
hiddenNeurons2 = 256  # Number of neurons in the second hidden layer
#hiddenNeurons3 = 128  # Number of neurons in the third hidden layer
inputSize = 32*32*3  # Size of the input images
batchSize = 64  # Batch size
gamma=0.15
stepSize=5
#dropout=0.3

# Transformations to be applied to images in the dataset
transform = transforms.Compose(
    [transforms.ToTensor(),  # converting images to tensors
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])  # normalizing the images

# Loading CIFAR10 dataset
trainDataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testDataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# Creating data loaders for the training and testing datasets
trainLoader = torch.utils.data.DataLoader(trainDataset, batchSize, shuffle=True)
testLoader = torch.utils.data.DataLoader(testDataset, batchSize, shuffle=False)

# Defining MLP model 
class MLP(nn.Module):
    def __init__(self, inputSize,hiddenNeurons1,hiddenNeurons2, numClasses):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten() #Flattening 2D image
        self.fc1 = nn.Linear(inputSize, hiddenNeurons1)
        #self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hiddenNeurons1, hiddenNeurons2)
        self.fc3 = nn.Linear(hiddenNeurons2, numClasses)
        #self.fc_last = nn.Linear(hiddenNeurons3, numClasses)
        self.output = nn.LogSoftmax(dim=1)

    def forward(self, x):
        # Batch x of shape (B, C, W, H) B - batch, 
      x = self.flatten(x) # Batch now has shape (B, C*W*H)
      x = T.relu(self.fc1(x))  # First Hidden Layer
      x = T.relu(self.fc2(x))  # Second Hidden Layer
      x = self.fc3(x)  # Output Layer
      x = self.output(x)  # For multi-class classification
      return x  # Has shape (B, 10)

# Creating instance of model
model = MLP(inputSize, hiddenNeurons1,hiddenNeurons2,numClasses).to(device)
       
# Defining optimizer and loss function with weight decay
lossFunction = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learningRate, momentum=0.9, weight_decay=weightDecay)

# Defining the learning rate scheduler
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=stepSize, gamma=gamma)

# Adding a timer
startTime = time.time()

# Training the MLP model
def train():
    
    for epoch in range(numEpochs):
        model.train()
        totalRunningLoss=0.00      
        for i, (images, labels) in enumerate(trainLoader,0):
            # Move the images and labels to the device
            images = images.reshape(-1, 32*32*3).to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            loss = lossFunction(outputs, labels)

            # Backward and optimize
            loss.backward()
            optimizer.step()
            totalRunningLoss+=loss.item()
            if (i+1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                    .format(epoch+1, numEpochs, i+1, len(trainLoader), totalRunningLoss/len((trainLoader))))

        scheduler.step()
    return totalRunningLoss
#Testing MLP
def test():
    #First switch to evaluation mode
    model.eval()
    positive=0
    total=0
    with torch.no_grad():
        for images, labels in testLoader:
            #Moving images and labels to device
            images=images.to(device)
            labels=labels.to(device)
            
            #Forward pass
            outputs=model(images)
            
            #Getting predicted class from the outputs
            _, predicted = torch.max(outputs.data, 1)
            
            #Updating counts
            total+=(labels.size(0))
            positive+=(predicted==labels).sum().item()

    accuracy=math.ceil(100* (positive/total))
    #Checking test accuracy
    print('Test accuracy of the MLP is',100* (positive/total),'%')
    print('Test accuracy of the MLP rounded up is',accuracy,'%')
    return accuracy

if args.save:
    # save model parameters
    # saving model
    torch.save(model.state_dict(), 'MLP_model.pth')
    print('Saving model...')
    train()
    print('Done!')
    torch.save(model.state_dict(), 'MLP_cifar10.pth')
    print('Model parameters saved to MLP_cifar10.pth')

elif args.load:
    # load model parameters
    model.load_state_dict(torch.load('MLP_cifar10.pth'))
    print('Model parameters loaded from MLP_cifar10.pth')
    # Load model parameters from file
    with open('MLP_cifar10.pth', 'rb') as f:
        buffer = io.BytesIO(f.read())
    model.load_state_dict(torch.load(buffer))
    test()
    print('Model parameters loaded successfully!')
    
else:
    print("Invalid Arguments")

    
endTime=time.time()

duration=(endTime-startTime)/60

print(f"Elapsed time: {duration:.2f} minutes")
