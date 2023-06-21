#This file contains the implementation using a CNN

#importing packagaes
import torch
import torch.nn as nn
import torch.nn.functional as T 
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import time
import math
import io
import argparse

parser = argparse.ArgumentParser(description='Train a neural network')
parser.add_argument('-save', action='store_true', help='Save model parameters')
parser.add_argument('-load', action='store_true', help='Load model parameters')
args = parser.parse_args()


# Define the device to be used for training the model (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Define the hyperparameters of the model
learningRate=0.015 #Learning rate of model
numClasses=10 #Number of classes in the CIFAR10 dataset
numEpochs=15 #Number of epochs
batchSize=64 #Size of batch
weightDecay=0.0012#Weight decay variable
gamma=0.15



# Defining the transformations to be applied to the images in the dataset
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# Load the CIFAR10 dataset
trainDataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testDataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# Create data loaders for the training and testing datasets
trainLoader = torch.utils.data.DataLoader(trainDataset, batchSize, shuffle=True)
testLoader = torch.utils.data.DataLoader(testDataset, batchSize, shuffle=False)

#Defining the LeNet5 model
class LeNet5(nn.Module):
    def __init__(self, num_classes):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten=nn.Flatten()
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
        #self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.pool1(T.relu(self.conv1(x)))
        x = self.pool2(T.relu(self.conv2(x)))
        x = self.flatten(x)
        x = T.relu(self.fc1(x))
        x = T.relu(self.fc2(x))
        x = self.fc3(x)
        return x

#Creating instance of model
model=LeNet5(numClasses).to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer=optim.SGD(model.parameters(), lr=learningRate, momentum=0.9, weight_decay=weightDecay)

#Scheduler
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=gamma)
# Adding a timer
startTime = time.time()
def train():
    # Train the model
    for epoch in range(numEpochs):
        model.train()
        runningLoss=0.0
        for i, (images, labels) in enumerate(trainLoader):
            # Move the images and labels to the device
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            loss.backward()
            optimizer.step()
            runningLoss+=loss.item()

            if (i+1) % 100 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, numEpochs, i+1, len(trainLoader), loss.item()))
        scheduler.step()
        
def test():
    model.eval()  # switch to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in testLoader:
            # Move the images and labels to the device
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)

            # Get the predicted class from the outputs
            _, predicted = torch.max(outputs.data, 1)

            # Update the total and correct counts
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        # Print the test accuracy
    accuracy = math.ceil(100 * correct / total)
    print('Test Accuracy of the CNN model on the {} test images: {} %'.format(len(testLoader.dataset), 100 * correct / total))
    print('Test Accuracy of the CNN model on the {} test images rounded up: {} %'.format(len(testLoader.dataset),accuracy))

if args.save:
    # save model parameters
    # saving model
    torch.save(model.state_dict(), 'CNN_model.pth')
    print('Saving model...')
    train()
    print('Done!')
    torch.save(model.state_dict(), 'CNN_cifar10.pth')
    print('Model parameters saved to CNN_cifar10.pth')

elif args.load:
    # load model parameters
    model.load_state_dict(torch.load('CNN_cifar10.pth'))
    print('Model parameters loaded from CNN_cifar10.pth')
    # Load model parameters from file
    with open('CNN_cifar10.pth', 'rb') as f:
        buffer = io.BytesIO(f.read())
    model.load_state_dict(torch.load(buffer))
    test()
    print('Model parameters loaded successfully!')
    
else:
    print("Invalid Arguments")


endTime=time.time()

duration=(endTime-startTime)/60

print(f"Elapsed time: {duration:.2f} minutes")