import torch
import torch.nn as nn
import torch.nn.functional as T
import torch.optim as optim
from torchvision import datasets, transforms
import time
import math
import io
import argparse

parser = argparse.ArgumentParser(description='Train a neural network')
parser.add_argument('-save', action='store_true', help='Save model parameters')
parser.add_argument('-load', action='store_true', help='Load model parameters')
args = parser.parse_args()


# Defining the device to be used for training the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batchSize=64
numEpochs=15

# Defining the transformations to be applied to the images in the dataset
transform_train = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load the CIFAR10 dataset
trainDataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testDataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# Create data loaders for the training and testing datasets
trainLoader = torch.utils.data.DataLoader(trainDataset, batchSize, shuffle=True)
testLoader = torch.utils.data.DataLoader(testDataset, batchSize, shuffle=False)

class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.residual = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64)
        )
        self.fc = nn.Linear(64 * 32 * 32, 10)
        self.flatten = nn.Flatten()

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        residual = out
        out = self.residual(out)
        out += residual
        out = self.relu(out)
        out = self.flatten(out)
        out = self.fc(out)
        return out

    
# Initialize the model and optimizer
model = ResNet().to(device)

# Defining optimizer and loss function with weight decay

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.015, momentum=0.9, weight_decay=0.0012)

# Defining the learning rate scheduler
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.15)

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
    return runningLoss

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
    print('Test Accuracy of the RESNET model on the {} test images: {} %'.format(len(testLoader.dataset), 100 * correct / total))
    print('Test Accuracy of the RESNET model on the {} test images rounded up: {} %'.format(len(testLoader.dataset),accuracy))
    return accuracy

if args.save:
    # save model parameters
    # saving model
    torch.save(model.state_dict(), 'RESNET_model.pth')
    print('Saving model...')
    train()
    print('Done!')
    torch.save(model.state_dict(), 'RESNET_cifar10.pth')
    print('Model parameters saved to RESNET_cifar10.pth')

elif args.load:
    # load model parameters
    model.load_state_dict(torch.load('RESNET_cifar10.pth'))
    print('Model parameters loaded from RESNET_cifar10.pth')
    # Load model parameters from file
    with open('RESNET_cifar10.pth', 'rb') as f:
        buffer = io.BytesIO(f.read())
    model.load_state_dict(torch.load(buffer))
    test()
    print('Model parameters loaded successfully!')
    
else:
    print("Invalid Arguments")


endTime=time.time()

duration=(endTime-startTime)/60

print(f"Elapsed time: {duration:.2f} minutes")