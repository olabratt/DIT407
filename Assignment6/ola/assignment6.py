import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from torchvision import datasets
from torch.utils.data import DataLoader


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(NeuralNet, self).__init__()
        layer_sizes = [input_size] + hidden_sizes + [output_size]
        
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            if i < len(layer_sizes) - 2:  # Add ReLU and batch normalization except for the last layer
                layers.append(nn.BatchNorm1d(layer_sizes[i+1])) 
                layers.append(nn.ReLU())
            else:
                layers.append(nn.LogSoftmax(dim=1)) #layers.append(nn.Softmax(dim=1))


        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.model(x)
        return x

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = x.view(-1, 32 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)
        return x

def plot_images(dataloader, classes):

    for images, labels in train_loader:
        print("Image shape:", images.size())
        print("Label shape:", labels.size()) 

        fig = plt.figure(figsize=(10, 10))
        for i in range(4):
            plt.subplot(5, 5, i + 1)
            plt.imshow(images[i].squeeze(), cmap='gray') 
            plt.title(f'Label: {labels[i]}')
            plt.axis('off')
        plt.show()
        fig.savefig('mnist_images.png', bbox_inches='tight')
        break  



def train(model, criterion, optimizer, train_loader, test_loader, num_epochs, name):
    train_losses = []  
    test_losses = []

    for epoch in range(num_epochs):
        model.train()  
        running_loss = 0.0

        for images, labels in train_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        
        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)

        model.eval()
        correct = 0
        total = 0
        test_loss = 0.0
        with torch.no_grad():
            for images, labels in test_loader:
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
        accuracy = correct / total
        test_loss /= len(test_loader)
        test_losses.append(test_loss)

        print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {epoch_loss:.4f}, Test Loss: {test_loss:.4f}, Test Accuracy: {accuracy:.4f}")


    fig, ax = plt.subplots(figsize=(8, 6), layout='constrained')
    ax.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
    ax.plot(range(1, num_epochs + 1), test_losses, label='Test Loss')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.set_title('Training and Test Losses')
    ax.legend()
    plt.show()
    fig.savefig(name  +  ".pdf", bbox_inches='tight')


# Importing the dataset
batch_size = 32
transform = transforms.Compose([
    transforms.Resize((28, 28)), 
    transforms.ToTensor(),  
    transforms.Normalize((0.5,), (0.5,))  
])

train_dataset = datasets.MNIST(root='Assignment6/', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='Assignment6/', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

print("train dataset: ", len(train_dataset))
print("test dataset: ", len(test_dataset))


# Single hidden layer
input_size = 28 * 28 
hidden_sizes = [300]
output_size = 10  

modelSHL = NeuralNet(input_size, hidden_sizes, output_size)
learning_rate = 0.1
optimizer = optim.SGD(modelSHL.parameters(), lr=learning_rate)
num_epochs = 10
criterion = nn.CrossEntropyLoss()

train(modelSHL, criterion, optimizer, train_loader, test_loader, num_epochs, "single_hidden_layer")


# Two hidden layers
hidden_sizes = [500, 300]  
weight_decay = 0.0001
modelTHL = NeuralNet(input_size, hidden_sizes, output_size)
optimizer = optim.SGD(modelTHL.parameters(), lr=learning_rate, weight_decay=weight_decay)
num_epochs = 40

train(modelTHL, criterion, optimizer, train_loader, test_loader, num_epochs, "two_hidden_layer")


# Convolutional neural network
modelCNN = CNN() 
weight_decay = 0.0001
optimizer = optim.SGD(modelCNN.parameters(), lr=learning_rate, weight_decay=weight_decay)
num_epochs = 40

train(modelCNN, criterion, optimizer, train_loader, test_loader, num_epochs, "cnn")


