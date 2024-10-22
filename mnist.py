import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import tkinter as tk
from PIL import Image, ImageDraw

# Define the neural network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output
    
    def save(self, path):
        torch.save(self.state_dict(), path)
    
# Load the MNIST dataset
def load_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_loader = DataLoader(
        datasets.MNIST('data', train=True, download=True, transform=transform),
        batch_size=64, shuffle=True
    )
    test_loader = DataLoader(
        datasets.MNIST('data', train=False, transform=transform),
        batch_size=1000, shuffle=True
    )
    return train_loader, test_loader

# Train the model
def train(model, train_loader, optimizer):
    model.train()
    for data, target in train_loader:
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

# Test the model
def test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    return test_loss, accuracy

# Main function
def main():
    train_loader, test_loader = load_data()
    model = Net()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(1, 11):
        train(model, train_loader, optimizer)
        test_loss, accuracy = test(model, test_loader)
        print(f'Epoch {epoch}: Test loss: {test_loss}, Accuracy: {accuracy}')
    model.save('model.pth')

if __name__ == '__main__':
    main()