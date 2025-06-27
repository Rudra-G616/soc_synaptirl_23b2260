# Implementation of a SNN to clasify digits in MNIST dataset

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import DataLoader, Subset

class SpikingNeuron(nn.Module):
    def __init__(self, threshold=1.0, leak=0.5):
        super(SpikingNeuron, self).__init__()
        self.threshold = threshold
        self.leak = leak
        self.mem_potential = 0
        
    def forward(self, x):
        self.mem_potential = self.leak * self.mem_potential + x
        spike = (self.mem_potential >= self.threshold).float()
        self.mem_potential = self.mem_potential * (1 - spike)
        return spike

class SNN(nn.Module):
    def __init__(self):
        super(SNN, self).__init__()
        self.fc1 = nn.Linear(28*28, 256)
        self.sn1 = SpikingNeuron()
        self.fc2 = nn.Linear(256, 128)
        self.sn2 = SpikingNeuron()
        self.fc3 = nn.Linear(128, 1)
        
    def forward(self, x, time_steps=10):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        
        outputs = torch.zeros(time_steps, batch_size, 1).to(x.device)
        
        for t in range(time_steps):
            x1 = self.fc1(x)
            s1 = self.sn1(x1)
            x2 = self.fc2(s1)
            s2 = self.sn2(x2)
            outputs[t] = self.fc3(s2)
            
        return torch.sigmoid(outputs.mean(0))

def load_binary_mnist():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    mnist_train = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    mnist_test = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    train_indices = [(mnist_train.targets == 0) | (mnist_train.targets == 1)]
    test_indices = [(mnist_test.targets == 0) | (mnist_test.targets == 1)]
    
    train_binary = Subset(mnist_train, torch.where(train_indices[0])[0])
    test_binary = Subset(mnist_test, torch.where(test_indices[0])[0])
    
    train_loader = DataLoader(train_binary, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_binary, batch_size=64, shuffle=False)
    
    return train_loader, test_loader

def train_snn():
    train_loader, test_loader = load_binary_mnist()
    
    model = SNN()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    epochs = 5
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device).float().unsqueeze(1)
            targets = (targets == 1).float()
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            predicted = (outputs > 0.5).float()
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
        
        accuracy = 100.0 * correct / total
        print(f"Epoch {epoch+1}/{epochs}, Loss: {train_loss/len(train_loader):.4f}, Accuracy: {accuracy:.2f}%")
    
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device).float().unsqueeze(1)
            targets = (targets == 1).float()
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            test_loss += loss.item()
            predicted = (outputs > 0.5).float()
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    
    accuracy = 100.0 * correct / total
    print(f"Test Loss: {test_loss/len(test_loader):.4f}, Test Accuracy: {accuracy:.2f}%")
    
    return model

if __name__ == "__main__":
    model = train_snn()