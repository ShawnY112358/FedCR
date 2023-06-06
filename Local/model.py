from torch import nn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import math
import config

conf = config.conf

class Net_cifar(nn.Module):
    def __init__(self):
        super(Net_cifar, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(16 * 5 * 5, conf.prototype_size)
        self.fc2 = nn.Linear(conf.prototype_size, 84)
        self.fc3 = nn.Linear(84, conf.num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = torch.flatten(self.pool2(x), start_dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class Net_mnist(nn.Module):
    def __init__(self):
        super(Net_mnist, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
        )
        self.fc1 = nn.Linear(1568, conf.prototype_size)
        self.fc2 = nn.Linear(conf.prototype_size, 512)
        self.fc3 = nn.Linear(512, conf.num_classes)

    def forward(self, x):
        x = self.conv(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc3(self.fc2(self.fc1(x)))
        return x

class Net_DomainNet(nn.Module):
    def __init__(self):
        super(Net_DomainNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 12, 4, stride=2, padding=1),  # [batch, 12, 16, 16]
            nn.ReLU(),
            nn.Conv2d(12, 24, 4, stride=2, padding=1),  # [batch, 24, 8, 8]
            nn.ReLU(),
            nn.Conv2d(24, 48, 4, stride=2, padding=1),  # [batch, 48, 4, 4]
            nn.ReLU(),
        )
        self.fc1 = nn.Linear(37632, 512)
        self.fc2 = nn.Linear(512, conf.num_classes)

    def forward(self, x):
        x = self.conv(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc2(self.fc1(x))
        return x

class Net_Digit5(nn.Module):
    def __init__(self):
        super(Net_Digit5, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(16 * 5 * 5, conf.prototype_size)
        self.fc2 = nn.Linear(conf.prototype_size, 84)
        self.fc3 = nn.Linear(84, conf.num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = torch.flatten(self.pool2(x), start_dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def load_model():
    if conf.dataset == 'cifar':
        model = Net_cifar()
    elif conf.dataset == 'mnist':
        model = Net_mnist()
    elif conf.dataset == 'DomainNet':
        model = Net_DomainNet()
    elif conf.dataset == 'Digit5':
        model = Net_Digit5()
    return model