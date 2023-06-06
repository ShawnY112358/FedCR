from torch import nn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import math

import config

conf = config.conf

class cifar_extractor(nn.Module):
    def __init__(self):
        super(cifar_extractor, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(p=0.25),
            nn.Flatten(),
        )
        self.fc = nn.Linear(4096, conf.prototype_size)

    def forward(self, x):
        x = self.conv(x)
        x = F.relu(self.fc(x))
        return x

class cifar_classifier(nn.Module):
    def __init__(self, input_size=conf.prototype_size):
        super(cifar_classifier, self).__init__()
        self.fc = nn.Linear(input_size, conf.num_classes)

    def forward(self, x):
        x = self.fc(x)
        return x


class mnist_extractor(nn.Module):
    def __init__(self):
        super(mnist_extractor, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten()
        )
        self.fc = nn.Linear(1024, conf.prototype_size)

    def forward(self, x):
        x = self.conv(x)
        x = F.relu(self.fc(x))
        return x

class mnist_classifier(nn.Module):
    def __init__(self, input_size=conf.prototype_size):
        super(mnist_classifier, self).__init__()
        self.fc = nn.Linear(input_size, conf.num_classes)

    def forward(self, x):
        x = self.fc(x)
        return x

class Bottleneck(nn.Module):
    """
    [1 * 1, x]
    [3 * 3, x]
    [1 * 1, x * 4]
    """

    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels * self.expansion, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        if self.downsample is not None:
            residual = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)


        out += residual
        out = self.relu(out)

        return out

class BasicBlock(nn.Module):
    """
    [3 * 3, 64]
    [3 * 3, 64]
    """

    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        if self.downsample is not None:
            residual = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, pooling_size, channels=3, head=True, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.linear = nn.Linear(256 * block.expansion, num_classes)
        self.head = head
        self.pooling_size=pooling_size


    def _make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            downsample = None
            if stride != 1 or self.in_channels != channels * block.expansion:
                downsample = nn.Sequential(
                    nn.Conv2d(self.in_channels, channels * block.expansion, stride=stride, kernel_size=1),
                    nn.BatchNorm2d(channels * block.expansion)
                )
            layers.append(block(self.in_channels, channels, stride, downsample=downsample))
            self.in_channels = channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, self.pooling_size)
        out = out.view(out.size(0), -1)
        if self.head:
            out = self.linear(out)
        return out

def ResNet8_cifar():
    return ResNet(channels=3, block=BasicBlock, num_blocks=[1, 1, 1], num_classes=conf.num_classes, head=False, pooling_size=8)

def ResNet8_mnist():
    return ResNet(channels=1, block=BasicBlock, num_blocks=[1, 1, 1], num_classes=conf.num_classes, head=False, pooling_size=7)


class mlp_mnist(nn.Module):
    def __init__(self):
        super(mlp_mnist, self).__init__()
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 200),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.ReLU(),
        )

    def forward(self, x):

        out = self.fc(x)

        return out

class mlp_cifar(nn.Module):
    def __init__(self):
        super(mlp_cifar, self).__init__()
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 32 * 32, 1024),
            nn.ReLU(),
            nn.Linear(3 * 32 * 32, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU()
        )

    def forward(self, x):
        out = self.fc(x)
        return out



def load_extractor():
    if conf.dataset == 'cifar':
        if conf.model == 'cnn':
            model = cifar_extractor()
        elif conf.model == 'resnet':
            model = ResNet8_cifar()
        elif conf.model == 'mlp':
            model = mlp_cifar()

    elif conf.dataset == 'mnist':
        if conf.model == 'cnn':
            model = mnist_extractor()
        elif conf.model == 'resnet':
            model = ResNet8_mnist()
        elif conf.model == 'mlp':
            model = mlp_mnist()

    return model

def load_classifier():
    if conf.dataset == 'cifar':
        model = cifar_classifier()
    elif conf.dataset == 'mnist':
        if conf.model == 'mlp':
            model = mnist_classifier(input_size=200)
        else:
            model = mnist_classifier()

    return model