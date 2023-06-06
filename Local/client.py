from Local.model import load_model
import torch
import numpy as np
from data_loader import Data
import torch.optim as optim
import torch.nn as nn
import config
import json
import torchvision.transforms as transforms
from PIL import Image

conf = config.conf

class local_Client():
    def __init__(self, index, train_data, test_data):
        self.index = index
        self.train_data = train_data
        self.test_data = test_data
        self.num_data = len(self.train_data)

        self.model = load_model().to(conf.device)

        self.batch_size = conf.batch_size
        self.lr = conf.learning_rate

        self.test_acc = []

    def train(self):
        print("Local:")
        self.model.train()
        optimizer = optim.SGD(self.model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss().to(conf.device)

        self.loss_avg = []
        trainloader = torch.utils.data.DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)
        for l_epoch in range(conf.nums_l_epoch):
            avg_loss = 0
            for i, (x, y) in enumerate(trainloader):
                x, y = x.to(conf.device), y.to(conf.device)
                optimizer.zero_grad()
                output = self.model(x)

                loss = criterion(output, y)
                loss.backward()
                optimizer.step()

                avg_loss += loss.cpu().item()

            avg_loss /= int(len(self.train_data) / self.batch_size)
            print("client: %d\t epoch: (%d/%d)\t loss: %f"
                  % (self.index, l_epoch, conf.nums_l_epoch, avg_loss))
            self.loss_avg.append(avg_loss)
            if (l_epoch + 1) % (conf.nums_l_epoch * 10) == 0:
                torch.save(self.model, './Local/model/model_%d.pt' % self.index)

    def test(self):
        test_loader = torch.utils.data.DataLoader(self.test_data, batch_size=self.batch_size, shuffle=True)
        self.model.eval()
        total = 0
        correct = 0
        with torch.no_grad():
            for data in test_loader:
                inputs, labels = data
                inputs, labels = inputs.to(conf.device), labels.to(conf.device)
                output = self.model(inputs)
                _, predicted = torch.max(output.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print('accuracy:%d/%d = %f' % (correct, total, correct / total))

        self.test_acc.append(correct / total)
        with open('./Local/log/test_acc_%d.txt' % self.index, 'w') as fp:
            json.dump(self.test_acc, fp)

        self.model.train()