from model import load_classifier, load_extractor
import torch
import numpy as np
from data_loader import Data
import torch.optim as optim
import torch.nn as nn
import config
import json
import torch.nn.functional as F
from itertools import cycle
from torchvision import transforms
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
conf = config.conf

class ccvr_Client():
    def __init__(self, index, train_data, test_data, server):
        self.index = index
        self.train_data = train_data
        # self.test_data = test_data
        self.server = server
        self.num_data = len(self.train_data)

        self.extractor = load_extractor().to(conf.device)
        self.classifier = load_classifier().to(conf.device)

        self.num_l_epochs = conf.l_epoch
        self.batch_size = conf.batchsize
        self.lr = conf.learning_rate
        self.sigma = 0.1

        self.test_acc = []

        # local pdf
        self.train_data.sort(key=lambda i: i[1])
        ends_tr = [0]
        for i in range(1, len(self.train_data)):
            if self.train_data[i][1] != self.train_data[i - 1][1]:
                ends_tr.append(i)
        ends_tr.append(len(self.train_data))
        self.pdf = [0 for i in range(conf.num_classes)]
        for i in range(len(ends_tr) - 1):
            self.pdf[self.train_data[ends_tr[i]][1].item()] = ends_tr[i + 1] - ends_tr[i]

    def train(self):
        print("CCVR:")
        self.extractor.train()
        self.classifier.train()

        # train classifier and extractor together
        optimizer_e = optim.SGD(self.extractor.parameters(), lr=self.lr)
        optimizer_c = optim.SGD(self.classifier.parameters(), lr=self.lr)

        criterion_c = nn.CrossEntropyLoss().to(conf.device)

        self.avg_loss = []
        local_loader = torch.utils.data.DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)

        for l_epoch in range(self.num_l_epochs):
            if l_epoch % 100 == 0:
                self.save_model()
            for i, data in enumerate(local_loader):
                x, y = data
                x, y = x.to(conf.device), y.to(conf.device)

                optimizer_e.zero_grad()
                optimizer_c.zero_grad()
                feature = self.extractor(x)
                loss = criterion_c(self.classifier(feature), y)
                loss.backward()
                optimizer_c.step()
                optimizer_e.step()

            print("client: %d\t epoch: (%d/%d) \t loss: %f"
                  %(self.index, l_epoch, self.num_l_epochs, loss.item()))

    def update_distribution_info(self):

        def cov(m):
            x = m - torch.mean(m, dim=0)
            cov_matrix = torch.matmul(x.T, x) / (x.shape[0] - 1)
            return cov_matrix

        self.extractor.eval()
        self.feature = [None for i in range(conf.num_classes)]
        local_loader = torch.utils.data.DataLoader(self.train_data, batch_size=1, shuffle=False)
        for i, data in enumerate(local_loader):
            x, y = data
            x = x.to(conf.device)
            if self.feature[y.item()] == None:
                self.feature[y.item()] = self.extractor(x).cpu().detach()
            else:
                self.feature[y.item()] = torch.cat([self.feature[y.item()], self.extractor(x).cpu().detach()], dim=0)

        self.mean, self.std, self.pdf = [], [], []
        for c in range(conf.num_classes):
            if self.feature[c] == None:
                self.mean.append(None)
                self.std.append(None)
                self.pdf.append(0)
            else:
                self.mean.append(torch.mean(self.feature[c], dim=0).unsqueeze(0))
                self.std.append(cov(self.feature[c]))
                self.pdf.append(self.feature[c].shape[0])

    def down_model(self):
        for key in self.extractor.state_dict().keys():
            self.extractor.state_dict()[key].data.copy_(self.server.extractor.state_dict()[key])
        for key in self.classifier.state_dict().keys():
            self.classifier.state_dict()[key].data.copy_(self.server.classifier.state_dict()[key])

    def test(self):

        test_loader = torch.utils.data.DataLoader(self.test_data, batch_size=self.batch_size, shuffle=True)
        self.extractor.eval()
        self.classifier.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data in test_loader:
                inputs, labels = data
                inputs, labels = inputs.to(conf.device), labels.to(conf.device)
                output = self.classifier(self.extractor(inputs))
                _, predicted = torch.max(output.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print('accuracy:%d/%d = %f' % (correct, total, correct / total))
        self.test_acc.append(correct / total)
        with open('./CCVR/log/test_acc_%d.txt' % self.index, 'w') as fp:
            json.dump(self.test_acc, fp)

    def save_model(self):
        torch.save(self.extractor, './CCVR/model/extractor_%d.pt' % self.index)
        torch.save(self.classifier, './CCVR/model/classifier_%d.pt' % self.index)