from model import load_classifier, load_extractor
import torch
import numpy as np
from data_loader import Data
import torch.optim as optim
import torch.nn as nn
import config
import json

conf = config.conf

class prox_Client():
    def __init__(self,  index, data, test_data, server):
        self.index = index
        self.train_data = Data(data)
        self.test_data = test_data
        self.server = server
        self.num_data = len(self.train_data)
        self.cal_pdf()

        self.extractor = load_extractor().to(conf.device)
        self.classifier = load_classifier().to(conf.device)

        self.num_l_epochs = conf.l_epoch
        self.batch_size = conf.batchsize
        self.lr = conf.learning_rate

        self.prox_cfft = 0.01
        self.test_acc = []

    def cal_pdf(self):
        self.pdf = [0.0 for i in range(conf.num_classes)]
        self.train_data.data.sort(key=lambda i: i[1])
        ends = [0]
        for i in range(1, len(self.train_data)):
            if self.train_data[i][1] != self.train_data[i - 1][1]:
                self.pdf[self.train_data[i - 1][1].item()] = (i - ends[len(ends) - 1]) / len(self.train_data)
                ends.append(i)
        self.pdf[self.train_data[self.num_data - 1][1].item()] = (self.num_data - ends[len(ends) - 1]) / len(self.train_data)


    def train(self):
        print("FedProx:")
        self.extractor.train()
        self.classifier.train()
        optimizer_e = optim.SGD(self.extractor.parameters(), lr=self.lr)
        optimizer_c = optim.SGD(self.classifier.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss().to(conf.device)
        prox_cfft = conf.prox_cfft # coefficient of FedProx

        self.loss_avg = []
        trainloader = torch.utils.data.DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)
        for l_epoch in range(self.num_l_epochs):
            avg_loss = 0
            for i, (x, y) in enumerate(trainloader):
                optimizer_e.zero_grad()
                optimizer_c.zero_grad()

                x, y = x.to(conf.device), y.to(conf.device)
                prox_term = 0
                # calculate proximal term
                for (w_l, w_g) in zip(self.extractor.parameters(), self.server.extractor.parameters()):
                    prox_term += (w_l - w_g).norm(2)
                for (w_l, w_g) in zip(self.classifier.parameters(), self.server.classifier.parameters()):
                    prox_term += (w_l - w_g).norm(2)

                output = self.classifier(self.extractor(x))
                loss = criterion(output, y) + prox_term * prox_cfft
                loss.backward()
                optimizer_c.step()
                optimizer_e.step()

                avg_loss += loss.cpu().item()

            avg_loss /= int(len(self.train_data)/self.batch_size)
            print("client: %d\t epoch: (%d/%d)\t loss: %f"
                  % (self.index, l_epoch, self.num_l_epochs, avg_loss))
            self.loss_avg.append(avg_loss)

    def finetune(self):

        print("FedProx + ft:")
        self.classifier.train()
        for key, value in self.extractor.named_parameters():
            value.requires_grad = False
        # setup optimizer
        optimizer = optim.SGD(self.classifier.parameters(), lr=self.lr)

        criterion = nn.CrossEntropyLoss().to(conf.device)
        prox_cfft = conf.prox_cfft

        self.loss_avg = []
        trainloader = torch.utils.data.DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)
        for l_epoch in range(self.num_l_epochs):
            avg_loss = 0
            for i, (x, y) in enumerate(trainloader):
                x, y = x.to(conf.device), y.to(conf.device)
                optimizer.zero_grad()

                output = self.classifier(self.extractor(x))
                loss = criterion(output, y)
                loss.backward()
                optimizer.step()

                avg_loss += loss.cpu().item()

            avg_loss /= int(len(self.train_data) / self.batch_size)
            print("client: %d\t epoch: (%d/%d)\t loss: %f"
                  % (self.index, l_epoch, self.num_l_epochs, avg_loss))
            self.loss_avg.append(avg_loss)

    def down_model(self):

        # FedProx
        for key in self.extractor.state_dict().keys():
            self.extractor.state_dict()[key].data.copy_(self.server.extractor.state_dict()[key])

        for key in self.classifier.state_dict().keys():
            self.classifier.state_dict()[key].data.copy_(self.server.classifier.state_dict()[key])

    def test(self):
        test_loader = torch.utils.data.DataLoader(self.test_data, batch_size=self.batch_size, shuffle=True)
        self.extractor.eval()
        self.classifier.eval()
        total = 0
        correct = 0
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
        with open('./FedProx/log/test_acc_%d.txt' % self.index, 'w') as fp:
            json.dump(self.test_acc, fp)

    def save_model(self):
        torch.save(self.extractor, './FedProx/model/extractor_%d.pt' % self.index)
        torch.save(self.classifier, './FedProx/model/classifier_%d.pt' % self.index)