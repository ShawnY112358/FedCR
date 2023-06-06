from model import load_extractor, load_classifier
import torch
import numpy as np
from data_loader import Data
import torch.optim as optim
import torch.nn as nn
import config
import json
from utils import FocalLoss
conf = config.conf

class focal_Client():
    def __init__(self, index, data, test_data, server):
        self.index = index
        self.train_data = Data(data)
        # self.test_data = Data(test_data)

        self.num_data = len(self.train_data)
        self.server = server
        self.cal_pdf()


        self.extractor = load_extractor().to(conf.device)
        self.classifier = load_classifier().to(conf.device)
        # self.model_avg = Net_cifar().to(conf.device) # FedAvg
        # self.model_ft = Net_cifar().to(conf.device) # FedMR
        # self.model_wa = Net_cifar().to(conf.device) # FedMR on the whole model
        # self.model_sr = Net_cifar().to(conf.device) # sample re-weighting

        self.num_l_epochs = conf.l_epoch
        self.batch_size = conf.batchsize
        self.lr = conf.learning_rate
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
        print("FedFocal:")
        self.extractor.train()
        self.classifier.train()
        optimizer_e = optim.SGD(self.extractor.parameters(), lr=self.lr)
        optimizer_c = optim.SGD(self.classifier.parameters(), lr=self.lr)
        criterion = FocalLoss(class_num=conf.num_classes).to(conf.device).to(conf.device)

        self.loss_avg = []
        trainloader = torch.utils.data.DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)
        for l_epoch in range(self.num_l_epochs):
            avg_loss = 0
            for i, (x, y) in enumerate(trainloader):
                optimizer_e.zero_grad()
                optimizer_c.zero_grad()

                x, y = x.to(conf.device), y.to(conf.device)
                output = self.classifier(self.extractor(x))
                loss = criterion(output, y)
                loss.backward()
                optimizer_e.step()
                optimizer_c.step()

                avg_loss += loss.cpu().item()

            avg_loss /= int(len(self.train_data)/self.batch_size)
            print("client: %d\t epoch: (%d/%d)\t loss: %f"
                  % (self.index, l_epoch, self.num_l_epochs, avg_loss))
            self.loss_avg.append(avg_loss)

    def down_model(self):

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
        with open('./FedFocal/log/test_acc_%d.txt' % self.index, 'w') as fp:
            json.dump(self.test_acc, fp)

    def save_model(self):
        torch.save(self.extractor, './FedFocal/model/extractor_%d.pt' % self.index)
        torch.save(self.classifier, './FedFocal/model/classifier_%d.pt' % self.index)