import random
from model import load_extractor, load_classifier
import torch
import numpy as np
from data_loader import Data
import torch.optim as optim
import torch.nn as nn
import config
import json
conf = config.conf

class creff_Client():
    def __init__(self, index, data, y_pdf, server):
        self.index = index
        self.train_data = Data(data)
        self.train_data.data.sort(key=lambda i: i[1])
        self.y_pdf = y_pdf
        self.server = server
        self.num_data = len(self.train_data)

        self.extractor = load_extractor().to(conf.device)
        self.classifier_b = load_classifier().to(conf.device)
        self.classifier_ub = load_classifier().to(conf.device)

        self.num_l_epochs = conf.l_epoch
        self.batch_size = conf.batchsize
        self.lr = conf.learning_rate
        self.test_acc = []


    def train(self):
        print("CReFF:")
        self.extractor.train()
        self.classifier_b.train()
        optimizer_e = optim.SGD(self.extractor.parameters(), lr=self.lr)
        optimizer_c = optim.SGD(self.classifier_b.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss().to(conf.device)

        self.loss_avg = []
        trainloader = torch.utils.data.DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)
        for l_epoch in range(self.num_l_epochs):
            avg_loss = 0
            for i, (x, y) in enumerate(trainloader):
                optimizer_e.zero_grad()
                optimizer_c.zero_grad()

                x, y = x.to(conf.device), y.to(conf.device)
                output = self.classifier_b(self.extractor(x))
                loss = criterion(output, y)
                loss.backward()
                optimizer_e.step()
                optimizer_c.step()

                avg_loss += loss.cpu().item()

            avg_loss /= int(len(self.train_data)/self.batch_size)
            print("client: %d\t epoch: (%d/%d)\t loss: %f"
                  % (self.index, l_epoch, self.num_l_epochs, avg_loss))
            self.loss_avg.append(avg_loss)

    def compute_gradients(self):
        def get_data_batches(batch_size=self.batch_size):
            ends = [0]
            for i in range(1, len(self.train_data)):
                if self.train_data[i][1] != self.train_data[i - 1][1]:
                    ends.append(i)
            ends.append(len(self.train_data))
            batches = []
            for i in range(len(ends) - 1):
                t = batch_size
                b = []
                while t != 0:
                    b.extend(random.sample(self.train_data.data[ends[i]: ends[i + 1]], min(t, ends[i + 1] - ends[i])))
                    t -= min(t, ends[i + 1] - ends[i])
                batches.append(b)
            return batches

        print('compute gradients:')
        self.extractor.train()
        self.classifier_ub.train()
        criterion = nn.CrossEntropyLoss()
        gradients = [[] for i in range(conf.num_classes)]

        # choose to repeat 10 times
        for num_compute in range(10):
            data_batches = get_data_batches()
            for batch in data_batches:
                data_loader = torch.utils.data.DataLoader(batch, batch_size=self.batch_size, shuffle=True)
                for i, (x, y) in enumerate(data_loader):
                    x, y = x.to(conf.device), y.to(conf.device)
                    output = self.classifier_ub(self.extractor(x))
                    loss = criterion(output, y)
                    gradients_c = torch.autograd.grad(loss, self.classifier_ub.parameters())
                    gradients[y[0].item()].append(list(gradients_c))

        self.gradients = [[] for c in range(conf.num_classes)]
        for c in range(conf.num_classes):
            if gradients[c] == []:
                continue
            gw_real_temp = []
            gradient_all = gradients[c]
            weight = 1.0 / len(gradient_all)
            for name_param in range(len(gradient_all[0])):
                list_values_param = []
                for client_one in gradient_all:
                    list_values_param.append(client_one[name_param] * weight)
                value_global_param = sum(list_values_param)
                gw_real_temp.append(value_global_param)
            # the real feature gradients of all classes
            self.gradients[c] = gw_real_temp



    def down_model(self):

        for key in self.extractor.state_dict().keys():
            self.extractor.state_dict()[key].data.copy_(self.server.extractor.state_dict()[key])

        for key in self.classifier_ub.state_dict().keys():
            self.classifier_ub.state_dict()[key].data.copy_(self.server.classifier_ub.state_dict()[key])

        for key in self.classifier_b.state_dict().keys():
            self.classifier_b.state_dict()[key].data.copy_(self.server.classifier_b.state_dict()[key])

    def save_model(self):
        torch.save(self.extractor, './FedAvg/model/extractor_%d.pt' % self.index)
        torch.save(self.classifier_b, './FedAvg/model/classifier_b_%d.pt' % self.index)