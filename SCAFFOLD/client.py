from model import load_classifier, load_extractor
import torch
import numpy as np
from data_loader import Data
import torch.optim as optim
import torch.nn as nn
import config
import json
import copy

conf = config.conf

class scaffold_Client():
    def __init__(self,  index, data, test_data, server):
        self.index = index
        self.train_data = Data(data)
        self.test_data = Data(test_data)
        self.server = server
        self.num_data = len(self.train_data)
        self.cal_pdf()


        self.extractor = load_extractor().to(conf.device)
        self.classifier = load_classifier().to(conf.device)

        self.c_e = load_extractor().to(conf.device) # control variate
        self.c_c = load_classifier().to(conf.device)

        # init c with 0
        for key in self.c_e.state_dict().keys():
            temp = torch.zeros_like(self.c_e.state_dict()[key]).to(conf.device)
            self.c_e.state_dict()[key].data.copy_(temp)
        for key in self.c_c.state_dict().keys():
            temp = torch.zeros_like(self.c_c.state_dict()[key]).to(conf.device)
            self.c_c.state_dict()[key].data.copy_(temp)

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
        print("SCAFFOLD:")
        self.extractor.train()
        self.classifier.train()
        optimizer_e = optim.SGD(self.extractor.parameters(), lr=self.lr)
        optimizer_c = optim.SGD(self.classifier.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss().to(conf.device)

        ce_global_para = self.server.c_e.state_dict()
        ce_local_para = self.c_e.state_dict()

        cc_global_para = self.server.c_c.state_dict()
        cc_local_para = self.c_c.state_dict()

        trainloader = torch.utils.data.DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)
        batch_cnt = 0
        for l_epoch in range(self.num_l_epochs):
            avg_loss = 0
            for i, (x, y) in enumerate(trainloader):
                x, y = x.to(conf.device), y.to(conf.device)
                optimizer_e.zero_grad()
                optimizer_c.zero_grad()
                output = self.classifier(self.extractor(x))
                loss = criterion(output, y)
                loss.backward()
                optimizer_e.step()
                optimizer_c.step()

                # tune update with c_e
                net_para = self.extractor.state_dict()
                for key in net_para:
                    if 'num_batches_tracked' in key:
                        continue
                    net_para[key] = net_para[key] - self.lr * (ce_global_para[key] - ce_local_para[key])
                self.extractor.load_state_dict(net_para)
                avg_loss += loss.cpu().item()

                net_para = self.classifier.state_dict()
                for key in net_para:
                    if 'num_batches_tracked' in key:
                        continue
                    net_para[key] = net_para[key] - self.lr * (cc_global_para[key] - cc_local_para[key])
                self.classifier.load_state_dict(net_para)
                avg_loss += loss.cpu().item()

                batch_cnt += 1

            avg_loss /= int(len(self.train_data)/self.batch_size)
            print("client: %d\t epoch: (%d/%d)\t loss: %f"
                  % (self.index, l_epoch, self.num_l_epochs, avg_loss))

        # update c
        c_new_para = copy.deepcopy(self.c_e.state_dict())
        self.ce_delta_para = copy.deepcopy(self.c_e.state_dict())
        global_model_para = self.server.extractor.state_dict()
        net_para = self.extractor.state_dict()
        for key in net_para:
            c_new_para[key] = c_new_para[key] - ce_global_para[key] + (global_model_para[key] - net_para[key]) / (batch_cnt * self.lr)
            self.ce_delta_para[key] = c_new_para[key] - ce_local_para[key]
        self.c_e.load_state_dict(c_new_para)

        c_new_para = copy.deepcopy(self.c_c.state_dict())
        self.cc_delta_para = copy.deepcopy(self.c_c.state_dict())
        global_model_para = self.server.classifier.state_dict()
        net_para = self.classifier.state_dict()
        for key in net_para:
            c_new_para[key] = c_new_para[key] - cc_global_para[key] + (global_model_para[key] - net_para[key]) / (batch_cnt * self.lr)
            self.cc_delta_para[key] = c_new_para[key] - cc_local_para[key]
        self.c_c.load_state_dict(c_new_para)

    def finetune(self):
        print("SCAFFOLD with fine-tune:")
        self.classifier.train()
        for key, value in self.extractor.named_parameters():
            value.requires_grad = False
        optimizer = optim.SGD(self.classifier.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss().to(conf.device)

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

            avg_loss /= int(len(self.train_data)/self.batch_size)
            print("client: %d\t epoch: (%d/%d)\t loss: %f"
                  % (self.index, l_epoch, self.num_l_epochs, avg_loss))

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
        with open('./SCAFFOLD/log/test_acc_%d.txt' % self.index, 'w') as fp:
            json.dump(self.test_acc, fp)
