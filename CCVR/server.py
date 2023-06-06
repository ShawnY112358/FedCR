import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
import torch.optim as optim
from model import load_classifier, load_extractor
import json
import config
from torchvision import transforms
import os
from sklearn.manifold import TSNE
from data_loader import load_dataset, Data
conf = config.conf

class ccvr_Server():
    def __init__(self):
        self.clients = []
        self.avg_acc = []
        self.classifier = load_classifier().to(conf.device)
        self.extractor = load_extractor().to(conf.device)
        self.test_set = load_dataset(is_train=False)
        self.test_loader = torch.utils.data.DataLoader(self.test_set, batch_size=conf.batchsize, shuffle=False)
        self.test_acc = []

    def aggregate(self):
        # extractor
        for key in self.extractor.state_dict().keys():
            if 'num_batches_tracked' in key:
                self.extractor.state_dict()[key].data.copy_(self.clients[0].extractor.state_dict()[key])
                continue
            temp = torch.zeros_like(self.extractor.state_dict()[key]).to(conf.device)
            N = 0
            for i in range(len(self.clients)):
                temp += self.clients[i].num_data * self.clients[i].extractor.state_dict()[key]
                N += self.clients[i].num_data
            self.extractor.state_dict()[key].data.copy_(temp / N)

        # classifier
        for key in self.classifier.state_dict().keys():
            if 'num_batches_tracked' in key:
                self.classifier.state_dict()[key].data.copy_(self.clients[0].classifier.state_dict()[key])
                continue
            temp = torch.zeros_like(self.classifier.state_dict()[key]).to(conf.device)
            N = 0
            for i in range(len(self.clients)):
                temp += self.clients[i].num_data * self.clients[i].classifier.state_dict()[key]
                N += self.clients[i].num_data
            self.classifier.state_dict()[key].data.copy_(temp / N)

    def calibration(self):
        self.mean, self.std = [], []

        for class_id in range(conf.num_classes):
            mean = torch.zeros([1, conf.prototype_size])
            sum_data = 0
            for client in self.clients:
                if client.pdf[class_id] != 0: # client没有这一类
                    mean += client.pdf[class_id] * client.mean[class_id]
                    sum_data += client.pdf[class_id]

            # 所有client都没有这一类
            if sum_data == 0:
                self.mean[class_id] = None
                self.std[class_id] = None
                continue
            else:
                self.mean.append(mean / sum_data)

            std = torch.zeros([conf.prototype_size, conf.prototype_size])
            for client in self.clients:
                if client.pdf[class_id] != 0:
                    std += (client.pdf[class_id] - 1) * client.std[class_id] / (sum_data - 1)
                    std += client.pdf[class_id] * torch.mm(client.mean[class_id].t(), client.mean[class_id]) / (sum_data - 1)
            std -= sum_data * torch.mm(self.mean[class_id].t(), self.mean[class_id]) / (sum_data - 1)
            self.std.append(std)

        proxy_data = []
        for i in range(conf.num_classes):
            if self.mean[i] != None:
                feature = np.random.multivariate_normal(self.mean[i].squeeze().numpy(), self.std[i].numpy(), (int)(conf.num_proxy_data / conf.num_classes))
                for j in range(feature.shape[0]):
                    proxy_data.append((torch.from_numpy(feature[j]).float(), torch.tensor(i).long()))
        self.proxy_data = Data(data=proxy_data)

        self.classifier = load_classifier().to(conf.device)
        data_loader = torch.utils.data.DataLoader(self.proxy_data, batch_size=conf.batchsize, shuffle=True)
        optimizer_c = optim.SGD(self.classifier.parameters(), lr=conf.learning_rate)
        criterion_c = nn.CrossEntropyLoss().to(conf.device)

        for l_epoch in range(conf.nums_ft_epoch):
            for i, data in enumerate(data_loader):
                x, y = data
                x, y = x.to(conf.device), y.to(conf.device)
                optimizer_c.zero_grad()
                loss = criterion_c(self.classifier(x), y)
                loss.backward()
                optimizer_c.step()

                print("Calibration\t epoch: (%d/%d) \t loss: %f" %(l_epoch, conf.nums_ft_epoch, loss.item()))
            self.test()


    def test(self):
        self.extractor.eval()
        self.classifier.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data in self.test_loader:
                inputs, labels = data
                inputs, labels = inputs.to(conf.device), labels.to(conf.device)
                output = self.classifier(self.extractor(inputs))
                _, predicted = torch.max(output.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print('accuracy:%d/%d = %f' % (correct, total, correct / total))
        self.test_acc.append(correct / total)
        with open('./CCVR/log/test_acc.txt', 'w') as fp:
            json.dump(self.test_acc, fp)

        self.classifier.train()
