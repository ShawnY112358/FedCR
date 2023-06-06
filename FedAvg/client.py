from model import load_extractor, load_classifier
import torch
import numpy as np
from data_loader import Data, load_dataset
import torch.optim as optim
import torch.nn as nn
import config
import json
conf = config.conf

class avg_Client():
    def __init__(self, index, data, test_data, server):
        self.index = index
        self.train_data = Data(data)
        self.train_data.data.sort(key=lambda i: i[1])

        self.num_data = len(self.train_data)
        self.server = server
        self.cal_pdf()

        self.extractor = load_extractor().to(conf.device)
        self.classifier = load_classifier().to(conf.device)

        self.num_l_epochs = conf.l_epoch
        self.batch_size = conf.batchsize
        self.lr = conf.learning_rate
        self.test_acc = []
        self.grad = []
        self.sigma = 0.1

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
        print("FedAvg:")
        self.extractor.train()
        self.classifier.train()
        optimizer_e = optim.SGD(self.extractor.parameters(), lr=self.lr)
        optimizer_c = optim.SGD(self.classifier.parameters(), lr=self.lr)
        criterion_e = nn.MSELoss().to(conf.device)
        criterion = nn.CrossEntropyLoss().to(conf.device)

        # criterion = nn.CrossEntropyLoss(weight=self.server.sample_weight).to(conf.device)   # sample re-weighting

        self.loss_avg = []
        trainloader = torch.utils.data.DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)
        for l_epoch in range(self.num_l_epochs):
            avg_loss, avg_loss_e = 0, 0
            for i, (x, y) in enumerate(trainloader):
                optimizer_e.zero_grad()
                optimizer_c.zero_grad()
                prototype = torch.tensor([self.prototype[i].detach().numpy() for i in y]).to(conf.device)

                x, y = x.to(conf.device), y.to(conf.device)
                x = self.extractor(x)
                loss_e = criterion_e(x, prototype)
                output = self.classifier(x)
                loss_c = criterion(output, y)
                loss = loss_c + self.sigma * loss_e
                loss.backward()
                optimizer_e.step()
                optimizer_c.step()

                avg_loss += loss_c.cpu().item()
                avg_loss_e += loss_e.cpu().item()

            avg_loss /= int(len(self.train_data)/self.batch_size)
            print("client: %d\t epoch: (%d/%d)\t loss_e: %f \t loss_c: %f"
                  % (self.index, l_epoch, self.num_l_epochs, avg_loss_e, avg_loss))
            self.loss_avg.append(avg_loss)

        feature = []
        self.class_count = [0 for i in range(conf.num_classes)]
        x = torch.tensor([self.train_data[i][0].numpy() for i in range(len(self.train_data))]).to(conf.device)
        y = torch.tensor([self.train_data[i][1] for i in range(len(self.train_data))]).to(conf.device)
        with torch.no_grad():
            output = self.extractor(x).cpu()
            for i in range(conf.num_classes):
                f = torch.zeros(conf.prototype_size)
                count = 0
                for j in range(len(y)):
                    if y[j].cpu().item() == i:
                        f += output[j]
                        count += 1
                self.class_count[i] = count
                if count != 0:
                    feature.append(f / count)
                else:
                    feature.append(None)
        self.feature = feature

    def finetune(self):
        print("FedAvg with fine-tune:")
        self.classifier.train()

        # fix feature extraction parameters
        for key, value in self.extractor.named_parameters():
            value.requires_grad = False
        # setup optimizer

        optimizer = optim.SGD(self.classifier.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss().to(conf.device)
        # criterion = nn.CrossEntropyLoss(weight=self.server.sample_weight).to(conf.device)

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

        # grad record
        grad = 0
        for (w_l, w_g) in zip(self.classifier.parameters(), self.server.classifier.parameters()):
            grad += (w_l - w_g).norm(2)
        self.grad.append(grad.item())
        with open('./FedAvg/log/grad_%d.txt' % self.index, 'w') as fp:
            json.dump(self.grad, fp)

    def down_model(self):

        self.prototype = self.server.prototype

        for key in self.extractor.state_dict().keys():
            self.extractor.state_dict()[key].data.copy_(self.server.extractor.state_dict()[key])

        for key in self.classifier.state_dict().keys():
            self.classifier.state_dict()[key].data.copy_(self.server.classifier.state_dict()[key])


    def test(self):
        self.extractor.eval()
        self.classifier.eval()

        # test_loader = torch.utils.data.DataLoader(self.test_data, batch_size=self.batch_size, shuffle=True)
        # total = 0
        # correct = 0
        # with torch.no_grad():
        #     for data in test_loader:
        #         inputs, labels = data
        #         inputs, labels = inputs.to(conf.device), labels.to(conf.device)
        #         output = self.classifier(self.extractor(inputs))
        #         _, predicted = torch.max(output.data, 1)
        #         total += labels.size(0)
        #         correct += (predicted == labels).sum().item()
        # print('accuracy:%d/%d = %f' % (correct, total, correct / total))
        # self.test_acc.append(correct / total)
        # with open('./FedAvg/log/test_acc_%d.txt' % self.index, 'w') as fp:
        #     json.dump(self.test_acc, fp)

        # test accuracy of each class

        test_loader = torch.utils.data.DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False)
        total, correct = [0 for i in range(conf.num_classes)], [0 for i in range(conf.num_classes)]
        # ends = [0]
        # for i in range(1, len(self.test_data)):
        #     if self.test_data[i][1] != self.test_data[i - 1][1]:
        #         total[self.test_data[i - 1][1].item()] = i - ends[len(ends) - 1]
        #         ends.append(i)
        # total[self.test_data[len(self.test_data) - 1][1].item()] = len(self.test_data) - ends[len(ends) - 1]

        with torch.no_grad():
            for data in test_loader:
                inputs, labels = data
                inputs, labels = inputs.to(conf.device), labels.to(conf.device)
                output = self.classifier(self.extractor(inputs))
                _, predicted = torch.max(output.data, 1)
                for p, l in zip(predicted, labels):
                    total[l.item()] += 1
                    correct[l.item()] += int((p == l).item())

        self.test_acc.append([0 if total[i] == 0 else correct[i] / total[i] for i in range(conf.num_classes)])
        print('Accuracy:' + str([0 if total[i] == 0 else correct[i] / total[i] for i in range(conf.num_classes)]))
        with open('./FedAvg/log/test_acc_%d.txt' % self.index, 'w') as fp:
            json.dump(self.test_acc, fp)

    def save_model(self):
        torch.save(self.extractor, './FedAvg/model/extractor_%d.pt' % self.index)
        torch.save(self.classifier, './FedAvg/model/classifier_%d.pt' % self.index)