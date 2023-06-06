import torch
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from torch import nn
from model import load_extractor, load_classifier
import json
from data_loader import load_dataset, Data
import config
import copy

conf = config.conf

class creff_Server():
    def __init__(self):
        self.clients = []
        self.y_pdf = torch.from_numpy(np.array([1 / conf.num_classes for i in range(conf.num_classes)]))
        self.weight_cfft = torch.ones([conf.num_classes, 1]).to(conf.device)
        self.num_of_feature = 100
        self.extractor = load_extractor().to(conf.device)
        self.classifier_ub = load_classifier().to(conf.device)
        self.classifier_b = load_classifier().to(conf.device)

        self.test_set = load_dataset(name=conf.dataset, is_train=False)
        self.test_loader = torch.utils.data.DataLoader(self.test_set, batch_size=conf.batchsize, shuffle=False)

        self.fed_feature = torch.randn(size=(conf.num_classes * self.num_of_feature, conf.prototype_size), dtype=torch.float,
                                       requires_grad=True, device=conf.device)
        self.fed_label = torch.tensor([np.ones(self.num_of_feature) * i for i in range(conf.num_classes)], dtype=torch.long,
                                      requires_grad=False, device=conf.device).view(-1)
        self.feature_lr = 0.1
        self.feature_optimizer = optim.SGD([self.fed_feature,], lr=self.feature_lr)
        self.test_acc = []

    def aggregate(self, group):

        for key in self.extractor.state_dict().keys():
            if 'num_batches_tracked' in key:
                self.extractor.state_dict()[key].data.copy_(group[0].extractor.state_dict()[key])
                continue
            temp = torch.zeros_like(self.extractor.state_dict()[key]).to(conf.device)
            N = 0
            for i in range(len(group)):
                temp += group[i].num_data * group[i].extractor.state_dict()[key]
                N += group[i].num_data
            self.extractor.state_dict()[key].data.copy_(temp / N)

        for key in self.classifier_b.state_dict().keys():
            if 'num_batches_tracked' in key:
                self.classifier_b.state_dict()[key].data.copy_(group[0].classifier_b.state_dict()[key])
                continue
            temp = torch.zeros_like(self.classifier_b.state_dict()[key]).to(conf.device)
            N = 0
            for i in range(len(group)):
                temp += group[i].num_data * group[i].classifier_b.state_dict()[key]
                N += group[i].num_data
            self.classifier_b.state_dict()[key].data.copy_(temp / N)

    def update_feature(self, group):
        # aggregates feature gradients
        grad_list = [[] for i in range(conf.num_classes)]
        for client in group:
            for c in range(conf.num_classes):
                if client.gradients[c] == []:
                    continue
                grad_list[c].append(client.gradients[c])

        avg_grad = []
        for i, g in enumerate(grad_list):
            if g == []:
                continue
            t = [sum([g[j][k] for j in range(len(g))]) / len(g) for k in range(len(g[0]))]
            avg_grad.append((i, t))

        match_epochs = 100
        criterion = nn.CrossEntropyLoss().to(conf.device)
        for epoch in range(match_epochs):
            loss_feature = torch.tensor(0.0).to(conf.device)
            for g in avg_grad:
                label, data = g
                fed_feature = self.fed_feature[label * self.num_of_feature:(label + 1) * self.num_of_feature].reshape((self.num_of_feature, conf.prototype_size))
                fed_label = torch.ones((self.num_of_feature,), device=conf.device, dtype=torch.long) * label
                output = self.classifier_ub(fed_feature)
                loss = criterion(output, fed_label)
                fed_grad = torch.autograd.grad(loss, self.classifier_ub.parameters(), create_graph=True)
                loss_feature += self.match_loss(fed_grad, data)
            print('loss feature' + str(loss_feature))

            self.feature_optimizer.zero_grad()
            loss_feature.backward()
            self.feature_optimizer.step()

    def match_loss(self, src, tgt):
        gw_real_vec = []
        gw_syn_vec = []
        for ig in range(len(tgt)):
            gw_real_vec.append(tgt[ig].reshape((-1)))
            gw_syn_vec.append(src[ig].reshape((-1)))
        gw_real_vec = torch.cat(gw_real_vec, dim=0)
        gw_syn_vec = torch.cat(gw_syn_vec, dim=0)
        dis = 1 - torch.sum(gw_real_vec * gw_syn_vec, dim=-1) / (
                    torch.norm(gw_real_vec, dim=-1) * torch.norm(gw_syn_vec, dim=-1) + 0.000001)
        return dis

    def retrain(self):
        self.classifier_ub = load_classifier().to(conf.device)
        self.classifier_ub.train()
        feature = copy.deepcopy(self.fed_feature.detach())
        label = copy.deepcopy(self.fed_label.detach())
        data = Data([(feature[i], label[i]) for i in range(feature.shape[0])])
        data_loader = torch.utils.data.DataLoader(data, batch_size=conf.batchsize, shuffle=True)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.classifier_ub.parameters(), lr=conf.learning_rate)
        for epoch in range(300):
            for data_batch in data_loader:
                images, labels = data_batch
                images, labels = images.to(conf.device), labels.to(conf.device)
                outputs = self.classifier_ub(images)
                loss_net = criterion(outputs, labels)
                optimizer.zero_grad()
                loss_net.backward()
                optimizer.step()

    def test(self):
        self.extractor.eval()
        self.classifier_ub.eval()
        self.classifier_b.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data in self.test_loader:
                inputs, labels = data
                inputs, labels = inputs.to(conf.device), labels.to(conf.device)
                output = self.classifier_ub(self.extractor(inputs))
                _, predicted = torch.max(output.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print('accuracy:%d/%d = %f' % (correct, total, correct / total))
        self.test_acc.append(correct / total)
        with open('./CReFF/log/test_acc.txt', 'w') as fp:
            json.dump(self.test_acc, fp)

        correct = 0
        total = 0
        with torch.no_grad():
            for data in self.test_loader:
                inputs, labels = data
                inputs, labels = inputs.to(conf.device), labels.to(conf.device)
                output = self.classifier_b(self.extractor(inputs))
                _, predicted = torch.max(output.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print('accuracy:%d/%d = %f' % (correct, total, correct / total))