import torch
import numpy as np
import torch.nn.functional as F
from torch.nn import init
import torch.optim as optim
from model import load_extractor, load_classifier
import json
from data_loader import load_dataset, Data
import config
import math
import copy

conf = config.conf

class scaffold_Server():
    def __init__(self):
        self.clients = []
        self.y_pdf = torch.from_numpy(np.array([1 / conf.num_classes for i in range(conf.num_classes)]))
        self.weight_cfft = torch.ones([conf.num_classes, 1]).to(conf.device)

        self.extractor = load_extractor().to(conf.device)
        self.classifier = load_classifier().to(conf.device)
        self.c_e = load_extractor().to(conf.device)
        self.c_c = load_classifier().to(conf.device)

        # init c
        for key in self.c_e.state_dict().keys():
            temp = torch.zeros_like(self.c_e.state_dict()[key]).to(conf.device)
            self.c_e.state_dict()[key].data.copy_(temp)
        for key in self.c_c.state_dict().keys():
            temp = torch.zeros_like(self.c_c.state_dict()[key]).to(conf.device)
            self.c_c.state_dict()[key].data.copy_(temp)

        self.test_set = load_dataset(name=conf.dataset, is_train=False)
        self.test_loader = torch.utils.data.DataLoader(self.test_set, batch_size=conf.batchsize, shuffle=False)

        self.test_acc = []

    def init_classifier(self):
        self.classifier = load_classifier().to(conf.device)

    def init_weight_cfft(self):
        lr = 0.1
        num_steps = 100000
        # lambda_ = 0.1
        sigma = 1e-4

        P, N = [], []
        for client in self.clients:
            P.append(client.pdf)
            N.append([client.num_data])
        P = torch.from_numpy(np.array(P))
        N = torch.from_numpy(np.array(N)).double()

        A = torch.from_numpy(np.array([[0.5] for i in range(len(self.clients))])).double()
        A.requires_grad = True
        # A = torch.rand([len(self.clients), 1], requires_grad=True, dtype=torch.double)
        optimizer = optim.Adam([A], lr=lr)
        target_pdf = self.y_pdf
        print("composed_pdf")
        print(torch.mm((A * N).T, P) / torch.mm(A.T, N))
        with open('./SCAFFOLD/log/composed_pdf.txt', 'w') as fp:
            json.dump((torch.mm((A * N).T, P) / torch.mm(A.T, N)).tolist(), fp=fp)
        for step in range(num_steps):
            if step % 10000 == 0:
                print("step:" + str(step))
            optimizer.zero_grad()
            composed_pdf = torch.mm((A * N).T, P) / torch.mm(A.T, N)
            criterion = torch.nn.KLDivLoss()
            kl = criterion(composed_pdf.log(), target_pdf)
            # mse = F.mse_loss(composed_pdf, target_pdf)
            punish_term = -sigma * (torch.log(1 - torch.max(A)) + torch.log(torch.min(A)))

            obj = kl + punish_term
            obj.backward()
            optimizer.step()
            A.data.clamp_(0.01, 0.99)
        print("after optimization")
        print(torch.mm((A * N).T, P) / torch.mm(A.T, N))
        print(A)

        with open('./SCAFFOLD/log/opt_pdf.txt', 'w') as fp:
            json.dump((torch.mm((A * N).T, P) / torch.mm(A.T, N)).tolist(), fp=fp)

        with open('./SCAFFOLD/log/a.txt', 'w') as fp:
            json.dump(A.tolist(), fp=fp)
        self.weight_cfft = A.to(conf.device)

    def aggregate(self, group, finetune):
        #SCAFFOLD
        # global_para = self.model_scaff.state_dict()
        # N = 0
        # for i in range(len(self.clients)):
        #     N += self.clients[i].num_data
        # for i in range(len(self.clients)):
        #     net_para = self.clients[i].model_scaff.state_dict()
        #     if i == 0:
        #         for key in net_para:
        #             global_para[key] = net_para[key] * self.clients[i].num_data / N
        #     else:
        #         for key in net_para:
        #             global_para[key] += net_para[key] * self.clients[i].num_data / N
        # self.model_scaff.load_state_dict(global_para)
        if finetune:
            for key in self.classifier.state_dict().keys():
                if 'num_batches_tracked' in key:
                    self.classifier.state_dict()[key].data.copy_(self.clients[0].classifier.state_dict()[key])
                    continue
                temp = torch.zeros_like(self.classifier.state_dict()[key]).to(conf.device)
                N = 0
                for i in range(len(self.clients)):
                    temp += self.weight_cfft[i].item() * self.clients[i].num_data * self.clients[i].classifier.state_dict()[key]
                    N += self.weight_cfft[i].item() * self.clients[i].num_data
                self.classifier.state_dict()[key].data.copy_(temp / N)
        else:
            for key in self.extractor.state_dict().keys():
                if 'num_batches_tracked' in key:
                    self.extractor.state_dict()[key].data.copy_(group[0].extractor.state_dict()[key])
                    continue
                temp = torch.zeros_like(self.extractor.state_dict()[key]).to(conf.device)
                for i in range(len(group)):
                    temp += group[i].extractor.state_dict()[key]
                self.extractor.state_dict()[key].data.copy_(temp / len(group))
            for key in self.classifier.state_dict().keys():
                if 'num_batches_tracked' in key:
                    self.classifier.state_dict()[key].data.copy_(group[0].classifier.state_dict()[key])
                    continue
                temp = torch.zeros_like(self.classifier.state_dict()[key]).to(conf.device)
                for i in range(len(group)):
                    temp += group[i].classifier.state_dict()[key]
                self.classifier.state_dict()[key].data.copy_(temp / len(group))

            # update c
            c_global_para = copy.deepcopy(self.c_e.state_dict())
            total_delta = copy.deepcopy(self.c_e.state_dict())
            for key in total_delta:
                total_delta[key] = 0.0
            for key in self.c_e.state_dict():
                for client in group:
                    total_delta[key] += client.ce_delta_para[key]
                total_delta[key] /= len(group)
                if c_global_para[key].type() == 'torch.LongTensor':
                    c_global_para[key] += total_delta[key].type(torch.LongTensor)
                elif c_global_para[key].type() == 'torch.cuda.LongTensor':
                    c_global_para[key] += total_delta[key].type(torch.cuda.LongTensor)
                else:
                    c_global_para[key] += total_delta[key]
            self.c_e.load_state_dict(c_global_para)

            c_global_para = copy.deepcopy(self.c_c.state_dict())
            total_delta = copy.deepcopy(self.c_c.state_dict())
            for key in total_delta:
                total_delta[key] = 0.0
            for key in self.c_c.state_dict():
                for client in group:
                    total_delta[key] += client.cc_delta_para[key]
                total_delta[key] /= len(group)
                if c_global_para[key].type() == 'torch.LongTensor':
                    c_global_para[key] += total_delta[key].type(torch.LongTensor)
                elif c_global_para[key].type() == 'torch.cuda.LongTensor':
                    c_global_para[key] += total_delta[key].type(torch.cuda.LongTensor)
                else:
                    c_global_para[key] += total_delta[key]
            self.c_c.load_state_dict(c_global_para)

    def test(self):

        # SCAFFOLD
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
        with open('./SCAFFOLD/log/test_acc.txt', 'w') as fp:
            json.dump(self.test_acc, fp=fp)

