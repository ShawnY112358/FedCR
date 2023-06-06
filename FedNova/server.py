import torch
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from model import load_extractor, load_classifier
import json
from data_loader import load_dataset, Data
import config

conf = config.conf

class nova_Server():
    def __init__(self):
        self.clients = []
        self.y_pdf = torch.from_numpy(np.array([1 / conf.num_classes for i in range(conf.num_classes)]))
        self.weight_cfft = torch.ones([conf.num_classes, 1]).to(conf.device)

        self.extractor = load_extractor().to(conf.device)
        self.classifier = load_classifier().to(conf.device)

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
        with open('./FedNova/log/composed_pdf.txt', 'w') as fp:
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

        with open('./FedNova/log/opt_pdf.txt', 'w') as fp:
            json.dump((torch.mm((A * N).T, P) / torch.mm(A.T, N)).tolist(), fp=fp)

        with open('./FedNova/log/a.txt', 'w') as fp:
            json.dump(A.tolist(), fp=fp)
        self.weight_cfft = A.to(conf.device)

    def aggregate(self, group, finetune):
        # #FedNova
        # for key in self.model_nova.state_dict().keys():
        #     if 'num_batches_tracked' in key:
        #         self.model_nova.state_dict()[key].data.copy_(group[0].model_nova.state_dict()[key])
        #         continue
        #     temp = torch.zeros_like(self.model_nova.state_dict()[key]).to(conf.device)
        #     N, sum = 0, 0
        #     for i in range(len(group)):
        #         temp += group[i].num_data * group[i].model_nova.state_dict()[key] / (group[i].num_data / group[i].batch_size * group[i].num_l_epochs)
        #         sum += group[i].num_data * (group[i].num_data / group[i].batch_size * group[i].num_l_epochs)
        #         N += group[i].num_data
        #     self.model_nova.state_dict()[key].data.copy_((temp * sum) / (N * N))
        #
        # #Finetune
        # if 'ft' in conf.algorithm:
        #     for key in self.model_ft.state_dict().keys():
        #         if 'num_batches_tracked' in key:
        #             self.model_ft.state_dict()[key].data.copy_(group[0].model_ft.state_dict()[key])
        #             continue
        #         temp = torch.zeros_like(self.model_ft.state_dict()[key]).to(conf.device)
        #         N = 0
        #         if 'FC' in key:
        #             for i in range(len(group)):
        #                 temp += self.weight_cfft[group[i].index].item() * group[i].num_data * group[i].model_ft.state_dict()[key]
        #                 N += self.weight_cfft[group[i].index].item() * group[i].num_data
        #             self.model_ft.state_dict()[key].data.copy_(temp / N)

        #FedNova
        for key in self.extractor.state_dict().keys():
            if 'num_batches_tracked' in key:
                self.extractor.state_dict()[key].data.copy_(group[0].extractor.state_dict()[key])
                continue
            grad = torch.zeros_like(self.extractor.state_dict()[key]).to(conf.device)
            N, sum = 0, 0
            for i in range(len(group)):
                # normalized gradients
                grad += group[i].num_data * (group[i].extractor.state_dict()[key] - self.extractor.state_dict()[key]) / (group[i].num_data / group[i].batch_size * group[i].num_l_epochs)
                sum += group[i].num_data * (group[i].num_data / group[i].batch_size * group[i].num_l_epochs)
                N += group[i].num_data
            self.extractor.state_dict()[key].data.copy_(self.extractor.state_dict()[key] + (grad * sum) / (N * N))

        if not finetune:
            for key in self.classifier.state_dict().keys():
                if 'num_batches_tracked' in key:
                    self.classifier.state_dict()[key].data.copy_(group[0].classifier.state_dict()[key])
                    continue
                grad = torch.zeros_like(self.classifier.state_dict()[key]).to(conf.device)
                N, sum = 0, 0
                for i in range(len(group)):
                    # normalized gradients
                    grad += group[i].num_data * (group[i].classifier.state_dict()[key] - self.classifier.state_dict()[key]) / (group[i].num_data / group[i].batch_size * group[i].num_l_epochs)
                    sum += group[i].num_data * (group[i].num_data / group[i].batch_size * group[i].num_l_epochs)
                    N += group[i].num_data
                self.classifier.state_dict()[key].data.copy_(self.classifier.state_dict()[key] + (grad * sum) / (N * N))
        else:
            #Finetune
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


    def test(self, log=True, finetune=False):

        # FedNova
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
        with open('./FedNova/log/test_acc.txt', 'w') as fp:
            json.dump(self.test_acc, fp)

