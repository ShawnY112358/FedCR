import torch
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from model import load_extractor, load_classifier
import json
from data_loader import load_dataset, Data
import config

conf = config.conf

class avg_Server():
    def __init__(self):
        self.y_pdf = torch.from_numpy(np.array([1 / conf.num_classes for i in range(conf.num_classes)]))
        self.weight_cfft = torch.ones([conf.num_classes, 1]).to(conf.device)

        self.extractor = load_extractor().to(conf.device)
        self.classifier = load_classifier().to(conf.device)

        self.test_set = load_dataset(name=conf.dataset, is_train=False)
        self.test_loader = torch.utils.data.DataLoader(self.test_set, batch_size=conf.batchsize, shuffle=False)

        self.prototype = [torch.rand([conf.prototype_size]) for c in range(conf.num_classes)]
        self.test_acc = []

    def init_client_info(self, clients):
        self.clients = clients

    def init_classifier(self):
        self.classifier = load_classifier().to(conf.device)

    # optimize a and calculate aggregation weights of FedMR; calculate sample weights of sample re-weighting
    def init_weight_cfft(self):
        lr = 0.1 # learning rate
        num_steps = 100000 # optimization steps
        sigma = 1e-4 # punish term coefficient of FedMR, denoted by mu in the paper

        P, N = [], []
        for client in self.clients:
            P.append(client.pdf)
            N.append([client.num_data])

        P = torch.from_numpy(np.array(P))
        N = torch.from_numpy(np.array(N)).double()

        A = torch.from_numpy(np.array([[0.5] for i in range(len(self.clients))])).double() # initialize a with 0.5 uniformly
        A.requires_grad = True

        optimizer = optim.Adam([A], lr=lr)
        target_pdf = self.y_pdf # P_test
        print(P)
        print("composed_pdf")
        print(torch.mm((A * N).T, P) / torch.mm(A.T, N)) # P_train (hat)
        with open('./FedAvg/log/composed_pdf.txt', 'w') as fp:
            json.dump((torch.mm((A * N).T, P) / torch.mm(A.T, N)).tolist(), fp=fp)

        # calculate sample weights of sample re-weighting
        self.sample_weight = torch.div(self.y_pdf, torch.mm((A * N).T, P) / torch.mm(A.T, N)).float().detach().to(conf.device).reshape([conf.num_classes])
        print(self.sample_weight)

        # optimization
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
            A.data.clamp_(0.01, 0.99) # restrict the domain of a

        print("after optimization")
        print(torch.mm((A * N).T, P) / torch.mm(A.T, N)) # optimized pdf
        print(A)

        with open('./FedAvg/log/opt_pdf.txt', 'w') as fp:
            json.dump((torch.mm((A * N).T, P) / torch.mm(A.T, N)).tolist(), fp=fp)

        with open('./FedAvg/log/a.txt', 'w') as fp:
            json.dump(A.tolist(), fp=fp)
        self.weight_cfft = A.to(conf.device)

    def estimate_pdf(self):
        pass

    def aggregate(self, group):
        for c in range(conf.num_classes):
            prototype = torch.zeros_like(self.prototype[0])
            N = 0
            for client in group:
                if client.class_count[c] != 0:
                    prototype += client.feature[c] * client.class_count[c]
                N += client.class_count[c]
            self.prototype[c] = prototype / N

        for key in self.extractor.state_dict().keys():
            if 'num_batches_tracked' in key:
                self.extractor.state_dict()[key].data.copy_(group[0].extractor.state_dict()[key])
                continue
            temp = torch.zeros_like(self.extractor.state_dict()[key]).to(conf.device)
            N = 0
            for client in group:
                temp += client.num_data * client.extractor.state_dict()[key]
                N += client.num_data
            self.extractor.state_dict()[key].data.copy_(temp / N)

        for key in self.classifier.state_dict().keys():
            if 'num_batches_tracked' in key:
                self.classifier.state_dict()[key].data.copy_(group[0].classifier.state_dict()[key])
                continue
            temp = torch.zeros_like(self.classifier.state_dict()[key]).to(conf.device)
            N = 0
            for client in group:
                temp += client.num_data * client.classifier.state_dict()[key]
                N += client.num_data
            self.classifier.state_dict()[key].data.copy_(temp / N)

    def calibrate(self, group):
        for key in self.classifier.state_dict().keys():
            if 'num_batches_tracked' in key:
                self.classifier.state_dict()[key].data.copy_(group[0].classifier.state_dict()[key])
                continue
            temp = torch.zeros_like(self.classifier.state_dict()[key]).to(conf.device)
            N = 0
            for client in group:
                temp += self.weight_cfft[client.index].item() * client.num_data * client.classifier.state_dict()[key]
                N += self.weight_cfft[client.index].item() * client.num_data
            self.classifier.state_dict()[key].data.copy_(temp / N)

    def test(self):
        # FedAvg
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
        with open('./FedAvg/log/test_acc.txt', 'w') as fp:
            json.dump(self.test_acc, fp)