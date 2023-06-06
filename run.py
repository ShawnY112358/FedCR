import json

import data_loader
from data_loader import save_data, iid, load_from_file
import os
import config
from FedAvg.server import avg_Server
from FedAvg.client import avg_Client
from FedProx.client import prox_Client
from FedProx.server import prox_Server
from FedNova.server import nova_Server
from FedNova.client import nova_Client
from Local.client import local_Client
from CReFF.client import creff_Client
from CReFF.server import creff_Server
from SCAFFOLD.server import scaffold_Server
from SCAFFOLD.client import scaffold_Client
from FedFocal.client import focal_Client
from FedFocal.server import focal_Server
from CCVR.server import ccvr_Server
from CCVR.client import ccvr_Client
import random
import shutil
import torch
conf = config.conf

def init_Local():
    if os.path.exists('./Local/log'):
        shutil.rmtree("./Local/log")
    os.mkdir('./Local/log')
    if os.path.exists('./Local/model'):
        shutil.rmtree("./Local/model")
    os.mkdir('./Local/model')

    clients = []
    for i in range(conf.num_clients):
        train_data, test_data = load_from_file(i)
        client = local_Client(i, train_data, test_data)
        clients.append(client)

    return clients

def run_Local():
    clients = init_Local()
    for g_epoch in range(conf.nums_g_epoch):
        for client in clients:
            client.train()
        for client in clients:
            client.test()

    avg_acc = [sum([client.test_acc[i] for client in clients]) / len(clients) for i in range(len(clients[0].test_acc))]
    with open('./Local/log/test_acc_avg.txt', 'w') as fp:
        json.dump(avg_acc, fp=fp)


def init_FedAvg():
    if os.path.exists('./FedAvg/log'):
        shutil.rmtree("./FedAvg/log")
    os.mkdir('./FedAvg/log')
    if os.path.exists('./FedAvg/model'):
        shutil.rmtree("./FedAvg/model")
    os.mkdir('./FedAvg/model')
    clients = []
    server = avg_Server()
    for i in range(conf.num_clients):
        train_data, test_data = load_from_file(i)
        client = avg_Client(i, train_data, test_data, server)
        clients.append(client)
    server.init_client_info(clients)
    return clients, server

def run_FedAvg(finetune=False):
    clients, server = init_FedAvg()
    server.init_weight_cfft()
    for g_epoch in range(conf.nums_g_epoch):
        for client in clients:
            client.down_model()
        group = random.sample(clients, int(conf.num_clients * conf.select_rate))
        for client in group:
            print("global_epoch: %d" % g_epoch)
            client.train()
        server.aggregate(group)
        server.test()
    torch.save(server.extractor, './FedAvg/model/extractor.pt')
    torch.save(server.classifier, './FedAvg/model/classifier.pt')

    if finetune:
        server.init_classifier()
        for ft_epoch in range(conf.nums_ft_epoch):
            for client in clients:
                client.down_model()
                if (ft_epoch + 1) % 10 == 0:
                    client.save_model()
            group = random.sample(clients, int(conf.num_clients * conf.select_rate))
            for client in group:
                print("ft_epoch: %d" % ft_epoch)
                client.finetune()
            server.calibrate(group)
            server.test()
    torch.save(server.extractor, './FedAvg/model/extractor_ft.pt')
    torch.save(server.classifier, './FedAvg/model/classifier_ft.pt')

def init_FedProx():
    if os.path.exists('./FedProx/log'):
        shutil.rmtree("./FedProx/log")
    os.mkdir('./FedProx/log')
    if os.path.exists('./FedProx/model'):
        shutil.rmtree("./FedProx/model")
    os.mkdir('./FedProx/model')

    clients = []
    server = prox_Server()

    for i in range(conf.num_clients):
        train_data, test_data = load_from_file(i)
        client = prox_Client(i, train_data, test_data, server)
        server.clients.append(client)
        clients.append(client)

    return clients, server


def run_FedProx(finetune=False):
    clients, server = init_FedProx()
    for g_epoch in range(conf.nums_g_epoch):
        for client in clients:
            client.down_model()
        group = random.sample(clients, int(conf.num_clients * conf.select_rate))
        for client in group:
            print("global_epoch: %d" % g_epoch)
            client.train()
        # for client in clients:
        #     client.test()
        #     if (g_epoch + 1) % 10 == 0:
        #         client.save_model()
        server.aggregate()
        server.test()

    if finetune:
        server.init_weight_cfft()
        server.init_classifier()
        for g_epoch in range(conf.nums_ft_epoch):
            for client in clients:
                client.down_model()
            group = random.sample(clients, int(conf.num_clients * conf.select_rate))
            for client in group:
                print("ft_epoch: %d" % g_epoch)
                client.finetune()
            # for client in clients:
            #     client.test()
            #     if (g_epoch + 1) % 10 == 0:
            #         client.save_model()
            server.aggregate(finetune=True)
            server.test()


def init_FedNova():
    if os.path.exists('./FedNova/log'):
        shutil.rmtree("./FedNova/log")
    os.mkdir('./FedNova/log')
    if os.path.exists('./FedNova/model'):
        shutil.rmtree("./FedNova/model")
    os.mkdir('./FedNova/model')

    clients = []

    server = nova_Server()

    for i in range(conf.num_clients):
        train_data, test_data = load_from_file(i)
        client = nova_Client(i, train_data, test_data, server)
        server.clients.append(client)
        clients.append(client)

    return clients, server


def run_FedNova(finetune=False):
    clients, server = init_FedNova()
    for g_epoch in range(conf.nums_g_epoch):
        for client in clients:
            client.down_model()
        group = random.sample(clients, int(conf.num_clients * conf.select_rate))
        for client in group:
            print("global_epoch: %d" % g_epoch)
            client.train()
        # for client in clients:
        #     client.test()
        #     if (g_epoch + 1) % 10 == 0:
        #         client.save_model()
        server.aggregate(group=group, finetune=False)
        server.test()

    if finetune:
        server.init_weight_cfft()
        server.init_classifier()
        for ft_epoch in range(conf.nums_ft_epoch):
            for client in clients:
                client.down_model()
            group = random.sample(clients, int(conf.num_clients * conf.select_rate))
            for client in group:
                print("ft_epoch: %d" % ft_epoch)
                client.finetune()
            # for client in clients:
            #     client.test()
            #     if (ft_epoch + 1) % 10 == 0:
            #         client.save_model()
            server.aggregate(group=group, finetune=True)
            server.test()


def init_SCAFFOLD():
    if os.path.exists('./SCAFFOLD/log'):
        shutil.rmtree("./SCAFFOLD/log")
    os.mkdir('./SCAFFOLD/log')
    if os.path.exists('./SCAFFOLD/model'):
        shutil.rmtree("./SCAFFOLD/model")
    os.mkdir('./SCAFFOLD/model')

    clients = []
    server = scaffold_Server()

    for i in range(conf.num_clients):
        train_data, test_data = load_from_file(i)
        client = scaffold_Client(i, train_data, test_data, server)
        server.clients.append(client)
        clients.append(client)

    return clients, server


def run_SCAFFOLD(finetune=False):
    clients, server = init_SCAFFOLD()
    for g_epoch in range(conf.nums_g_epoch):
        for client in clients:
            client.down_model()
        group = random.sample(clients, int(conf.num_clients * conf.select_rate))
        for client in group:
            print("global_epoch: %d" % g_epoch)
            client.train()
        # for client in clients:
        #     client.test()
        #     if (g_epoch + 1) % 10 == 0:
        #         client.save_model()
        server.aggregate(group=group, finetune=False)
        server.test()

    if finetune:
        server.init_weight_cfft()
        server.init_classifier()
        for g_epoch in range(conf.nums_ft_epoch):
            for client in clients:
                client.down_model()
            group = random.sample(clients, int(conf.num_clients * conf.select_rate))
            for client in group:
                print("ft_epoch: %d" % g_epoch)
                client.finetune()
            # for client in clients:
            #     client.test()
            #     if (g_epoch + 1) % 10 == 0:
            #         client.save_model()
            server.aggregate(group=group, finetune=True)
            server.test()

def init_CReFF():

    if os.path.exists('./CReFF/log'):
        shutil.rmtree("./CReFF/log")
    os.mkdir('./CReFF/log')
    if os.path.exists('./CReFF/model'):
        shutil.rmtree("./CReFF/model")
    os.mkdir('./CReFF/model')

    clients = []
    server = creff_Server()

    for i in range(conf.num_clients):
        train_data, test_data = load_from_file(i)
        client = creff_Client(i, train_data, test_data, server)
        server.clients.append(client)
        clients.append(client)

    return clients, server

def run_CReFF():
    clients, server = init_CReFF()
    for g_epoch in range(conf.nums_g_epoch):
        for client in clients:
            client.down_model()
        group = random.sample(clients, int(conf.num_clients * conf.select_rate))
        for client in group:
            print("global_epoch: %d" % g_epoch)
            client.compute_gradients()
            client.train()
        server.aggregate(group)
        server.update_feature(group)
        server.retrain()
        server.test()


def init_FedFocal():

    if os.path.exists('./FedFocal/log'):
        shutil.rmtree("./FedFocal/log")
    os.mkdir('./FedFocal/log')
    if os.path.exists('./FedFocal/model'):
        shutil.rmtree("./FedFocal/model")
    os.mkdir('./FedFocal/model')

    clients = []
    server = focal_Server()

    for i in range(conf.num_clients):
        train_data, test_data = load_from_file(i)
        client = focal_Client(i, train_data, test_data, server)
        server.clients.append(client)
        clients.append(client)

    return clients, server

def run_FedFocal(finetune=False):
    clients, server = init_FedFocal()
    for g_epoch in range(conf.nums_g_epoch):
        for client in clients:
            client.down_model()
        # for client in clients:
        #     client.test()
        #     if (g_epoch + 1) % 10 == 0:
        #         client.save_model()
        group = random.sample(clients, int(conf.num_clients * conf.select_rate))
        for client in group:
            print("global_epoch: %d" % g_epoch)
            client.train()
        server.aggregate()
        server.test()

def init_CCVR():

    if os.path.exists('./CCVR/log'):
        shutil.rmtree("./CCVR/log")
    os.mkdir('./CCVR/log')
    if os.path.exists('./CCVR/model'):
        shutil.rmtree("./CCVR/model")
    os.mkdir('./CCVR/model')

    clients = []
    server = ccvr_Server()

    for i in range(conf.num_clients):
        train_data, test_data = load_from_file(i)
        client = ccvr_Client(i, train_data, test_data, server)
        server.clients.append(client)
        clients.append(client)

    return clients, server

def run_CCVR(finetune=False):
    clients, server = init_CCVR()
    for g_epoch in range(conf.nums_g_epoch):
        for client in clients:
            client.down_model()
        group = random.sample(clients, int(conf.num_clients * conf.select_rate))
        for client in group:
            print("global_epoch: %d" % g_epoch)
            client.train()
        server.aggregate()
        server.test()

    for client in clients:
        client.update_distribution_info()

    server.calibration()

