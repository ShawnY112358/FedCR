import os
from PIL import Image
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import random
import config
import json
import sys

conf = config.conf

transform_cifar = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
transform_mnist = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5,), (0.5,))])

class Data(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, item):
        img, label = self.data[item][0], self.data[item][1]
        return img, label
    def __len__(self):
        return len(self.data)

def load_dataset(name=conf.dataset, is_train=True):
    if name == 'cifar':
        dataset = torchvision.datasets.CIFAR10(root='./dataset', train=is_train, download=True, transform=transform_cifar)
    elif name == 'mnist':
        dataset = torchvision.datasets.MNIST(root='./dataset', train=is_train, download=True, transform=transform_mnist)
    dataset = [dataset[i] for i in range(len(dataset))]
    return dataset

def iid(train_set, test_set, num_samples=500):
    y_pdf = np.array([1 / conf.num_classes for i in range(conf.num_classes)])
    end_point = [0]
    for i in range(1, len(train_set)):
        if train_set[i][1] != train_set[i - 1][1]:
            end_point.append(i)
    end_point.append(len(train_set))
    train = []
    for i, p in enumerate(y_pdf):
        k = int(num_samples * p)
        rs = random.sample(train_set[end_point[i]: end_point[i + 1]], k)
        train.extend(rs)

    end_point = [0]
    for i in range(1, len(test_set)):
        if test_set[i][1] != test_set[i - 1][1]:
            end_point.append(i)
    end_point.append(len(test_set))
    test = []
    for i, p in enumerate(y_pdf):
        k = int(num_samples * p)
        rs = random.sample(test_set[end_point[i]: end_point[i + 1]], k)
        test.extend(rs)

    return train, test

def exp_imbalance(dataset, imb_factor=conf.imb_factor):
    dataset = [dataset[i] for i in range(len(dataset))]
    dataset.sort(key=lambda i: i[1])
    ends = [0]
    for i in range(1, len(dataset)):
        if dataset[i][1] != dataset[i - 1][1]:
            ends.append(i)
    ends.append(len(dataset))

    num_per_class = [ends[c + 1] - ends[c] for c in range(len(ends) - 1)]
    max_num = min(num_per_class)
    num_samples = [int(max_num * imb_factor ** (idx / (len(num_per_class) - 1))) for idx in range(len(num_per_class))]
    print('global pdf:' + str(num_samples))
    res = []
    for c in range(len(num_per_class)):
        res.extend(random.sample(dataset[ends[c]: ends[c + 1]], num_samples[c]))
    return res

def step_imbalance(dataset, imb_factor=conf.imb_factor, m_class_fraction=conf.m_class_fraction):
    dataset = [dataset[i] for i in range(len(dataset))]
    dataset.sort(key=lambda i: i[1])
    ends = [0]
    for i in range(1, len(dataset)):
        if dataset[i][1] != dataset[i - 1][1]:
            ends.append(i)
    ends.append(len(dataset))

    num_per_class = [ends[c + 1] - ends[c] for c in range(len(ends) - 1)]
    max_num = min(num_per_class)
    num_samples = [max_num if idx < len(num_per_class) * (1 - m_class_fraction) else int(max_num * imb_factor) for idx in range(len(num_per_class))]
    print('global pdf:' + str(num_samples))
    res = []
    for c in range(len(num_per_class)):
        res.extend(random.sample(dataset[ends[c]: ends[c + 1]], num_samples[c]))
    return res


# Non-overlap
def pathological(train_set, test_set, num_clients=conf.num_clients, num_classes=conf.num_classes, shard_size=500, pick_num=2):
    train_set = [train_set[i] for i in range(len(train_set))]
    train_set.sort(key=lambda i: i[1])
    test_set = [test_set[i] for i in range(len(test_set))]
    test_set.sort(key=lambda i: i[1])

    ends_tr = [0]
    for i in range(1, len(train_set)):
        if train_set[i][1] != train_set[i - 1][1]:
            ends_tr.append(i)
    ends_tr.append(len(train_set))
    ends_te = [0]
    for i in range(1, len(test_set)):
        if test_set[i][1] != test_set[i - 1][1]:
            ends_te.append(i)
    ends_te.append(len(test_set))

    shards = []
    shard_size = int(len(train_set) / (conf.num_clients * pick_num))
    for c in range(conf.num_classes):
        num_samples = ends_tr[c + 1] - ends_tr[c]
        num_shards = int(num_samples / shard_size)
        groups = [train_set[ends_tr[c] + j * shard_size: ends_tr[c] + (j + 1) * shard_size] for j in range(num_shards)]
        if num_shards * shard_size < num_samples:
            groups.append(train_set[ends_tr[c] + shard_size * num_shards: ends_tr[c] + num_samples])
        shards.extend(groups)

    random.shuffle(shards)
    print('NUM_SHARDS:' + str(len(shards)))

    train_sets, test_sets = [], []
    # each client take pick_num shards
    classes_list = [[0 for j in range(num_classes)] for i in range(num_clients)]

    for client_id in range(conf.num_clients):
        train = []
        for p in range(pick_num):
            train.extend(shards[client_id * pick_num + p])
            class_id = shards[client_id * pick_num + p][0][1]
            classes_list[client_id][class_id] += len(shards[client_id * pick_num + p])

        train_sets.append(train)

    rand_c = random.sample(range(num_clients), len(shards) - num_clients * pick_num)
    for s, c in zip(range(num_clients * pick_num, len(shards)), rand_c):
        train_sets[c].extend(shards[s])
        class_id = shards[s][0][1]
        classes_list[c][class_id] += len(shards[s])

    # build test_set
    for client_id in range(conf.num_clients):
        test = []
        num_samples = [ends_te[c + 1] - ends_te[c] for c in range(num_classes)]
        cardinal = min([int(num_samples[i] / classes_list[client_id][i]) if classes_list[client_id][i] != 0 else float('inf') for i in range(num_classes)])
        for c in range(num_classes):
            test.extend(random.sample(test_set[ends_te[c]: ends_te[c + 1]], cardinal * classes_list[client_id][c]))
        test_sets.append(test)

    print('Data distribution:')
    print(classes_list)

    return train_sets, test_sets

def n_way_k_shot(train_set, test_set, num_clients=conf.num_clients, num_classes=conf.num_classes):

    train_set = [train_set[i] for i in range(len(train_set))]
    train_set.sort(key=lambda i: i[1])
    test_set = [test_set[i] for i in range(len(test_set))]
    test_set.sort(key=lambda i: i[1])

    train_sets, test_sets = [], []

    # test samples
    ends_te = [0]
    for i in range(1, len(test_set)):
        if test_set[i][1] != test_set[i - 1][1]:
            ends_te.append(i)
    ends_te.append(len(test_set))
    num_samples = [ends_te[i + 1] - ends_te[i] for i in range(num_classes)]
    num_samples_test = min(num_samples)

    # train set
    ends_tr = [0]
    for i in range(1, len(train_set)):
        if train_set[i][1] != train_set[i - 1][1]:
            ends_tr.append(i)
    ends_tr.append(len(train_set))

    num_samples = [ends_tr[i + 1] - ends_tr[i] for i in range(num_classes)]
    k_list = [random.randint(50, int(min(num_samples) / 2)) for i in range(num_clients)] # k取值随机选取最少50个样本，最大为数量最少的类的样本数的一半
    n_list = [random.randint(2, num_clients) for i in range(num_clients)]

    for client_id in range(num_clients):
        classes = random.sample(range(num_classes), n_list[client_id])
        train, test = [], []
        for c in classes:
            rs = random.sample(train_set[ends_tr[c]: ends_tr[c + 1]], k_list[client_id])
            train.extend(rs)
            rs = random.sample(test_set[ends_te[c]: ends_te[c + 1]], num_samples_test)
            test.extend(rs)

        train_sets.append(train)
        test_sets.append(test)

    return train_sets, test_sets

def load_from_file(client_no):
    x = torch.load('./data/train_data_%d.pt' % client_no)
    y = torch.load('./data/train_label_%d.pt' % client_no).long()
    train_data = [(x[i], y[i]) for i in range(x.shape[0])]

    if os.path.exists('./data/test_data_%d.pt' % client_no):
        x2 = torch.load('./data/test_data_%d.pt' % client_no)
    else:
        x2 = None
    if os.path.exists('./data/test_label_%d.pt' % client_no):
        y2 = torch.load('./data/test_label_%d.pt' % client_no).long()
    else:
        y2 = None

    test_data = [(x2[i], y2[i]) for i in range(x2.shape[0])] if x2 != None and y2 != None else None
    return train_data, test_data


def save_data(train_data, test_data, index):
    x = torch.stack([train_data[i][0] for i in range(len(train_data))], dim=0)
    y = torch.from_numpy(np.array([train_data[i][1] for i in range(len(train_data))]))
    torch.save(x, './data/train_data_%d.pt' % index)
    torch.save(y, './data/train_label_%d.pt' % index)

    x = torch.stack([test_data[i][0] for i in range(len(test_data))], dim=0)
    y = torch.from_numpy(np.array([test_data[i][1] for i in range(len(test_data))]))
    torch.save(x, './data/test_data_%d.pt' % index)
    torch.save(y, './data/test_label_%d.pt' % index)
