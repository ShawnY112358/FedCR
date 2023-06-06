import json
import matplotlib.pyplot as plt
import seaborn
import numpy as np
import config
import data_loader
from data_loader import load_dataset, load_from_file
import torch

conf = config.conf

def fogetting(path):
    test_acc_g, test_acc_l = [], []
    for c in range(conf.num_clients):
        with open(path + 'test_acc_%d.txt' % c, 'r') as fp:
            acc = json.load(fp)
            acc_g = acc[: :2]
            acc_l = acc[1: :2]
            test_acc_g.append(acc_g)
            test_acc_l.append(acc_l)

    distribution = []
    for c in range(conf.num_clients):
        train_data, test_data = load_from_file(c)
        pdf = [0 for i in range(conf.num_classes)]
        train_data.sort(key=lambda i: i[1])
        ends = [0]
        for i in range(1, len(train_data)):
            if train_data[i][1] != train_data[i - 1][1]:
                pdf[train_data[i - 1][1].item()] = i - ends[len(ends) - 1]
                ends.append(i)
        pdf[train_data[-1][1].item()] = len(train_data) - ends[-1]
        distribution.append(pdf)
    print(distribution)
    avg_mg_acc, avg_ml_acc = [], []

    for c in range(conf.num_clients):
        for k in range(conf.num_classes):
            if distribution[c][k] == 0:
                avg_mg_acc.append([test_acc_g[c][i][k] for i in range(len(test_acc_g[c]))])
                avg_ml_acc.append([test_acc_l[c][i][k] for i in range(len(test_acc_l[c]))])

    avg_mg_acc = [sum([avg_mg_acc[i][j] for i in range(len(avg_mg_acc))]) / len(avg_mg_acc) for j in range(len(avg_mg_acc[0]))]
    avg_ml_acc = [sum([avg_ml_acc[i][j] for i in range(len(avg_ml_acc))]) / len(avg_ml_acc) for j in range(len(avg_ml_acc[0]))]

    x = np.array(range(len(avg_ml_acc)))
    avg_ml_acc, avg_mg_acc = np.array(avg_ml_acc), np.array(avg_mg_acc)
    return avg_mg_acc, avg_ml_acc

def grad_std(path):
    grad = [[] for i in range(conf.num_clients)]
    for c in range(conf.num_clients):
        with open(path + 'grad_%d.txt' % c, 'r') as fp:
            grad[c] = json.load(fp)

    grad_std = np.array([np.std(np.array([grad[i][j] for i in range(conf.num_clients)])) for j in range(len(grad[0]))])
    return grad_std

def test_acc(path):
    with open(path + 'test_acc.txt', 'r') as fp:
        test_acc = json.load(fp)
        return test_acc

def test_model(path, test_data):
    extractor = torch.load(path + 'extractor.pt').to(conf.device)
    classifier = torch.load(path + 'classifier.pt').to(conf.device)
    extractor_ft = torch.load(path + 'extractor_ft.pt').to(conf.device)
    classifier_ft = torch.load(path + 'classifier_ft.pt').to(conf.device)

    extractor.eval()
    classifier.eval()
    extractor_ft.eval()
    classifier_ft.eval()
    # test accuracy of each class

    test_loader = torch.utils.data.DataLoader(test_data, batch_size=conf.batchsize, shuffle=False)
    total, correct, correct_ft = [0 for i in range(conf.num_classes)], [0 for i in range(conf.num_classes)], [0 for i in range(conf.num_classes)]
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            inputs, labels = inputs.to(conf.device), labels.to(conf.device)
            output = classifier(extractor(inputs))
            _, predicted = torch.max(output.data, 1)
            for p, l in zip(predicted, labels):
                total[l.item()] += 1
                correct[l.item()] += int((p == l).item())

            inputs, labels = data
            inputs, labels = inputs.to(conf.device), labels.to(conf.device)
            output = classifier_ft(extractor_ft(inputs))
            _, predicted = torch.max(output.data, 1)
            for p, l in zip(predicted, labels):
                correct_ft[l.item()] += int((p == l).item())

    acc = [correct[i] / total[i] for i in range(len(correct))]
    acc_ft = [correct_ft[i] / total[i] for i in range(len(correct))]
    return acc, acc_ft



fig_root = 'C:/Users/FNRG/Desktop/figure/'


# # missing classes
# g_sr, l_sr = fogetting(path='C:/Users/FNRG/Desktop/mnist/cnn/1000+500SR/log/')
# g_mr, l_mr = fogetting(path='C:/Users/FNRG/Desktop/mnist/cnn/1000+500MR/log/')
# x_sr = np.array(range(len(g_sr)))
# x_mr = np.array(range(len(g_mr)))
# fig = plt.figure()
# plt.plot(x_sr, g_sr, ls='-', color='b', label='Global')
# plt.plot(x_sr, l_sr, ls='-.', color='b', label='Local')
# plt.plot(x_mr, g_mr, ls='-', color='r', label='Global')
# plt.plot(x_mr, l_mr, ls='-.', color='r', label='Local')
# plt.show()
# plt.close()

# delta w
grad_std_sr = grad_std(path='C:/Users/FNRG/Desktop/20 clients/mnist/IF=0.01/cnn/1000+500SR/log_1/')
grad_std_mr = grad_std(path='C:/Users/FNRG/Desktop/20 clients/mnist/IF=0.01/cnn/FedAvg/log_1/')
fig = plt.figure()
plt.xlabel('# Communication Rounds')
plt.ylabel('$\sigma(||\Delta v||)$')
plt.plot(grad_std_sr, ls='-', color='b', label='Class Re-weighting')
plt.plot(grad_std_mr, ls='-', color='r', label='FedCR')
plt.legend()
# plt.show()
plt.savefig(fig_root + 'grad.jpg')
plt.close()

# test_acc
test_acc_sr = test_acc(path='C:/Users/FNRG/Desktop/mnist/cnn/1000+500SR/log/')
test_acc_mr = test_acc(path='C:/Users/FNRG/Desktop/mnist/cnn/1000+500MR/log/')
fig = plt.figure()
plt.xlabel('# Communication Rounds')
plt.ylabel('Test Accuracy')
plt.plot(test_acc_sr[1000:], ls='-', color='b', label='Class Re-weighting')
plt.plot(test_acc_mr[1000:], ls='-', color='r', label='FedCR')
plt.legend()
# plt.show()
plt.savefig(fig_root + 'test_acc.jpg')
plt.close()



# # top1 acc, mean, std, ablation study
# root = 'C:/Users/FNRG/Desktop/100 clients/cifar/0.01/FedNova/log_'
# top1_w, top1_o, boost = [], [], []
# for i in range(3):
#     path = root + str(i + 1) + '/'
#     acc = test_acc(path=path)
#     top1_o.append(max(acc[:1000]))
#     top1_w.append(max(acc[1000:]))
#     boost.append(top1_w[-1] - top1_o[-1])
# top1_w, top1_o, boost = np.array(top1_w), np.array(top1_o), np.array(boost)
# print(np.mean(top1_w))
# print(np.std(top1_w))
# print(np.mean(top1_o))
# print(np.std(top1_o))
# print(np.mean(boost))
# print(np.std(boost))



# # top1 acc curve w/o calibration
# root = 'C:/Users/FNRG/Desktop/50 clients/cifar/'
# baselines = ['FedAvg', 'FedProx', 'FedNova', 'SCAFFOLD']
# ifs = [0.01, 0.02, 0.1, 0.2, 1]
# top1s = {}
# for b in baselines:
#     top1_w, top1_o = [], []
#     for i in ifs:
#         path = root + 'IF=' + str(i) + '/' + b + '/log_1/'
#         acc = test_acc(path)
#         top1_w.append(max(acc[1000:]))
#         top1_o.append(max(acc[:1000]))
#     top1s[b] = [top1_w, top1_o]
#
# plt.figure(figsize=(12, 5))
# plt.xlabel('# Imbalance Degree ($\\rho$)')
# plt.ylabel('Test Accuracy')
# plt.grid(linestyle=":", color="black", axis='y')
# colors = [(0, 0.4470, 0.7410), (0.8500, 0.3250, 0.0980), (0.9290, 0.6940, 0.1250), (0.4940, 0.1840, 0.5560)]
# markers = ['>', 'o', '*', '+']
# for i in range(len(baselines)):
#     b = baselines[i]
#     plt.plot([1 / f for f in ifs], top1s[b][0], ls='-', marker=markers[i], color=colors[i], label=b + ' + FedCR')
#     plt.plot([1 / f for f in ifs], top1s[b][1], ls='-.', marker=markers[i], color=colors[i], label=b)
#     # plt.plot(ifs, top1s[b][0], ls='-', marker=markers[i], color=colors[i], label=b + ' + Calibration')
#     # plt.plot(ifs, top1s[b][1], ls='-.', marker=markers[i], color=colors[i], label=b)
#
#
# plt.legend()
# # plt.show()
# plt.savefig(fig_root + 'top1_cureve_with_ifs.jpg')
# plt.close()



# # top1 acc, mean, std, main study
# root = 'C:/Users/FNRG/Desktop/100 clients/mnist/0.05/CReFF/log_'
# top1 = []
# for i in range(3):
#     path = root + str(i + 1) + '/'
#     acc = test_acc(path=path)
#     top1.append(max(acc))
# top1 = np.array(top1)
# print(np.mean(top1))
# print(np.std(top1))


# # acc curve with different local epoch
# root = 'C:/Users/FNRG/Desktop/l_epoch/'
# l_epoch = [1, 2, 5, 10, 20, 50, 100]
# top1s = {'FedAvg': [], 'FedFocal': [], 'FedCR': [], 'CCVR': [], 'CReFF': []}
# for l in l_epoch:
#     acc = test_acc(path=root + str(l) + '/FedAvg/log_1/')
#     top1s['FedAvg'].append(max(acc[:int(1000 * 5 / l)]))
#     top1s['FedCR'].append(max(acc[int(1000 * 5 / l):]))
#     acc = test_acc(path=root + str(l) + '/FedFocal/log_1/')
#     top1s['FedFocal'].append(max(acc[:int(1000 * 5 / l)]))
#     acc = test_acc(path=root + str(l) + '/CCVR/log_1/')
#     top1s['CCVR'].append(max(acc))
#     acc = test_acc(path=root + str(l) + '/CReFF/log_1/')
#     top1s['CReFF'].append(max(acc))
#
# plt.figure(figsize=(12, 5))
# plt.xlabel('# Local Epochs')
# plt.ylabel('Test Accuracy')
# plt.grid(linestyle=":", color="black", axis='y')
# colors = [(0, 0.4470, 0.7410), (0.8500, 0.3250, 0.0980), (0.9290, 0.6940, 0.1250), (0.4940, 0.1840, 0.5560), (0.4660, 0.6740, 0.1880), (0.3010, 0.7450, 0.9330)]
# markers = ['>', 'o', '*', '+', '.', '<']
# for i, k in enumerate(top1s.keys()):
#     plt.plot(l_epoch, top1s[k], ls='-', marker=markers[i], color=colors[i], label=k)
#
# plt.legend()
# # plt.show()
# plt.savefig(fig_root + 'l_epoch.jpg')
# plt.close()
#
# # bar
# root = 'C:/Users/FNRG/Desktop/class/cifar/model_1/'
# test_data = data_loader.load_dataset(name='cifar', is_train=False)
# cifar, cifar_ft = test_model(root, test_data)
# root = 'C:/Users/FNRG/Desktop/class/mnist/model_1/'
# test_data = data_loader.load_dataset(name='mnist', is_train=False)
# mnist, mnist_ft = test_model(root, test_data)
#
# bar_width = 0.35
#
# # 绘图
# plt.bar(np.arange(conf.num_classes) - bar_width / 2, cifar, label = 'FedAvg', color = 'steelblue', alpha = 0.8, width = bar_width)
# plt.bar(np.arange(conf.num_classes) + bar_width / 2, cifar_ft, label = 'FedCR', color = 'indianred', alpha = 0.8, width = bar_width)
# # 添加轴标签
# plt.xlabel('# Class ID')
# plt.ylabel('Recall Rate')
# plt.xticks(np.arange(conf.num_classes), np.arange(conf.num_classes))
#
# plt.legend()
# # 显示图形
# # plt.show()
# plt.savefig(fig_root + 'cifar_recall.jpg')
# plt.close()
#
# # 绘图
# plt.bar(np.arange(conf.num_classes) - bar_width / 2, mnist, label = 'FedAvg', color = 'steelblue', alpha = 0.8, width = bar_width)
# plt.bar(np.arange(conf.num_classes) + bar_width / 2, mnist_ft, label = 'FedCR', color = 'indianred', alpha = 0.8, width = bar_width)
# # 添加轴标签
# plt.xlabel('# Class ID')
# plt.ylabel('Recall Rate')
# plt.xticks(np.arange(conf.num_classes), np.arange(conf.num_classes))
#
# plt.legend()
# # 显示图形
# # plt.show()
# plt.savefig(fig_root + 'mnist_recall.jpg')
# plt.close()