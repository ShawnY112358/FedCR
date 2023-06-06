import torch
from data_loader import Data, load_dataset, load_from_file
import config
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set()
conf = config.conf

def test(net):
    data = load_testset()
    data = Data(data)
    testloader = torch.utils.data.DataLoader(data, batch_size=1000, shuffle=True)

    # prepare to count predictions for each class
    correct_pred = [0 for i in range(conf.num_classes)]
    total_pred = [0 for i in range(conf.num_classes)]

    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predictions = torch.max(outputs, 1)
            # collect the correct predictions for each class
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[label] += 1
                print(label)
                total_pred[label] += 1
    return [correct_pred[i] / total_pred[i] for i in range(len(total_pred))]

def plot(path, name):
    model_avg = torch.load(path + 'model_avg.pt', map_location=conf.device).eval()
    model_ft = torch.load(path + 'model_ft.pt', map_location=conf.device).eval()
    model_wa = torch.load(path + 'model_wa.pt', map_location=conf.device).eval()
    model_sr = torch.load(path + 'model_sr.pt', map_location=conf.device).eval()

    avg_acc = test(model_avg)
    ft_acc = test(model_ft)
    wa_acc = test(model_wa)
    sr_acc = test(model_sr)


    width = 0.2
    x = list(range(conf.num_classes))
    q1 = [i - 3 * width / 2 for i in x]
    q2 = [i - width / 2 for i in x]
    q3 = [i + width / 2 for i in x]
    q4 = [i + 3 * width / 2 for i in x]

    plt.bar(q1, avg_acc, width=width, label='FedAvg')
    plt.bar(q2, ft_acc, width=width, label='FedMR')
    plt.bar(q3, wa_acc, width=width, label='FedAvg-v2')
    plt.bar(q4, sr_acc, width=width, label='FedAvg-sr')
    plt.xlabel('Class')
    plt.ylabel('Test accuracy')
    plt.legend()
    plt.savefig(name)
    plt.close()


# path1 = './model/dirichlet-5clients/model1/'
# path2 = './model/dirichlet-10clients/model1/'
# path3 = './model/pathological-5clients/model1/'
# path4 = './model/pathological-10clients/model1/'
#
# plot(path1, name='./model/fig8_a')
# plot(path2, name='./model/fig8_b')
# plot(path3, name='./model/fig8_c')
# plot(path4, name='./model/fig8_d')

def plot_pathological():
    data = []
    for client_no in range(5):
        classes = [0 for i in range(10)]
        y = torch.load('./partition/pathological-5clients/data/label_%d.pt' % client_no).long()
        for i in y:
            classes[i] += 1
        data.append(classes)

    data = np.array(data).T
    sns.heatmap(data, annot=True, fmt='d', cmap="YlGnBu")
    plt.xlabel('Client No.')
    plt.ylabel('Class No.')
    plt.savefig('./partition/p5_distribution')
    plt.close()

    data = []
    for client_no in range(10):
        classes = [0 for i in range(10)]
        y = torch.load('./partition/pathological-10clients/data/label_%d.pt' % client_no).long()
        for i in y:
            classes[i] += 1
        data.append(classes)

    data = np.array(data).T
    sns.heatmap(data, annot=True, fmt='d', cmap="YlGnBu")
    plt.xlabel('Client No.')
    plt.ylabel('Class No.')
    plt.savefig('./partition/p10_distribution')
    plt.close()

    data = []
    for client_no in range(5):
        classes = [0 for i in range(10)]
        y = torch.load('./partition/dirichlet-5clients/data/label_%d.pt' % client_no).long()
        for i in y:
            classes[i] += 1
        data.append(classes)

    data = np.array(data).T
    sns.heatmap(data, annot=True, fmt='d', cmap="YlGnBu")
    plt.xlabel('Client No.')
    plt.ylabel('Class No.')
    plt.savefig('./partition/d5_distribution')
    plt.close()

    data = []
    for client_no in range(10):
        classes = [0 for i in range(10)]
        y = torch.load('./partition/dirichlet-10clients/data/label_%d.pt' % client_no).long()
        for i in y:
            classes[i] += 1
        data.append(classes)

    data = np.array(data).T
    sns.heatmap(data, annot=True, fmt='d', cmap="YlGnBu")
    plt.xlabel('Client No.')
    plt.ylabel('Class No.')
    plt.savefig('./partition/d10_distribution')
    plt.close()

plot_pathological()