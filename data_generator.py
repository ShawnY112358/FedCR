import shutil
import data_loader
from data_loader import save_data, iid, load_from_file, exp_imbalance, step_imbalance
import config
import torch
import numpy as np
import os
import argparse

conf = config.conf

def generate_data(partition='n_way_k_shot'):
    if os.path.exists('./data'):
        shutil.rmtree('./data')
    os.mkdir('./data')

    train_set, test_set = data_loader.load_dataset(name=conf.dataset), data_loader.load_dataset(name=conf.dataset)
    if conf.longtail_type == 'exp':
        train_set = exp_imbalance(dataset=train_set)
    else:
        train_set = step_imbalance(dataset=train_set)

    if partition == 'n_way_k_shot':
        train_sets, test_sets = data_loader.n_way_k_shot(train_set, test_set)
    elif partition == 'pathological':
        train_sets, test_sets = data_loader.pathological(train_set, test_set)

    for i, data in enumerate(zip(train_sets, test_sets)):
        train_data, test_data = data
        data_loader.save_data(train_data, test_data, i)

# parser = argparse.ArgumentParser()
# parser.add_argument('-p', '--partition', help='partitioning methods')
# args = parser.parse_args()
generate_data(conf.partition)