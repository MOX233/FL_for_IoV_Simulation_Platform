from ast import If
import copy
import abc
import torch
from torch.utils.data import DataLoader, Dataset



def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg

def avg_dict_list(dict_list):
    if type(dict_list) != list:
        return None
    list_len = len(dict_list)
    if list_len == 0:
        return None
    avg_dict = dict_list[0]
    if type(avg_dict) != dict:
        return None
    for i, d in enumerate(dict_list):
        if i == 0:
            continue
        for k in avg_dict.keys():
            avg_dict[k] += d[k]
    for k in avg_dict.keys():
            avg_dict[k] = avg_dict[k] / list_len
    return avg_dict




class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        data = self.dataset[self.idxs[item]]
        return data



