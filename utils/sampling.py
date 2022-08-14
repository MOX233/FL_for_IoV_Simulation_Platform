#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


from functools import partial
import numpy as np
import csv

def sample_iid(dataset, num_items, num_users=1):
    """
    Sample I.I.D. client data from Argoverse dataset
    :param dataset:
    :param num_items: specify every user's local dataset size. type: int or list
    :param num_users:
    :return: dict of image index
    """
    if type(num_items) == int:
        assert num_items>0
        num_items = min([num_items, len(dataset)])
        dict_users, all_idxs = {}, [i for i in range(len(dataset))]
        for i in range(num_users):
            dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=True))
        return dict_users
    elif type(num_items) == list:
        assert len(num_items) == num_users
        assert max(num_items) <= len(dataset)
        dict_users, all_idxs = {}, [i for i in range(len(dataset))]
        for i in range(num_users):
            dict_users[i] = set(np.random.choice(all_idxs, num_items[i], replace=True))
        return dict_users
    else:
        exit('The type of num_items is wrong!')
    

def sample_noniid(dataset, split_dict, num_items, num_users):
    """
    Sample non-I.I.D client data from Argoverse dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    if type(num_items) == int:
        assert num_items>0
        dict_users = {}
        for i in range(num_users):
            key_table = list(split_dict.keys())
            partial_idxs = split_dict[key_table[np.random.randint(len(key_table))]]
            if num_items <= len(partial_idxs):
                dict_users[i] = set(np.random.choice(partial_idxs, num_items, replace=False))
            else:
                dict_users[i] = set(partial_idxs)
        return dict_users
    elif type(num_items) == list:
        assert len(num_items) == num_users
        dict_users = {}
        for i in range(num_users):
            key_table = list(split_dict.keys())
            partial_idxs = split_dict[key_table[np.random.randint(len(key_table))]]
            if num_items[i] <= len(partial_idxs):
                dict_users[i] = set(np.random.choice(partial_idxs, num_items[i], replace=False))
            else:
                dict_users[i] = set(partial_idxs)
        return dict_users
    else:
        exit('The type of num_items is wrong!')