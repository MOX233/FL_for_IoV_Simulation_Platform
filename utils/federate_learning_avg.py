#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torch import nn


def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg
    
def FedAvg_city_weighted(w, c, skew=0.5):
    eps = 1e-12
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        w_avg[k] *= skew if c[0]=='MIA' else (1 - skew)
    div_norm = 0 + eps
    for city in c:
        div_norm += skew if city=='MIA' else  (1 - skew)
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += skew * w[i][k] if c[i]=='MIA' else  (1 - skew) * w[i][k]
        w_avg[k] = torch.div(w_avg[k], div_norm)
    return w_avg

def FedAvg_behavior_weighted(w, b, skew=0.5):
    eps = 1e-12
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        w_avg[k] *= skew if b[0]=='go straight' else (1 - skew)
    div_norm = 0 + eps
    for behavior in b:
        div_norm += skew if behavior=='go straight' else  (1 - skew)
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += skew * w[i][k] if b[i]=='go straight' else  (1 - skew) * w[i][k]
        w_avg[k] = torch.div(w_avg[k], div_norm)

    #debug
    #import ipdb;ipdb.set_trace()
    return w_avg
