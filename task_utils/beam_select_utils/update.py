#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
import torch.nn.functional as F
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        lidar, beam = self.dataset[self.idxs[item]]
        return lidar, beam


# =============================================================================
# class LocalUpdate(object):
#     def __init__(self, args, dataset=None, idxs=None):
#         self.args = args
#         self.loss_func = lambda y_pred, y_true: -torch.sum(torch.mean(y_true[y_pred>0] * torch.log(y_pred[y_pred>0]), axis=0))
#         self.selected_clients = []
#         self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)
# 
#     def train(self, net):
#         net.train()
#         # train and update
#         optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
# 
#         iter_loss = []
#         for iter in range(self.args.local_iter):
#             batch_loss = []
#             for batch_idx, (lidars, beams) in enumerate(self.ldr_train):
#                 lidars, beams = lidars.to(self.args.device), beams.to(self.args.device)
#                 net.zero_grad()
#                 pred_beams = net(lidars)
#                 pred_beams = F.softmax(pred_beams, dim=1)
#                 loss = self.loss_func(pred_beams, beams)
#                 loss.backward()
#                 optimizer.step()
#                 if self.args.verbose and batch_idx % 10 == 0:
#                     print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
#                         iter, batch_idx * len(lidars), len(self.ldr_train.dataset),
#                                100. * batch_idx / len(self.ldr_train), loss.item()))
#                 batch_loss.append(loss.item())
#             iter_loss.append(sum(batch_loss)/len(batch_loss))
#         return net.state_dict(), sum(iter_loss) / len(iter_loss)
# =============================================================================

# =============================================================================
# class LocalUpdateFlexible(object):
#     # local_iter and local_batch_size can be given flexibly
#     def __init__(self, args, dataset=None, idxs=None, local_bs=1):
#         self.args = args
#         self.loss_func = lambda y_pred, y_true: -torch.sum(torch.mean(y_true[y_pred>0] * torch.log(y_pred[y_pred>0]), axis=0))
#         self.selected_clients = []
#         self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=local_bs, shuffle=True, drop_last=True)
#         #self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=local_bs, shuffle=True, drop_last=True, persistent_workers=True)
# 
#     def train(self, net, local_iter=1):
#         net.train()
#         # train and update
#         optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
#         iter_loss = []
#         for iter in range(local_iter):
#             batch_loss = []
#             for batch_idx, (lidars, beams) in enumerate(self.ldr_train):
#                 lidars, beams = lidars.to(self.args.device), beams.to(self.args.device)
#                 net.zero_grad()
#                 pred_beams = net(lidars)
#                 pred_beams = F.softmax(pred_beams, dim=1)
#                 loss = self.loss_func(pred_beams, beams)
#                 loss.backward()
#                 optimizer.step()
#                 if self.args.verbose and batch_idx % 10 == 0:
#                     print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
#                         iter, batch_idx * len(lidars), len(self.ldr_train.dataset),
#                                100. * batch_idx / len(self.ldr_train), loss.item()))
#                 batch_loss.append(loss.item())
#             iter_loss.append(sum(batch_loss)/len(batch_loss))
#         return net.state_dict(), sum(iter_loss) / len(iter_loss)
# =============================================================================
    
class LocalUpdate_for_beam_select(object):
    # local_iter and local_batch_size can be given flexibly
    def __init__(self, args, dataset=None, idxs=None, local_bs=1):
        self.args = args
        self.loss_func = lambda y_pred, y_true: -torch.sum(torch.mean(y_true[y_pred>0] * torch.log(y_pred[y_pred>0]), axis=0))
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=local_bs, shuffle=True, drop_last=True)

    def train(self, net, local_iter=1):
        net.train()
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        iter_loss = []
        local_epoch = int(local_iter/len(self.ldr_train))
        for epoch in range(local_epoch):
            for batch_idx, (lidars, beams) in enumerate(self.ldr_train):
                lidars, beams = lidars.to(self.args.device), beams.to(self.args.device)
                net.zero_grad()
                pred_beams = net(lidars)
                pred_beams = F.softmax(pred_beams, dim=1)
                loss = self.loss_func(pred_beams, beams)
                loss.backward()
                optimizer.step()
                iter_loss.append(loss.item())
        for batch_idx, (lidars, beams) in enumerate(self.ldr_train):
            if batch_idx<local_iter-local_epoch*len(self.ldr_train):
                lidars, beams = lidars.to(self.args.device), beams.to(self.args.device)
                net.zero_grad()
                pred_beams = net(lidars)
                pred_beams = F.softmax(pred_beams, dim=1)
                loss = self.loss_func(pred_beams, beams)
                loss.backward()
                optimizer.step()
                iter_loss.append(loss.item())
            else:
                break
        return net.state_dict(), sum(iter_loss) / len(iter_loss)

