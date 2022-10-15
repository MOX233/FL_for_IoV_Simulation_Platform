#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @python: 3.6

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from ignite.metrics import TopKCategoricalAccuracy

def test_beam_select(net_g, datatest, args):
    net_g.eval()
    # testing
    test_loss = 0
    loss_func = lambda y_pred, y_true: -torch.sum(torch.mean(y_true[y_pred>0] * torch.log(y_pred[y_pred>0]), axis=0))
    data_loader = DataLoader(datatest, batch_size=1, shuffle=False)
    preds_all = torch.empty((len(data_loader), 256))
    top_1 = TopKCategoricalAccuracy(k=1)
    top_5 = TopKCategoricalAccuracy(k=5)
    top_10 = TopKCategoricalAccuracy(k=10)
    for idx, (lidar, beams) in enumerate(data_loader):
        lidar, beams = lidar.to(args.device), beams.to(args.device)
        pred_beams = net_g(lidar)
        pred_beams = F.softmax(pred_beams, dim=1)

        preds_all[idx, :] = pred_beams
        # sum up batch loss
        test_loss += loss_func(pred_beams, beams).item()
        # get the index of the max log-probability
        top_1.update((pred_beams, torch.argmax(beams)))
        top_5.update((pred_beams, torch.argmax(beams)))
        top_10.update((pred_beams, torch.argmax(beams)))
    net_g.train()
    test_loss /= len(data_loader.dataset)
    top1_acc, top5_acc, top10_acc = top_1.compute(), top_5.compute(), top_10.compute()
    if args.verbose:
        '''print('\nTest set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(data_loader.dataset), accuracy))'''
        print('\nTest set: Average loss: {:.4f} \n Top-1:{:.4f}% Top-5:{:.4f}% Top-10:{:.4f}%\n'.format(
            test_loss, top1_acc * 100., top5_acc * 100., top10_acc * 100.))
    return [top1_acc, top5_acc, top10_acc], test_loss