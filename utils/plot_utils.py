#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.8
from genericpath import exists
import os
import matplotlib  # noqa
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa


def min_ignore_None(data_list):
    data_list_ignore_None = []
    for data in data_list:
        if data != None:
            data_list_ignore_None.append(data)
    if len(data_list_ignore_None) == 0:
        return None
    else:
        return min(data_list_ignore_None)

def max_ignore_None(data_list):
    data_list_ignore_None = []
    for data in data_list:
        if data != None:
            data_list_ignore_None.append(data)
    if len(data_list_ignore_None) == 0:
        return None
    else:
        return max(data_list_ignore_None)

def plot_dashed_line_for_None_samples(x, y, color='r'):
    it = iter(y)
    cur_pos = next(it)
    cnt = 0
    n2v_list = []
    v2n_list = []
    while 1:
        try:
            nxt_pos = next(it)
            if cur_pos == None and nxt_pos != None:
                n2v_list.append(cnt+1)
            elif cur_pos != None and nxt_pos == None:
                v2n_list.append(cnt)
            cnt += 1
            cur_pos = nxt_pos
        except StopIteration:
            break
    if len(n2v_list)>0 and len(v2n_list)>0:
        if n2v_list[0] < v2n_list[0]:
            n2v_list.pop(0)
        if n2v_list[-1] <= v2n_list[-1]:
            v2n_list.pop(-1)
    for s,t in zip(v2n_list,n2v_list):
        plt.plot([x[s],x[t]],[y[s],y[t]], linestyle='dashed', color=color)

"""def plot_for_traj_pred_task(args, loss_train, loss_val, metrices_eval, rounds):
    plt.figure(figsize=(15, 15), dpi=100)
    # loss
    plt.subplot(2, 2, 1)
    plt.plot(range(len(loss_train)), loss_train,
             color='r', marker='o', markersize=4, label='Training Loss')
    plot_dashed_line_for_None_samples(range(len(loss_train)), loss_train, color='r')

    plt.plot(range(len(loss_val)), loss_val,
             color='b', label='Validation Loss')
    plt.xlabel('round')
    plt.ylabel('train_loss')
    plt.legend()
    # accuracy
    plt.subplot(2, 2, 2)
    plt.plot(range(len(metrices_eval)), [i["minADE"]
             for i in metrices_eval], color='r', linestyle='--', marker='*', label='minADE(K=6)')
    plt.plot(range(len(metrices_eval)), [i["minFDE"]
             for i in metrices_eval], color='g', linestyle='--', marker='*', label='minFDE(K=6)')
    plt.xlabel('round')
    plt.ylabel('minADE/minFDE')
    plt.legend()
    plt.subplot(2, 2, 4)
    plt.plot(range(len(metrices_eval)), [i["MR"]
             for i in metrices_eval], color='b', label='MR')
    plt.xlabel('round')
    plt.ylabel('MR')
    plt.legend()
    # params description
    plt.subplot(2, 2, 3)
    plt.axis([0, 11, 0, 11])
    plt.axis('off')
    fontsize = 12
    plt.text(0, 10, 'Round Num: {}'.format(rounds), fontsize=fontsize)
    plt.text(0, 9, 'Round Duration: {}'.format(
        args.round_duration), fontsize=fontsize)
    plt.text(0, 8, 'mu for Local Train Delay: {}'.format(
        args.mu_local_train), fontsize=fontsize)
    plt.text(0, 7, 'beta for Local Train Delay: {}'.format(
        args.beta_local_train), fontsize=fontsize)
    plt.text(0, 6, 'Local Iter Num: {}'.format(
        args.local_iter), fontsize=fontsize)
    plt.text(0, 5, 'Local Batch Size: {}'.format(
        args.local_bs), fontsize=fontsize)
    plt.text(0, 4, 'Learning Rate: {}'.format(args.lr), fontsize=fontsize)
    plt.text(0, 3, 'non-i.i.d.: {}'.format(args.non_iid), fontsize=fontsize)
    if min_ignore_None(loss_train)!=None:
        plt.text(0, 1, 'min_train_loss: {:.6f}'.format(
            min_ignore_None(loss_train)), fontsize=fontsize)
    else:
        plt.text(0, 1, 'min_train_loss: {}'.format(
            min_ignore_None(loss_train)), fontsize=fontsize)
    plt.text(0, 0, 'min_val_loss: {:.6f}'.format(min_ignore_None(loss_val)), fontsize=fontsize)

    plt.text(6, 10, 'Lambda: {}'.format(args.Lambda), fontsize=fontsize)
    plt.text(6, 9, 'maxSpeed: {}'.format(args.maxSpeed), fontsize=fontsize)
    plt.text(6, 8, 'delay_download: {}'.format(
        args.delay_download), fontsize=fontsize)
    plt.text(6, 7, 'delay_upload: {}'.format(args.delay_upload), fontsize=fontsize)

    plt.text(6, 5, 'min_minADE: {:.6f}'.format(
        min_ignore_None([i["minADE"] for i in metrices_eval])), fontsize=fontsize)
    plt.text(6, 4, 'min_minADE1: {:.6f}'.format(
        min_ignore_None([i["minADE1"] for i in metrices_eval])), fontsize=fontsize)
    plt.text(6, 3, 'min_minFDE: {:.6f}'.format(
        min_ignore_None([i["minFDE"] for i in metrices_eval])), fontsize=fontsize)
    plt.text(6, 2, 'min_minFDE1: {:.6f}'.format(
        min_ignore_None([i["minFDE1"] for i in metrices_eval])), fontsize=fontsize)
    plt.text(6, 1, 'min_MR: {:.6f}'.format(
        min_ignore_None([i["MR"] for i in metrices_eval])), fontsize=fontsize)
    plt.text(6, 0, 'min_MR1: {:.6f}'.format(
        min_ignore_None([i["MR1"] for i in metrices_eval])), fontsize=fontsize)

    savePath = "./save"
    if args.plot_save_path != "default":
        savePath = args.plot_save_path
    os.makedirs(savePath, exist_ok=True)
    savePath = os.path.join(savePath, 'RoundDuration{}_LocalTrainDelay_mu{}_beta{}_LocalIterNum{}_LocalBatchSize{}_Lambda{}_maxSpeed{}_noniid{}.png'.format(
        args.round_duration, args.mu_local_train, args.beta_local_train, args.local_iter, args.local_bs, args.Lambda, args.maxSpeed, args.non_iid))
    #debug
    #import ipdb;ipdb.set_trace()
    plt.savefig(savePath)
    plt.close()
"""

"""def plot_for_digit_cls_task(args, loss_train, loss_val, metrices_eval, rounds):
    plt.figure(figsize=(15, 15), dpi=100)
    # loss
    plt.subplot(2, 2, 1)
    plt.plot(range(len(loss_train)), loss_train,
             color='r', marker='o', markersize=4, label='Training Loss')
    plot_dashed_line_for_None_samples(range(len(loss_train)), loss_train, color='r')

    plt.plot(range(len(loss_val)), loss_val,
             color='b', label='Validation Loss')
    plt.xlabel('round')
    plt.ylabel('train_loss')
    plt.legend()
    # accuracy
    plt.subplot(2, 2, 2)
    plt.plot(range(len(metrices_eval)), [i["acc"]
             for i in metrices_eval], color='r', linestyle='--', marker='*', label='accuracy')
    plt.xlabel('round')
    plt.ylabel('accuracy')
    plt.legend()
    # params description
    plt.subplot(2, 2, 3)
    plt.axis([0, 11, 0, 11])
    plt.axis('off')
    fontsize = 12
    plt.text(0, 10, 'Round Num: {}'.format(rounds), fontsize=fontsize)
    plt.text(0, 9, 'Round Duration: {}'.format(
        args.round_duration), fontsize=fontsize)
    plt.text(0, 8, 'mu for Local Train Delay: {}'.format(
        args.mu_local_train), fontsize=fontsize)
    plt.text(0, 7, 'beta for Local Train Delay: {}'.format(
        args.beta_local_train), fontsize=fontsize)
    plt.text(0, 6, 'Local Iter Num: {}'.format(
        args.local_iter), fontsize=fontsize)
    plt.text(0, 5, 'Local Batch Size: {}'.format(
        args.local_bs), fontsize=fontsize)
    plt.text(0, 4, 'Learning Rate: {}'.format(args.lr), fontsize=fontsize)
    plt.text(0, 3, 'non-i.i.d.: {}'.format(args.non_iid), fontsize=fontsize)
    if args.non_iid:
        plt.text(0, 2, 'non-i.i.d. type: {}'.format(args.split_type), fontsize=fontsize)
    if min_ignore_None(loss_train)!=None:
        plt.text(0, 1, 'min_train_loss: {:.6f}'.format(
            min_ignore_None(loss_train)), fontsize=fontsize)
    else:
        plt.text(0, 1, 'min_train_loss: {}'.format(
            min_ignore_None(loss_train)), fontsize=fontsize)
    plt.text(0, 0, 'min_val_loss: {:.6f}'.format(min_ignore_None(loss_val)), fontsize=fontsize)

    plt.text(6, 10, 'Lambda: {}'.format(args.Lambda), fontsize=fontsize)
    plt.text(6, 9, 'maxSpeed: {}'.format(args.maxSpeed), fontsize=fontsize)
    plt.text(6, 8, 'delay_download: {}'.format(
        args.delay_download), fontsize=fontsize)
    plt.text(6, 7, 'delay_upload: {}'.format(args.delay_upload), fontsize=fontsize)

    plt.text(6, 5, 'max_accuracy: {:.6f}'.format(
        max_ignore_None([i["acc"] for i in metrices_eval])), fontsize=fontsize)

    savePath = "./save"
    if args.plot_save_path != "default":
        savePath = args.plot_save_path
    os.makedirs(savePath, exist_ok=True)
    savePath = os.path.join(savePath, 'RoundDuration{}_LocalTrainDelay_mu{}_beta{}_LocalIterNum{}_LocalBatchSize{}_Lambda{}_maxSpeed{}_noniid{}_task_{}.png'.format(
        args.round_duration, args.mu_local_train, args.beta_local_train, args.local_iter, args.local_bs, args.Lambda, args.maxSpeed, args.non_iid, args.task))
    #debug
    #import ipdb;ipdb.set_trace()
    plt.savefig(savePath)
    plt.close()
"""