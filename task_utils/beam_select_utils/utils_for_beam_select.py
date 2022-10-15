#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.8
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


def plot_loss_curve(loss_train, loss_val, rounds, args):
    plt.figure(1)
    plt.plot(range(len(loss_train)), loss_train,
             color='r', label='Training Loss')
    plt.plot(range(len(loss_val)), loss_val,
             color='b', label='Validation Loss')
    plt.xlabel('round')
    plt.ylabel('train_loss')
    plt.legend()
    #plt.savefig('./save/loss_round{}_localiter{}_localbs{}.png'.format(rounds, args.local_iter, args.local_bs))
    plt.savefig('./save/LOSS_RoundDuration{}_LocalTrainTime{}_RoundNum{}_LocalIterNum_{}_Localbs{}.png'.format(
        args.round_duration, args.local_train_time, rounds, args.local_iter, args.local_bs))
    plt.close()


def plot_acc_curve(acc_val, rounds, args):
    plt.figure(2)
    plt.plot(range(len(acc_val)), [i[0]
             for i in acc_val], color='r', label='Top-1')
    plt.plot(range(len(acc_val)), [i[1]
             for i in acc_val], color='g', label='Top-5')
    plt.plot(range(len(acc_val)), [i[2]
             for i in acc_val], color='b', label='Top-10')
    plt.xlabel('round')
    plt.ylabel('validation_accuracy')
    plt.legend()
    #plt.savefig('./save/acc_round{}_localiter{}_localbs{}.png'.format(rounds, args.local_iter, args.local_bs))
    plt.savefig('./save/ACC_RoundDuration{}_LocalTrainTime{}_RoundNum{}_LocalIterNum_{}_Localbs{}.png'.format(
        args.round_duration, args.local_train_time, rounds, args.local_iter, args.local_bs))
    plt.close()


def plot_for_beam_select_task(args, loss_train, loss_val, eval_metrices, rounds):
    plt.figure(figsize=(6, 13), dpi=100)
    # loss
    plt.subplot(3, 1, 1)
    plt.plot(range(len(loss_train)), loss_train,
             color='r', label='Training Loss')
    plt.plot(range(len(loss_val)), loss_val,
             color='b', label='Validation Loss')
    plt.xlabel('round')
    plt.ylabel('train_loss')
    plt.legend()
    # accuracy
    plt.subplot(3, 1, 2)
    plt.plot(range(len(eval_metrices)), [i['top1_acc']
             for i in eval_metrices], color='r', label='Top-1')
    plt.plot(range(len(eval_metrices)), [i['top5_acc']
             for i in eval_metrices], color='g', label='Top-5')
    plt.plot(range(len(eval_metrices)), [i['top10_acc']
             for i in eval_metrices], color='b', label='Top-10')
    plt.xlabel('round')
    plt.ylabel('validation_accuracy')
    plt.legend()
    # params description
    plt.subplot(3, 1, 3)
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
    if min_ignore_None(loss_train) != None:
        plt.text(0, 2, 'min_train_loss {:.5f}'.format(
            min_ignore_None(loss_train)), fontsize=fontsize)
        plt.text(0, 1, 'min_val_loss {:.5f}'.format(min_ignore_None(loss_val)), fontsize=fontsize)


    plt.text(6, 10, 'Lambda: {}'.format(args.Lambda), fontsize=fontsize)
    plt.text(6, 9, 'maxSpeed: {}'.format(args.maxSpeed), fontsize=fontsize)
    plt.text(6, 6, 'delay_download: {}'.format(
        args.delay_download), fontsize=fontsize)
    plt.text(6, 5, 'delay_upload: {}'.format(args.delay_upload), fontsize=fontsize)
    plt.text(6, 2, 'max_top1_acc {:.4f}%'.format(
        100*max_ignore_None([i['top1_acc'] for i in eval_metrices])), fontsize=fontsize)
    plt.text(6, 1, 'max_top5_acc {:.4f}%'.format(
        100*max_ignore_None([i['top5_acc'] for i in eval_metrices])), fontsize=fontsize)
    plt.text(6, 0, 'max_top10_acc {:.4f}%'.format(
        100*max_ignore_None([i['top10_acc'] for i in eval_metrices])), fontsize=fontsize)

    savePath = "./save"
    if args.plot_save_path != "default":
        savePath = args.plot_save_path
    os.makedirs(savePath, exist_ok=True)
    png_savePath = os.path.join(savePath, 'RoundDuration{}_LocalTrainDelay_mu{}_beta{}_LocalIterNum{}_LocalBatchSize{}_Lambda{}_maxSpeed{}_noniid{}.png'.format(
        args.round_duration, args.mu_local_train, args.beta_local_train, args.local_iter, args.local_bs, args.Lambda, args.maxSpeed, args.non_iid))
    plt.savefig(png_savePath)

    pdf_savePath = os.path.join(savePath, 'RoundDuration{}_LocalTrainDelay_mu{}_beta{}_LocalIterNum{}_LocalBatchSize{}_Lambda{}_maxSpeed{}_noniid{}.pdf'.format(
        args.round_duration, args.mu_local_train, args.beta_local_train, args.local_iter, args.local_bs, args.Lambda, args.maxSpeed, args.non_iid))
    plt.savefig(pdf_savePath)

    
    plt.close()
