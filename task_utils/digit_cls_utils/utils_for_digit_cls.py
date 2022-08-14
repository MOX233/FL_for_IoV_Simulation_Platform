import os
import matplotlib  # noqa
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa
from itertools import combinations
from utils.plot_utils import min_ignore_None, max_ignore_None, plot_dashed_line_for_None_samples

def generate_split_dict(dataset, type_num):
    _split_dict = {}
    for i in range(10):
        _split_dict[i] = []
    for idx,data in enumerate(dataset):
        img, label = data
        _split_dict[label].append(idx)
    split_keys = list(combinations(_split_dict.keys(), type_num))
    
    split_dict = {}
    for C_type in split_keys:
        split_dict[str(C_type)] = merge_lists([_split_dict[type] for type in C_type])
        
    return split_dict

def merge_lists(lists):
    merged_list = []
    for l in lists:
        merged_list += l
    return merged_list


def plot_for_digit_cls_task(args, loss_train, loss_val, metrices_eval, rounds):
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
    savePath = os.path.join(savePath, str(args.save_id)+'.png')
    #debug
    #import ipdb;ipdb.set_trace()
    plt.savefig(savePath)
    plt.close()