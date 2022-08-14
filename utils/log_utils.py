#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.8
import os
import torch
import pickle as pkl

def save_ckpt(net, save_dir, round):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    state_dict = net.state_dict()
    for key in state_dict.keys():
        state_dict[key] = state_dict[key].cpu()

    #save_name = "%3.0f.ckpt" % round
    save_name = "{}.ckpt".format(round)
    torch.save(
        {"epoch": round, "state_dict": state_dict},
        os.path.join(save_dir, save_name),
    )

def save_training_log(args, train_loss_list, val_loss_list, eval_metrices_list):

    log_dict = {}
    log_dict['loss_train'] = train_loss_list
    log_dict['loss_val'] = val_loss_list
    log_dict['metrices_eval'] = eval_metrices_list
    log_dict.update(vars(args))
    savePath = "./save"
    if args.log_save_path != "default":
            savePath = args.log_save_path
    os.makedirs(savePath, exist_ok=True)
    savePath = os.path.join(savePath, str(args.save_id)+'.pkl')
    with open(savePath,'wb') as f:
        pkl.dump(log_dict, f)

def load_training_log(savePath):
    with open(savePath,'rb') as f:
        log_dict = pkl.load(f)
    return log_dict