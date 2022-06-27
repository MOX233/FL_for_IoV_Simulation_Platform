#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.8

import copy
import random
import numpy as np
import torch
import sys
import time
import tempfile
import shutil
import os
import torch.nn as nn
import pickle as pkl
import pandas as pd
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
sys.path.append("..") 

from torch.utils.data.dataloader import DataLoader
from utils.options import args_parser
from utils.sampling import sample_iid, sample_noniid
#from utils.lstm_utils import ModelUtils, LSTMDataset, EncoderRNN, DecoderRNN, train, validate, evaluate, infer_helper, get_city_names_from_features, get_m_trajectories_along_n_cl, get_pruned_guesses, viz_predictions_helper
from task_utils.traj_pred_utils.update import LocalUpdate,DatasetSplit    # need to rewrite
from utils.federate_learning_avg import FedAvg, FedAvg_city_weighted, FedAvg_behavior_weighted
from utils.plot_utils import min_ignore_None, plot_loss_acc_curve
from utils.log_utils import save_training_log
from typing import Any, Dict, List, Tuple, Union
from argoverse.evaluation.eval_forecasting import compute_forecasting_metrics
from argoverse.data_loading.argoverse_forecasting_loader import ArgoverseForecastingLoader
from utils.logger import Logger

from importlib import import_module


from task_utils.traj_pred_utils.utils_for_traj_pred import Logger, load_pretrain, save_ckpt, evaluate, get_behavior_split_dict, get_behavior_split_dict_v2

def FL_training(args,FL_table,car_tripinfo):
    # parse args

    # fix random_seed, so that the experiment can be repeat
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    print(f"Using device ({args.device}) ...")
    #model_utils = ModelUtils()

    # build model
    if args.model in ['lanegcn', 'LaneGCN']:
        model = import_module(args.model)
        config, Dataset, collate_fn, net, Loss, post_process = model.get_model(args)

        #load trained model
        if args.resume or args.weight:
            ckpt_path = args.resume or args.weight
            if not os.path.isabs(ckpt_path):
                ckpt_path = os.path.join(config["save_dir"], ckpt_path)
            ckpt = torch.load(ckpt_path, map_location=args.device)
            load_pretrain(net, ckpt["state_dict"])
            if args.resume:
                config["epoch"] = ckpt["epoch"]
        net_glob = copy.deepcopy(net)
        #print(net_glob)
    else:
        exit('Error: unrecognized model')

    # load dataset and split users
    num_users = len(car_tripinfo)
    if args.dataset == 'Argoverse':
        # Get PyTorch Dataset
        print("Loading dataset")
        start_time = time.time()
        dataset_train = Dataset(args.train_features, config, train=True)
        dataset_val = Dataset(args.val_features, config, train=False)
        #print('dataset size',len(dataset_train),len(dataset_val))
        #import ipdb;ipdb.set_trace()
        if args.simple_eval:
            simple_val_idxs = np.random.choice(list(range(len(dataset_val))),args.simple_eval_num,replace=False)
            dataset_val = DatasetSplit(dataset_val,simple_val_idxs)

        if not args.non_iid:
            dict_users = sample_iid(dataset_train, args.num_items, num_users)
        else:
            if args.split_dict == 0:
                # non-i.i.d. dataset by city name
                city_split_dict = {'MIA':[], 'PIT':[]}
                for idx,data in enumerate(dataset_train.split):
                    city_split_dict[data['city']].append(idx)
                dict_users = sample_noniid(dataset_train, city_split_dict, args.num_items, num_users)
            elif args.split_dict == 1:
                # non-i.i.d. dataset by car behavior
                bhv_train_ldr = DataLoader(
                    dataset_train,
                    batch_size=1,
                    shuffle=False,
                    collate_fn=collate_fn
                )
                """behavior_split_dict = get_behavior_split_dict(bhv_train_ldr, sth=0.01)
                print(len(behavior_split_dict['go straight']), len(behavior_split_dict['turn left']), len(behavior_split_dict['turn right']))"""
                behavior_split_dict = get_behavior_split_dict_v2(bhv_train_ldr, sth=0.01)
                print(len(behavior_split_dict['go straight']), len(behavior_split_dict['turn']))

                dict_users = sample_noniid(dataset_train, behavior_split_dict, args.num_items, num_users)
            else:
                exit('Error: unrecognized type of non i.i.d. split dict')

        end_time = time.time()
        print("Complete dataset loading with running time {:.3f}s".format(end_time-start_time))

    else:
        exit('Error: unrecognized dataset')


    # Create log and copy all code
    save_dir = config["save_dir"]
    log_save_dir = os.path.join(save_dir, "log")
    ckpt_save_dir = os.path.join(save_dir, "ckpt")
    log_save_path = os.path.join(log_save_dir, args.save_address_id)
    ckpt_save_path = os.path.join(ckpt_save_dir, args.save_address_id)
    if not os.path.exists(log_save_dir):
        os.makedirs(log_save_dir)
    if not os.path.exists(ckpt_save_dir):
        os.makedirs(ckpt_save_dir)
    sys.stdout = Logger(log_save_path)

    net_glob.train()
    # copy weights
    w_glob = net_glob.state_dict()

    # training
    train_loss_list = []
    val_loss_list = []
    eval_metrices_list = []
    #net_best = net_glob
    rounds = len(FL_table.keys())
    for round in range(rounds):
        loss_locals = []
        w_locals = []
        city_locals = []
        behavior_locals = []
        idxs_users = [int(car.split('_')[-1]) for car in FL_table[round].keys()]
        print("Round {:3d}, Car num {}, Training start".format(round, len(idxs_users)))
        if idxs_users == []:

            # print loss
            loss_avg = train_loss_list[-1] if round>0 else None
            if loss_avg==None:
                print('Round {:3d}, No Car, Average Training Loss None'.format(round))
            else:
                print('Round {:3d}, No Car, Average Training Loss {:.5f}'.format(round, loss_avg))
            train_loss_list.append(loss_avg)
    
            # validation part
            if args.no_eval == False:
                round_val_loss = val_loss_list[-1] if round>0 else None
                metric_results = eval_metrices_list[-1] if round>0 else {"minADE": None, "minFDE": None, "MR": None, "minADE1": None, "minFDE1": None, "MR1": None, "DAC": None}
                val_loss_list.append(round_val_loss)
                eval_metrices_list.append(metric_results)
                print("Validation Metrices: {}".format(metric_results))
            continue
            
        else:
            for idx in idxs_users:
                local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx], local_bs=args.local_bs)
                print("localUpdate start for user {}".format(idx))
                w, loss, _city, _behavior = local.train(net=copy.deepcopy(net_glob), config=config, local_iter=args.local_iter)
                w_locals.append(copy.deepcopy(w))

                city_locals.append(_city)
                behavior_locals.append(_behavior[0][0])

                loss_locals.append(copy.deepcopy(loss))
            # update global weights
            if args.city_skew and args.non_iid:
                w_glob = FedAvg_city_weighted(w_locals, city_locals, skew=args.skew)
            elif args.behavior_skew and args.non_iid:
                w_glob = FedAvg_behavior_weighted(w_locals, behavior_locals, skew=args.skew)
            else:
                w_glob = FedAvg(w_locals)
    
            # copy weight to net_glob
            net_glob.load_state_dict(w_glob)
    
            # print loss
            loss_avg = sum(loss_locals) / len(loss_locals)
            print('Round {:3d}, Car num: {:3d}, Average Training Loss {:.5f}'.format(round, len(idxs_users), loss_avg))
            train_loss_list.append(loss_avg)

            # validation part
            #metric_results, iter_val_loss = test_beam_select(net_glob, dataset_val, args)
            if args.no_eval == False:
                print('build val_loader')
                val_loader = DataLoader(
                    dataset_val,
                    batch_size=config["val_batch_size"],
                    shuffle=True,
                    collate_fn=collate_fn,
                    pin_memory=True,
                )
                print('val begin')
                round_val_loss, _cls, _reg, ade1, fde1, mr1, ade, fde, mr = val(args, val_loader, net_glob, Loss, post_process, round)

                val_loss_list.append(round_val_loss)
                metric_results = {"round":round, "minADE": ade, "minFDE": fde, "MR": mr, "minADE1": ade1, "minFDE1": fde1, "MR1": mr1, "DAC": None}
                eval_metrices_list.append(metric_results)
        plot_loss_acc_curve(args, train_loss_list, val_loss_list, eval_metrices_list, rounds)
        save_training_log(args, train_loss_list, val_loss_list, eval_metrices_list)

    # save checkpoint
    print('save ckpt')
    save_ckpt(net_glob, ckpt_save_path, round)
    # test part
    net_glob.eval()


def val(args, data_loader, net, loss, post_process, epoch):
    net.eval()

    start_time = time.time()
    metrics = dict()
    for i, data in enumerate(data_loader):
        data = dict(data)
        #print('\r val progress: {}/{}'.format(i,len(data_loader)), end="")
        with torch.no_grad():
            output = net(data)
            loss_out = loss(output, data)
            post_out = post_process(output, data)
            post_process.append(metrics, loss_out, post_out)

    dt = time.time() - start_time
    loss, cls, reg, ade1, fde1, mr1, ade, fde, mr = post_process.display(metrics, dt, epoch)

    net.train()
    return loss, cls, reg, ade1, fde1, mr1, ade, fde, mr 