
from __future__ import absolute_import
from __future__ import print_function

import os
import torch
import random
import numpy as np

from utils.options import args_parser
from modules import platform

def sim_run(args):
    args.save_id = '{}_T{}_H{}_Lbd{}_mu{}_beta{}_v{}_lr{}_bs{}_noniid{}_type{}'.format(args.task, args.round_duration, args.local_iter, args.Lambda, args.mu_local_train, args.beta_local_train, args.maxSpeed, args.lr, args.local_bs, int(args.non_iid), args.split_type)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    print(f"Using device ({args.device}) ...")
    platform.run_platform(args)

if __name__ == "__main__":
    args = args_parser()

    args.task = 'digit_cls'
    args.num_steps = 3200 * 20
    args.num_items = 2048
    args.local_bs = 64
    args.lr = 0.1

    args.gpu = 7
    
    args.non_iid = True
    args.split_type = 1
    args.round_duration = 20
    args.local_iter = 20
    args.mu_local_train = 0.1
    args.beta_local_train = 0.1
    args.plot_save_path = './save/digit_cls/plot'
    args.log_save_path = './save/digit_cls/log'
    #args.ckpt_path = './save/digit_cls/ckpt'

    for local_iter in [1]:
        args.local_iter = local_iter
        sim_run(args)
