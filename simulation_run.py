
from __future__ import absolute_import
from __future__ import print_function

import os
import torch
import random
import numpy as np

from utils.options import args_parser
from modules import platform

# this is the main entry point of this script
if __name__ == "__main__":
    args = args_parser()

    file_path = os.path.abspath(__file__)
    root_path = os.path.dirname(file_path)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    print(f"Using device ({args.device}) ...")
    if args.save_id == "default":
        args.save_id = 'RoundDuration{}_LocalTrainDelay_mu{}_beta{}_LocalIterNum{}_LocalBatchSize{}_Lambda{}_maxSpeed{}_noniid{}'.format(args.round_duration, args.mu_local_train, args.beta_local_train, args.local_iter, args.local_bs, args.Lambda, args.maxSpeed, int(args.non_iid))
    
    platform.run_platform(args)