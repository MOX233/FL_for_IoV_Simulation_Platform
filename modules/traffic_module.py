import os
from utils.sumo_utils import read_tripInfo, sumo_run

def get_tripinfo(args):
    args.MU_local_train = args.local_iter * args.mu_local_train
    args.BETA_local_train = args.local_iter * args.beta_local_train

    if args.no_sumo_run == False:
        sumo_run(args, save_dir=args.sumo_data_dir)
    car_tripinfo = read_tripInfo(tripInfo_path=os.path.join(args.sumo_data_dir,'tripinfo.xml'))
    return car_tripinfo