import os
import numpy.random as s
from .sumo_utils import read_tripInfo, read_trajectoryInfo

def generate_FLtable_from_tripInfo(args):
    s.seed(args.seed)
    tripInfo = read_tripInfo(tripInfo_path=os.path.join(args.sumo_data_dir,"tripinfo.xml"))
    tripInfo_dict = {}
    for i in tripInfo:
        tripInfo_dict[i['id']] = [float(i['depart']),float(i['arrival'])]

    # params which need to add into options or add into function argument
    T = args.num_steps                      # total_training_time
    T_round = args.round_duration           # duration of a round
    MU_local_train = args.MU_local_train   # param of shift exponential distribution function for local training delay
    BETA_local_train = args.BETA_local_train   # param of shift exponential distribution function for local training delay
    delay_download = args.delay_download      # download delay
    delay_upload = args.delay_upload          # upload delay
    


    num_rounds = int(T/T_round)
    FL_table = {}
    for i in range(num_rounds):
        FL_table[i] = {}
        for k,v in tripInfo_dict.items():
            if v[0]<(i+1)*T_round and v[1]>i*T_round:
                T_tolerant = min(v[1],(i+1)*T_round)-max(v[0],i*T_round)
                t_download = delay_download
                t_upload = delay_upload
                t_local_train = float(s.exponential(BETA_local_train,(1,)))+MU_local_train
                if t_download+t_upload+t_local_train<T_tolerant:
                    FL_table[i][k] = v
    return FL_table