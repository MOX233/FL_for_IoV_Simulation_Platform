#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.8

import argparse


def args_parser():
    parser = argparse.ArgumentParser()

    # task
    parser.add_argument("--task", type=str, default="traj_pred") # identify the specific task.

    # cash
    parser.add_argument("--save_id", type=str, default="default") # identify saving path. Need to change in the future
    parser.add_argument("-f", type=str, default="default")

    # v2v
    parser.add_argument('--v2v', type=int, default=0,
                        help="determine type of FLtable genaration method and type of Trainer")

    # federated arguments
    parser.add_argument('--num_items', type=int, default=1024,
                        help="number of data from every user's local dataset. type: int or list")
    parser.add_argument('--local_iter', type=float, default=20,
                        help="Local iteration num")
    parser.add_argument('--local_bs', type=int, default=64,
                        help="local batch size: B")
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
    parser.add_argument('--momentum', type=float, default=0.5,
                        help="SGD momentum (default: 0.5)")

    # model arguments
    parser.add_argument("--model",
                        type=str,
                        default="lanegcn",
                        help="DL model")
    parser.add_argument("--model_path",
                        required=False,
                        type=str,
                        help="path to the saved model")
    parser.add_argument("--obs_len",
                        default=20,
                        type=int,
                        help="Observed length of the trajectory")
    parser.add_argument("--pred_len",
                        default=30,
                        type=int,
                        help="Prediction Horizon")
    parser.add_argument("--ckpt_path", default="", type=str, metavar="CKPTPATH", help="checkpoint path")

    # evaluation arguments
    parser.add_argument('--no_eval', action='store_true',
                        default=False,  help='if no_eval, we do not carry out evaluation.')
    parser.add_argument('--simple_eval', action='store_true',
                        default=False,  help='if simple_eval, we use only a subset of the whole validation dataset as the validation dataset.')
    parser.add_argument('--simple_eval_num', type=int,
                        default=1000,  help='if simple_eval, the volume of the validation dataset is simple_eval_num.')
                        
    parser.add_argument("--metrics",
                        action="store_true",
                        help="If true, compute metrics")
    parser.add_argument("--gt", default="", type=str, help="path to gt file")
    parser.add_argument("--miss_threshold",
                        default=2.0,
                        type=float,
                        help="Threshold for miss rate")
    parser.add_argument("--max_n_guesses",
                        default=0,
                        type=int,
                        help="Max number of guesses")
    parser.add_argument("--prune_n_guesses",
                        default=0,
                        type=int,
                        help="Pruned number of guesses of non-map baseline using map",)
    parser.add_argument("--n_guesses_cl",
                        default=0,
                        type=int,
                        help="Number of guesses along each centerline",)
    parser.add_argument("--n_cl",
                        default=0,
                        type=int,
                        help="Number of centerlines to consider")
    parser.add_argument("--viz",
                        action="store_true",
                        help="If true, visualize predictions")
    parser.add_argument("--viz_seq_id",
                        default="",
                        type=str,
                        help="Sequence ids for the trajectories to be visualized",)
    parser.add_argument("--max_neighbors_cl",
                        default=3,
                        type=int,
                        help="Number of neighbors obtained for each centerline by the baseline",)

    # other arguments
    parser.add_argument('--gpu', type=int, default=0,
                        help="GPU ID, -1 for CPU")
    parser.add_argument('--dataset', type=str,
                        default='Argoverse', help="name of dataset")
    parser.add_argument('--non_iid', action='store_true',
                        default=False,  help='whether i.i.d. or not')

    #parser.add_argument('--x_split_num', type=int, default=1, help="Not realized. Split the training dataset for non-i.i.d. dataset by GPS location. x_split_num means how many splittings we take along the x axis.")
    #parser.add_argument('--y_split_num', type=int, default=1, help="Not realized. Split the training dataset for non-i.i.d. dataset by GPS location. y_split_num means how many splittings we take along the y axis.")
    parser.add_argument('--split_type', type=int, default=0, help="Specifies which splitting method we will take for non-i.i.d. dataset. 0 means city_split_dict, 1 means behavior_split_dict.")

    parser.add_argument('--city_skew', action='store_true',
                        default=False,  help='whether to apply FedAvg_weighted')
    parser.add_argument('--behavior_skew', action='store_true',
                        default=False,  help='whether to apply FedAvg_weighted')
    parser.add_argument('--skew', type=float,
                        default=0.5, help='skew weight parameter for FedAvg_weighted')
                        
    parser.add_argument('--verbose', action='store_true',
                        default=False,  help='verbose print')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed which make tests reproducible (default: 1)')
    parser.add_argument('--plot_save_path', type=str, default="default",
                        help="The save path for the plots of loss and other metrices.")
    parser.add_argument('--log_save_path', type=str, default="default",
                        help="The save path for the training log of loss and other metrices.")
    parser.add_argument('--ckpt_save_path', type=str, default="default",
                        help="The save path for the ckpt.")           
    parser.add_argument("--traj_save_path",
                        required=False,
                        type=str,
                        help="path to the pickle file where forecasted trajectories will be saved.",)

    # SUMO arguments
    parser.add_argument("--sumo_data_dir", type=str, 
                         default="./sumo_data", help="the directory where saves the necessary config files for SUMO running")
    parser.add_argument("--no_sumo_run", action="store_true",
                        default=False, help="run sumo simulation to generate tripinfo.xml")
    parser.add_argument("--trajectoryInfo_path", type=str,
                        default='./sumo_result/trajectory.csv', help="the file path where stores the trajectory infomation of cars")
    parser.add_argument("--step_length", type=float,
                        default=0.1, help="sumo sampling interval")
    parser.add_argument("--num_steps", type=int,
                        default=10000, help="number of time steps, which means how many seconds the car flow takes")
    parser.add_argument("--round_duration", type=float,
                        default=20, help="duration time of each round")
    parser.add_argument("--delay_download", type=float,
                        default=1, help="download delay")
    parser.add_argument("--delay_upload", type=float,
                        default=1, help="upload delay")

    parser.add_argument("--mu_local_train", type=float,
                        default=0.1, help="param of shift exponential distribution function for local training delay")
    parser.add_argument("--beta_local_train", type=float,
                        default=0.1, help="param of shift exponential distribution function for local training delay")

    parser.add_argument("--Lambda", type=float,
                        default=0.1, help="arrival rate of car flow")
    parser.add_argument("--accel", type=float,
                        default=10, help="accelerate of car flow")
    parser.add_argument("--decel", type=float,
                        default=20, help="decelerate of car flow")
    parser.add_argument("--sigma", type=float,
                        default=0, help="imperfection of drivers, which takes value on [0,1], with 0 meaning perfection and 1 meaning imperfection")
    parser.add_argument("--carLength", type=float,
                        default=5, help="length of cars")
    parser.add_argument("--minGap", type=float,
                        default=2.5, help="minimum interval between adjacent cars")
    parser.add_argument("--maxSpeed", type=float,
                        default=20, help="maxSpeed for cars")
    parser.add_argument("--speedFactoer_mean", type=float,
                        default=1, help="")
    parser.add_argument("--speedFactoer_dev", type=float,
                        default=0, help="")
    parser.add_argument("--speedFactoer_min", type=float,
                        default=1, help="")
    parser.add_argument("--speedFactoer_max", type=float,
                        default=1, help="")
    args = parser.parse_args()
    return args
