
import os
import torch
from task_interface.traj_pred import get_net_for_traj_pred, get_dataset_for_traj_pred, generate_split_dict_for_traj_pred, Evaluator_for_traj_pred, Trainer_for_traj_pred

def get_task(args):
    if args.task == "traj_pred":
        dataset_train, net, generate_split_dict_for_task, evaluator_for_task, trainer_for_task =  get_traj_pred_task(args)
    else:
        exit("Unrecognized task")

    #load trained model
    if args.ckpt_path:
        ckpt_path = args.ckpt_path
        if not os.path.isabs(ckpt_path):
            file_path = os.path.abspath(__file__)
            root_path = os.path.dirname(file_path) ## TODO:may cause bug
            root_path = os.path.dirname(root_path)
            ckpt_path = os.path.join(root_path, "ckpt", args.task, ckpt_path)
        print("load ckpt from: "+ckpt_path)
        ckpt = torch.load(ckpt_path, map_location=args.device)
        load_pretrain(net, ckpt["state_dict"])

    return dataset_train, net, generate_split_dict_for_task, evaluator_for_task, trainer_for_task

def load_pretrain(net, pretrain_dict):
    state_dict = net.state_dict()
    for key in pretrain_dict.keys():
        if key in state_dict and (pretrain_dict[key].size() == state_dict[key].size()):
            value = pretrain_dict[key]
            if not isinstance(value, torch.Tensor):
                value = value.data
            state_dict[key] = value
    net.load_state_dict(state_dict)

def get_traj_pred_task(args):
    net = get_net_for_traj_pred(args)

    ##### get dataset #####
    dataset_train, dataset_val = get_dataset_for_traj_pred(args)
    evaluator_for_traj_pred = Evaluator_for_traj_pred(args, dataset_val)
    trainer_for_traj_pred = Trainer_for_traj_pred(args, dataset_train)
    return dataset_train, net, generate_split_dict_for_traj_pred, evaluator_for_traj_pred, trainer_for_traj_pred
