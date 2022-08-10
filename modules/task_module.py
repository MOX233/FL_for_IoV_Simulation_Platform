
import os
import torch
from task_interface.traj_pred import get_traj_pred_task
from task_interface.digit_cls import get_digit_cls_task

def get_task(args):
    if args.task == "traj_pred":
        dataset_train, net, generate_split_dict_for_task, evaluator_for_task, trainer_for_task, plot_func_for_task =  get_traj_pred_task(args)
    elif args.task == "digit_cls":
        dataset_train, net, generate_split_dict_for_task, evaluator_for_task, trainer_for_task, plot_func_for_task =  get_digit_cls_task(args)
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

    return dataset_train, net, generate_split_dict_for_task, evaluator_for_task, trainer_for_task, plot_func_for_task

def load_pretrain(net, pretrain_dict):
    state_dict = net.state_dict()
    for key in pretrain_dict.keys():
        if key in state_dict and (pretrain_dict[key].size() == state_dict[key].size()):
            value = pretrain_dict[key]
            if not isinstance(value, torch.Tensor):
                value = value.data
            state_dict[key] = value
    net.load_state_dict(state_dict)