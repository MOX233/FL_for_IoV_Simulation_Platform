import torch
from torch.utils.data import DataLoader, Dataset
import time
import numpy as np
import copy
from utils.FL_utils import FedAvg, DatasetSplit, Trainer_abc, Evaluator_abc
from task_utils.traj_pred_utils.lanegcn import get_model
from task_utils.traj_pred_utils.update import LocalUpdate_for_traj_pred
from task_utils.traj_pred_utils.FedAvg_for_traj_pred import FedAvg_city_weighted, FedAvg_behavior_weighted
from task_utils.traj_pred_utils.utils_for_traj_pred import generate_city_split_dict, generate_behavior_split_dict, plot_for_traj_pred_task

def get_traj_pred_task(args):
    net = get_net_for_traj_pred(args)
    dataset_train, dataset_val = get_dataset_for_traj_pred(args)
    evaluator_for_traj_pred = Evaluator_for_traj_pred(args, dataset_val)
    trainer_for_traj_pred = Trainer_for_traj_pred(args, dataset_train)
    plot_func_for_traj_pred = plot_for_traj_pred_task
    return dataset_train, net, generate_split_dict_for_traj_pred, evaluator_for_traj_pred, trainer_for_traj_pred, plot_func_for_traj_pred

def get_net_for_traj_pred(args):
    config, Dataset, collate_fn, net, Loss, post_process = get_model(args)
    return net


def get_dataset_for_traj_pred(args):
    config, Dataset, collate_fn, net, Loss, post_process = get_model(args)
    print("Loading dataset")
    start_time = time.time()
    dataset_train = Dataset(config, train=True)
    dataset_val = Dataset(config, train=False)
    end_time = time.time()
    print("Complete dataset loading with running time {:.3f}s".format(end_time-start_time))
    #print('dataset size',len(dataset_train),len(dataset_val))
    if args.simple_eval:
        simple_val_idxs = np.random.choice(list(range(len(dataset_val))),args.simple_eval_num,replace=False)
        dataset_val = DatasetSplit(dataset_val,simple_val_idxs)
    return dataset_train, dataset_val


###################################################
################ data allocator ###################

def generate_split_dict_for_traj_pred(args, dataset):
    if args.split_type == 0:
        split_dict = generate_city_split_dict(dataset)        # non-i.i.d. dataset by city name
    elif args.split_type == 1:
        split_dict = generate_behavior_split_dict(dataset)    # non-i.i.d. dataset by car behavior
    else:
        exit('Error: unrecognized type of non i.i.d. split dict')
    return split_dict


################ data allocator ###################
###################################################


###################################################
################     trainer    ###################
class Trainer_for_traj_pred(Trainer_abc):
    # local_iter and local_batch_size can be given flexibly
    def __init__(self, args, dataset_train):
        super().__init__(args, dataset_train)
        #self.args = args
        #self.dataset_train = dataset_train
        self.config, _, _, _, _, _ = get_model(args)

    def train_a_round(self, idxs_list, net_glob):
        loss_locals = []
        w_locals = []
        city_locals = []
        behavior_locals = []
        for idxs in idxs_list:
            local = LocalUpdate_for_traj_pred(args=self.args, dataset=self.dataset_train, idxs=idxs, local_bs=self.args.local_bs)
            w, loss, _city, _behavior = local.train(net=copy.deepcopy(net_glob), config=self.config, local_iter=self.args.local_iter)
            w_locals.append(copy.deepcopy(w))

            city_locals.append(_city)
            behavior_locals.append(_behavior[0][0])

            loss_locals.append(copy.deepcopy(loss))
        # update global weights
        if self.args.city_skew and self.args.non_iid:
            w_glob = FedAvg_city_weighted(w_locals, city_locals, skew=self.args.skew)
        elif self.args.behavior_skew and self.args.non_iid:
            w_glob = FedAvg_behavior_weighted(w_locals, behavior_locals, skew=self.args.skew)
        else:
            w_glob = FedAvg(w_locals)

        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)

        # print loss
        train_loss = sum(loss_locals) / len(loss_locals)
        return train_loss

################     trainer    ###################
###################################################


###################################################
################   evaluation   ###################

class Evaluator_for_traj_pred(Evaluator_abc):
    def __init__(self, args, dataset_val) -> None:
        super().__init__(args, dataset_val)
        _, _, collate_fn, _, self.Loss, self.post_process = get_model(args)
        self.val_loader = DataLoader(
            dataset_val,
            #batch_size=args.val_batch_size, # To be added
            batch_size=1,
            shuffle=True,
            collate_fn=collate_fn,
            pin_memory=True,
        )


    def eval_a_round(self, net, round):
        start_time = time.time()
        metrics = dict()
        for i, data in enumerate(self.val_loader):
            data = dict(data)
            with torch.no_grad():
                output = net(data)
                loss_out = self.Loss(output, data)
                post_out = self.post_process(output, data)
                self.post_process.append(metrics, loss_out, post_out)
        dt = time.time() - start_time
        loss, cls, reg, ade1, fde1, mr1, ade, fde, mr = self.post_process.display(metrics, dt, round)
        eval_metrices = {"minADE": ade, "minFDE": fde, "MR": mr, "minADE1": ade1, "minFDE1": fde1, "MR1": mr1}
        return loss, eval_metrices

################   evaluation   ###################
###################################################