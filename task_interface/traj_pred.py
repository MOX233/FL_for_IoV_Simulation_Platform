import torch
from torch.utils.data import DataLoader, Dataset
import time
import numpy as np
#from torch.utils.data.dataloader import DataLoader
import os
import copy
from importlib import import_module
from task_utils.traj_pred_utils.update import LocalUpdate,DatasetSplit
from utils.federate_learning_avg import FedAvg, FedAvg_city_weighted, FedAvg_behavior_weighted


def get_net_for_traj_pred(args):
    model = import_module('lanegcn')
    config, Dataset, collate_fn, net, Loss, post_process = model.get_model(args)
    return net

def get_dataset_for_traj_pred(args):
    model = import_module('lanegcn')
    config, Dataset, collate_fn, net, Loss, post_process = model.get_model(args)
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


def generate_city_split_dict(dataset):
    city_split_dict = {'MIA':[], 'PIT':[]}
    for idx,data in enumerate(dataset.split):
        city_split_dict[data['city']].append(idx)
    return city_split_dict


def generate_behavior_split_dict(dataset, sth=0.01):
    behavior_split_dict = {'go straight':[], 'turn':[]}
    results = []
    bhv_train_ldr = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn
    )
    for idx, data_input in enumerate(bhv_train_ldr):
        results.append('go straight' if 'go straight' == judge_action_for_batch(data_input, sth=sth, prt=False)[0][0] else 'turn')
    for idx, result in enumerate(results):
        behavior_split_dict[result].append(idx)
    return behavior_split_dict

def judge_action_for_batch(data_input, sth=0.01, prt=False):
    results = []
    for idx in range(len(data_input['feats'])):
        _feats = data_input['feats'][idx][0,...,:2]
        obs_feats = torch.zeros_like(_feats)
        obs_feats[0] = _feats[0]
        for t in range(1, len(_feats)):
            obs_feats[t] = obs_feats[t-1] + _feats[t]
        obs_feats -= obs_feats[-1]-obs_feats[0]
        obs = torch.matmul(obs_feats, data_input['rot'][idx]) + data_input['orig'][idx].view(1,-1) 
        obs = obs.numpy()
        gt = [x[0:1].numpy() for x in data_input["gt_preds"]][idx][0,...]
        traj = np.concatenate((obs,gt), axis=0)
        result = [judge_action(traj, sth),data_input['city'][idx]]
        results.append(result)
        if prt:
            print(idx,result)
    return results

def judge_action(traj, sth=0.1):
    
    #traj shape (50,2)
    num_step = len(traj)
    traj = np.squeeze(traj)
    eps = 1e-8
    traj_norm = traj - traj[0,:]
    xt,yt = traj_norm[-1,0], traj_norm[-1,1]
    D = np.sqrt(xt**2+yt**2)
    a = -xt/(yt+eps)
    get_dist = lambda pos: np.abs((pos[0]+a*pos[1])/np.sqrt(1+a**2))
    left, right = 0,0
    for pos in traj_norm:
        if (pos[0]+a*pos[1])*yt>0:
            right += get_dist(pos)
        elif (pos[0]+a*pos[1])*yt<0:
            left += get_dist(pos)
    feat = (right-left)/D/num_step
    #print("k,left,right",-1/a,left,right)
    #print(feat)
    if feat > sth:
        return "turn left"
    elif feat < -sth:
        return "turn right"
    else:
        return "go straight"

def collate_fn(batch):
    batch = from_numpy(batch)
    return_batch = dict()
    # Batching by use a list for non-fixed size
    for key in batch[0].keys():
        return_batch[key] = [x[key] for x in batch]
    return return_batch


def from_numpy(data):
    """Recursively transform numpy.ndarray to torch.Tensor.
    """
    if isinstance(data, dict):
        for key in data.keys():
            data[key] = from_numpy(data[key])
    if isinstance(data, list) or isinstance(data, tuple):
        data = [from_numpy(x) for x in data]
    if isinstance(data, np.ndarray):
        """Pytorch now has bool type."""
        data = torch.from_numpy(data)
    return data


################ data allocator ###################
###################################################


###################################################
################     trainer    ###################

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        #import ipdb;ipdb.set_trace()
        data = self.dataset[self.idxs[item]]
        return data

class Trainer_for_traj_pred(object):
    # local_iter and local_batch_size can be given flexibly
    def __init__(self, args, dataset_train):
        self.args = args
        self.dataset_train = dataset_train
        model = import_module('lanegcn')
        self.config, _, _, _, _, _ = model.get_model(args)

    def train_a_round(self, idxs_list, net_glob):
        loss_locals = []
        w_locals = []
        city_locals = []
        behavior_locals = []
        for idxs in idxs_list:
            local = LocalUpdate(args=self.args, dataset=self.dataset_train, idxs=idxs, local_bs=self.args.local_bs)
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

class Evaluator_for_traj_pred():
    def __init__(self, args, dataset_val) -> None:
        model = import_module('lanegcn')
        _, _, collate_fn, _, self.Loss, self.post_process = model.get_model(args)

        self.val_loader = DataLoader(
            dataset_val,
            #batch_size=args.val_batch_size, # To be added
            batch_size=1,
            shuffle=True,
            collate_fn=collate_fn,
            pin_memory=True,
        )



    def eval_for_traj_pred(self, net, round):
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