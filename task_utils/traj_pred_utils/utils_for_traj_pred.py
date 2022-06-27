# Copyright (c) 2020 Uber Technologies, Inc.
# Please check LICENSE for more detail


import sys
import cv2
import os
import tempfile
import shutil
import torch
import numpy as np
import pandas as pd
import pickle as pkl
from torch import optim
from typing import Any, Dict, List, Tuple, Union
import utils.baseline_utils as baseline_utils
from utils.baseline_utils import viz_predictions
from utils.baseline_config import FEATURE_FORMAT
from torch.utils.data import DataLoader
from argoverse.map_representation.map_api import ArgoverseMap
from argoverse.evaluation.eval_forecasting import compute_forecasting_metrics

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

def get_behavior_split_dict(dataloader, sth=0.01):
    behavior_dict = {'go straight':[], 'turn left':[], 'turn right':[]}
    results = []
    for idx, data_input in enumerate(dataloader):
        results.append(judge_action_for_batch(data_input, sth=sth, prt=False)[0][0])
    for idx, result in enumerate(results):
        behavior_dict[result].append(idx)
    return behavior_dict

def get_behavior_split_dict_v2(dataloader, sth=0.01):
    behavior_dict = {'go straight':[], 'turn':[]}
    results = []
    for idx, data_input in enumerate(dataloader):
        results.append('go straight' if 'go straight' == judge_action_for_batch(data_input, sth=sth, prt=False)[0][0] else 'turn')
    for idx, result in enumerate(results):
        behavior_dict[result].append(idx)
    return behavior_dict

def index_dict(data, idcs):
    returns = dict()
    for key in data:
        returns[key] = data[key][idcs]
    return returns


def rotate(xy, theta):
    st, ct = torch.sin(theta), torch.cos(theta)
    rot_mat = xy.new().resize_(len(xy), 2, 2)
    rot_mat[:, 0, 0] = ct
    rot_mat[:, 0, 1] = -st
    rot_mat[:, 1, 0] = st
    rot_mat[:, 1, 1] = ct
    xy = torch.matmul(rot_mat, xy.unsqueeze(2)).view(len(xy), 2)
    return xy


def merge_dict(ds, dt):
    for key in ds:
        dt[key] = ds[key]
    return


def save_ckpt(net, save_dir, round):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    state_dict = net.state_dict()
    for key in state_dict.keys():
        state_dict[key] = state_dict[key].cpu()

    #save_name = "%3.0f.ckpt" % round
    save_name = "{}.ckpt".format(round)
    torch.save(
        {"epoch": round, "state_dict": state_dict},
        os.path.join(save_dir, save_name),
    )

class Logger(object):
    def __init__(self, log):
        self.terminal = sys.stdout
        self.log = open(log, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        pass


def load_pretrain(net, pretrain_dict):
    state_dict = net.state_dict()
    for key in pretrain_dict.keys():
        if key in state_dict and (pretrain_dict[key].size() == state_dict[key].size()):
            value = pretrain_dict[key]
            if not isinstance(value, torch.Tensor):
                value = value.data
            state_dict[key] = value
    net.load_state_dict(state_dict)


def gpu(data, device):
    """
    Transfer tensor in `data` to gpu recursively
    `data` can be dict, list or tuple
    """
    if isinstance(data, list) or isinstance(data, tuple):
        data = [gpu(x, device) for x in data]
    elif isinstance(data, dict):
        data = {key:gpu(_data, device) for key,_data in data.items()}
    elif isinstance(data, torch.Tensor):
        #data = data.contiguous().cuda(non_blocking=True)
        data = data.contiguous().to(device)
    return data


def to_long(data):
    if isinstance(data, dict):
        for key in data.keys():
            data[key] = to_long(data[key])
    if isinstance(data, list) or isinstance(data, tuple):
        data = [to_long(x) for x in data]
    if torch.is_tensor(data) and data.dtype == torch.int16:
        data = data.long()
    return data

class Optimizer(object):
    def __init__(self, params, config, coef=None):
        if not (isinstance(params, list) or isinstance(params, tuple)):
            params = [params]

        if coef is None:
            coef = [1.0] * len(params)
        else:
            if isinstance(coef, list) or isinstance(coef, tuple):
                assert len(coef) == len(params)
            else:
                coef = [coef] * len(params)
        self.coef = coef

        param_groups = []
        for param in params:
            param_groups.append({"params": param, "lr": 0})

        opt = config["opt"]
        assert opt == "sgd" or opt == "adam"
        if opt == "sgd":
            self.opt = optim.SGD(
                param_groups, momentum=config["momentum"], weight_decay=config["wd"]
            )
        elif opt == "adam":
            self.opt = optim.Adam(param_groups, weight_decay=0)

        self.lr_func = config["lr_func"]

        if "clip_grads" in config:
            self.clip_grads = config["clip_grads"]
            self.clip_low = config["clip_low"]
            self.clip_high = config["clip_high"]
        else:
            self.clip_grads = False

    def zero_grad(self):
        self.opt.zero_grad()

    def step(self, epoch):
        if self.clip_grads:
            self.clip()

        lr = self.lr_func(epoch)
        for i, param_group in enumerate(self.opt.param_groups):
            param_group["lr"] = lr * self.coef[i]
        self.opt.step()
        return lr

    def clip(self):
        low, high = self.clip_low, self.clip_high
        params = []
        for param_group in self.opt.param_groups:
            params += list(filter(lambda p: p.grad is not None, param_group["params"]))
        for p in params:
            mask = p.grad.data < low
            p.grad.data[mask] = low
            mask = p.grad.data > high
            p.grad.data[mask] = high

    def load_state_dict(self, opt_state):
        self.opt.load_state_dict(opt_state)


class StepLR:
    def __init__(self, lr, lr_epochs):
        assert len(lr) - len(lr_epochs) == 1
        self.lr = lr
        self.lr_epochs = lr_epochs

    def __call__(self, epoch):
        idx = 0
        for lr_epoch in self.lr_epochs:
            if epoch < lr_epoch:
                break
            idx += 1
        return self.lr[idx]




def get_city_names_from_features(features_df: pd.DataFrame) -> Dict[int, str]:
    """Get sequence id to city name mapping from the features.

    Args:
        features_df: DataFrame containing the features
    Returns:
        city_names: Dict mapping sequence id to city name

    """
    city_names = {}
    for index, row in features_df.iterrows():
        city_names[row["SEQUENCE"]] = row["FEATURES"][0][
            FEATURE_FORMAT["CITY_NAME"]]
    return city_names


def get_pruned_guesses(
        args,
        forecasted_trajectories: Dict[int, List[np.ndarray]],
        city_names: Dict[int, str],
        gt_trajectories: Dict[int, np.ndarray],
) -> Dict[int, List[np.ndarray]]:
    """Prune the number of guesses using map.

    Args:
        forecasted_trajectories: Trajectories forecasted by the algorithm.
        city_names: Dict mapping sequence id to city name.
        gt_trajectories: Ground Truth trajectories.

    Returns:
        Pruned number of forecasted trajectories.

    """
    avm = ArgoverseMap()

    pruned_guesses = {}

    for seq_id, trajectories in forecasted_trajectories.items():

        city_name = city_names[seq_id]
        da_points = []
        for trajectory in trajectories:
            raster_layer = avm.get_raster_layer_points_boolean(
                trajectory, city_name, "driveable_area")
            da_points.append(np.sum(raster_layer))

        sorted_idx = np.argsort(da_points)[::-1]
        pruned_guesses[seq_id] = [
            trajectories[i] for i in sorted_idx[:args.prune_n_guesses]
        ]

    return pruned_guesses


def get_m_trajectories_along_n_cl(
        args,
        forecasted_trajectories: Dict[int, List[np.ndarray]]
) -> Dict[int, List[np.ndarray]]:
    """Given forecasted trajectories, get <args.n_guesses_cl> trajectories along each of <args.n_cl> centerlines.

    Args:
        forecasted_trajectories: Trajectories forecasted by the algorithm.

    Returns:
        <args.n_guesses_cl> trajectories along each of <args.n_cl> centerlines.

    """
    selected_trajectories = {}
    for seq_id, trajectories in forecasted_trajectories.items():
        curr_selected_trajectories = []
        max_predictions_along_cl = min(len(forecasted_trajectories[seq_id]),
                                       args.n_cl * args.max_neighbors_cl)
        for i in range(0, max_predictions_along_cl, args.max_neighbors_cl):
            for j in range(i, i + args.n_guesses_cl):
                curr_selected_trajectories.append(
                    forecasted_trajectories[seq_id][j])
        selected_trajectories[seq_id] = curr_selected_trajectories
    return selected_trajectories


def viz_predictions_helper(
        args,
        forecasted_trajectories: Dict[int, List[np.ndarray]],
        gt_trajectories: Dict[int, np.ndarray],
        features_df: pd.DataFrame,
        viz_seq_id: Union[None, List[int]],
) -> None:
    """Visualize predictions.

    Args:
        forecasted_trajectories: Trajectories forecasted by the algorithm.
        gt_trajectories: Ground Truth trajectories.
        features_df: DataFrame containing the features
        viz_seq_id: Sequence ids to be visualized

    """
    seq_ids = gt_trajectories.keys() if viz_seq_id is None else viz_seq_id
    for seq_id in seq_ids:
        gt_trajectory = gt_trajectories[seq_id]
        curr_features_df = features_df[features_df["SEQUENCE"] == seq_id]
        input_trajectory = (
            curr_features_df["FEATURES"].values[0]
            [:args.obs_len, [FEATURE_FORMAT["X"], FEATURE_FORMAT["Y"]]].astype(
                "float"))
        output_trajectories = forecasted_trajectories[seq_id]
        candidate_centerlines = curr_features_df[
            "CANDIDATE_CENTERLINES"].values[0]
        city_name = curr_features_df["FEATURES"].values[0][
            0, FEATURE_FORMAT["CITY_NAME"]]

        gt_trajectory = np.expand_dims(gt_trajectory, 0)
        input_trajectory = np.expand_dims(input_trajectory, 0)
        output_trajectories = np.expand_dims(np.array(output_trajectories), 0)
        candidate_centerlines = np.expand_dims(np.array(candidate_centerlines),
                                               0)
        city_name = np.array([city_name])

        viz_predictions(
            input_trajectory,
            output_trajectories,
            gt_trajectory,
            candidate_centerlines,
            city_name,
            idx=seq_id,
            show=True,
        )


def evaluate(args, post_process, round):

    #TODO: generate forecasted_trajectories
    traj_save_path = os.path.join(
        args.traj_save_path, args.save_address_id+".pth.tar")
    
    """infer_helper(args, device, test_data_dict, 0, encoder, decoder,
                 model_utils, temp_save_dir)"""

    print(
        f"Round {round}: forecasted trajectories for the test set are saved at {traj_save_path}")

    # Evaluating stage
    with open(args.gt, "rb") as f:
        gt_trajectories: Dict[int, np.ndarray] = pkl.load(f)

    with open(traj_save_path, "rb") as f:
        forecasted_trajectories: Dict[int, List[np.ndarray]] = pkl.load(f)

    with open(args.test_features, "rb") as f:
        features_df: pd.DataFrame = pkl.load(f)

    metric_results = None
    if args.metrics:

        city_names = get_city_names_from_features(features_df)

        # Get displacement error and dac on multiple guesses along each centerline
        if not args.prune_n_guesses and args.n_cl:
            forecasted_trajectories = get_m_trajectories_along_n_cl(
                args,
                forecasted_trajectories)
            num_trajectories = args.n_cl * args.n_guesses_cl

        # Get displacement error and dac on pruned guesses
        elif args.prune_n_guesses:
            forecasted_trajectories = get_pruned_guesses(
                args,
                forecasted_trajectories, city_names, gt_trajectories)
            num_trajectories = args.prune_n_guesses

        # Normal case
        else:
            num_trajectories = args.max_n_guesses


        #import ipdb;ipdb.set_trace()
        """
        max(forecasted_trajectories.keys())  41146
        min(forecasted_trajectories.keys())  1
        len(forecasted_trajectories.keys())  39472
        forecasted_trajectories[1]  list
        type(forecasted_trajectories[1][0])  <class 'numpy.ndarray'>
        
        forecasted_trajectories[1][0]
        forecasted_trajectories[1][0].shape  (30, 2)

        type(gt_trajectories[7694])  <class 'numpy.ndarray'>
        gt_trajectories[7694].shape  (30, 2)

        len(city_names.keys())  39472
        
        """

        metric_results = compute_forecasting_metrics(
            forecasted_trajectories,
            gt_trajectories,
            city_names,
            num_trajectories,
            args.pred_len,
            args.miss_threshold,
        )

    if args.viz:
        id_for_viz = None
        if args.viz_seq_id:
            with open(args.viz_seq_id, "rb") as f:
                id_for_viz = pkl.load(f)
        viz_predictions_helper(args, forecasted_trajectories, gt_trajectories,
                               features_df, id_for_viz)

    return metric_results