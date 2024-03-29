# Copyright (c) 2020 Uber Technologies, Inc.
# Please check LICENSE for more detail

import sys
import os
import tempfile
import shutil
import torch
import numpy as np
import pandas as pd
import pickle as pkl
import matplotlib.pyplot as plt

from torch import optim
from collections import OrderedDict
from typing import Any, Dict, List, Tuple, Union
from torch.utils.data import DataLoader
from argoverse.map_representation.map_api import ArgoverseMap
from argoverse.evaluation.eval_forecasting import compute_forecasting_metrics
from scipy.signal import savgol_filter
from utils.plot_utils import min_ignore_None, max_ignore_None, plot_dashed_line_for_None_samples

FEATURE_FORMAT = {
    "TIMESTAMP": 0,
    "TRACK_ID": 1,
    "OBJECT_TYPE": 2,
    "X": 3,
    "Y": 4,
    "CITY_NAME": 5,
    "MIN_DISTANCE_FRONT": 6,
    "MIN_DISTANCE_BACK": 7,
    "NUM_NEIGHBORS": 8,
    "OFFSET_FROM_CENTERLINE": 9,
    "DISTANCE_ALONG_CENTERLINE": 10,
}


def plot_for_traj_pred_task(args, loss_train, loss_val, metrices_eval, rounds):
    plt.figure(figsize=(15, 15), dpi=100)
    # loss
    plt.subplot(2, 2, 1)
    plt.plot(range(len(loss_train)), loss_train,
             color='r', marker='o', markersize=4, label='Training Loss')
    #plot_dashed_line_for_None_samples(range(len(loss_train)), loss_train, color='r')

    plt.plot(range(len(loss_val)), loss_val,
             color='b', label='Validation Loss')
    plt.xlabel('round')
    plt.ylabel('train_loss')
    plt.legend()
    # accuracy
    plt.subplot(2, 2, 2)
    plt.plot(range(len(metrices_eval)), [i["minADE"]
             for i in metrices_eval], color='r', linestyle='--', marker='*', label='minADE(K=6)')
    plt.plot(range(len(metrices_eval)), [i["minFDE"]
             for i in metrices_eval], color='g', linestyle='--', marker='*', label='minFDE(K=6)')
    plt.xlabel('round')
    plt.ylabel('minADE/minFDE')
    plt.legend()
    plt.subplot(2, 2, 4)
    plt.plot(range(len(metrices_eval)), [i["MR"]
             for i in metrices_eval], color='b', label='MR')
    plt.xlabel('round')
    plt.ylabel('MR')
    plt.legend()
    # params description
    plt.subplot(2, 2, 3)
    plt.axis([0, 11, 0, 11])
    plt.axis('off')
    fontsize = 12
    plt.text(0, 10, 'Round Num: {}'.format(rounds), fontsize=fontsize)
    plt.text(0, 9, 'Round Duration: {}'.format(
        args.round_duration), fontsize=fontsize)
    plt.text(0, 8, 'mu for Local Train Delay: {}'.format(
        args.mu_local_train), fontsize=fontsize)
    plt.text(0, 7, 'beta for Local Train Delay: {}'.format(
        args.beta_local_train), fontsize=fontsize)
    plt.text(0, 6, 'Local Iter Num: {}'.format(
        args.local_iter), fontsize=fontsize)
    plt.text(0, 5, 'Local Batch Size: {}'.format(
        args.local_bs), fontsize=fontsize)
    plt.text(0, 4, 'Learning Rate: {}'.format(args.lr), fontsize=fontsize)
    plt.text(0, 3, 'non-i.i.d.: {}'.format(args.non_iid), fontsize=fontsize)
    if min_ignore_None(loss_train)!=None:
        plt.text(0, 1, 'min_train_loss: {:.6f}'.format(
            min_ignore_None(loss_train)), fontsize=fontsize)
    else:
        plt.text(0, 1, 'min_train_loss: {}'.format(
            min_ignore_None(loss_train)), fontsize=fontsize)
    plt.text(0, 0, 'min_val_loss: {:.6f}'.format(min_ignore_None(loss_val)), fontsize=fontsize)

    plt.text(6, 10, 'Lambda: {}'.format(args.Lambda), fontsize=fontsize)
    plt.text(6, 9, 'maxSpeed: {}'.format(args.maxSpeed), fontsize=fontsize)
    plt.text(6, 8, 'delay_download: {}'.format(
        args.delay_download), fontsize=fontsize)
    plt.text(6, 7, 'delay_upload: {}'.format(args.delay_upload), fontsize=fontsize)

    plt.text(6, 5, 'min_minADE: {:.6f}'.format(
        min_ignore_None([i["minADE"] for i in metrices_eval])), fontsize=fontsize)
    plt.text(6, 4, 'min_minADE1: {:.6f}'.format(
        min_ignore_None([i["minADE1"] for i in metrices_eval])), fontsize=fontsize)
    plt.text(6, 3, 'min_minFDE: {:.6f}'.format(
        min_ignore_None([i["minFDE"] for i in metrices_eval])), fontsize=fontsize)
    plt.text(6, 2, 'min_minFDE1: {:.6f}'.format(
        min_ignore_None([i["minFDE1"] for i in metrices_eval])), fontsize=fontsize)
    plt.text(6, 1, 'min_MR: {:.6f}'.format(
        min_ignore_None([i["MR"] for i in metrices_eval])), fontsize=fontsize)
    plt.text(6, 0, 'min_MR1: {:.6f}'.format(
        min_ignore_None([i["MR1"] for i in metrices_eval])), fontsize=fontsize)

    savePath = "./save"
    if args.plot_save_path != "default":
        savePath = args.plot_save_path
    os.makedirs(savePath, exist_ok=True)
    savePath = os.path.join(savePath, str(args.save_id)+'.png')
    #debug
    #import ipdb;ipdb.set_trace()
    plt.savefig(savePath)
    plt.close()


def viz_predictions(
        input_: np.ndarray,
        output: np.ndarray,
        target: np.ndarray,
        centerlines: np.ndarray,
        city_names: np.ndarray,
        idx=None,
        show: bool = True,
) -> None:
    """Visualize predicted trjectories.

    Args:
        input_ (numpy array): Input Trajectory with shape (num_tracks x obs_len x 2)
        output (numpy array of list): Top-k predicted trajectories, each with shape (num_tracks x pred_len x 2)
        target (numpy array): Ground Truth Trajectory with shape (num_tracks x pred_len x 2)
        centerlines (numpy array of list of centerlines): Centerlines (Oracle/Top-k) for each trajectory
        city_names (numpy array): city names for each trajectory
        show (bool): if True, show

    """
    num_tracks = input_.shape[0]
    obs_len = input_.shape[1]
    pred_len = target.shape[1]

    plt.figure(0, figsize=(8, 7))
    #plt.figure(idx, figsize=(8, 7))
    avm = ArgoverseMap()
    for i in range(num_tracks):
        plt.plot(
            input_[i, :, 0],
            input_[i, :, 1],
            color="#ECA154",
            label="Observed",
            alpha=1,
            linewidth=3,
            zorder=15,
        )
        plt.plot(
            input_[i, -1, 0],
            input_[i, -1, 1],
            "o",
            color="#ECA154",
            label="Observed",
            alpha=1,
            linewidth=3,
            zorder=15,
            markersize=9,
        )
        plt.plot(
            target[i, :, 0],
            target[i, :, 1],
            color="#d33e4c",
            label="Target",
            alpha=1,
            linewidth=3,
            zorder=20,
        )
        plt.plot(
            target[i, -1, 0],
            target[i, -1, 1],
            "o",
            color="#d33e4c",#red
            label="Target",
            alpha=1,
            linewidth=3,
            zorder=20,
            markersize=9,
        )

        for j in range(len(centerlines[i])):
            plt.plot(
                centerlines[i][j][:, 0],
                centerlines[i][j][:, 1],
                "--",
                color="grey",
                alpha=1,
                linewidth=1,
                zorder=0,
            )

        for j in range(len(output[i])):
            plt.plot(
                output[i][j][:, 0],
                output[i][j][:, 1],
                color="#007672",#green
                label="Predicted",
                alpha=1,
                linewidth=3,
                zorder=15,
            )
            plt.plot(
                output[i][j][-1, 0],
                output[i][j][-1, 1],
                "o",
                color="#007672",
                label="Predicted",
                alpha=1,
                linewidth=3,
                zorder=15,
                markersize=9,
            )
            for k in range(pred_len):
                lane_ids = avm.get_lane_ids_in_xy_bbox(
                    output[i][j][k, 0],
                    output[i][j][k, 1],
                    city_names[i],
                    query_search_range_manhattan=2.5,
                )

        for j in range(obs_len):
            lane_ids = avm.get_lane_ids_in_xy_bbox(
                input_[i, j, 0],
                input_[i, j, 1],
                city_names[i],
                query_search_range_manhattan=2.5,
            )
            [avm.draw_lane(lane_id, city_names[i]) for lane_id in lane_ids]
        for j in range(pred_len):
            lane_ids = avm.get_lane_ids_in_xy_bbox(
                target[i, j, 0],
                target[i, j, 1],
                city_names[i],
                query_search_range_manhattan=2.5,
            )
            [avm.draw_lane(lane_id, city_names[i]) for lane_id in lane_ids]

        plt.axis("equal")
        plt.xticks([])
        plt.yticks([])
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        if show:
            print('visualize '+str(idx))
            os.makedirs('./viz_trajectories', exist_ok=True)
            plt.savefig('./viz_trajectories/'+str(idx)+'.png')
            plt.clf()

def smooth(traj):
    #traj (30,2)
    window_length = 15
    polyorder = 2
    length = len(traj)
    x_sm = savgol_filter(traj[:,0], window_length, polyorder, mode= 'nearest')
    y_sm = savgol_filter(traj[:,1], window_length, polyorder, mode= 'nearest')
    traj_sm = np.zeros_like(traj)
    traj_sm[:,0] = x_sm
    traj_sm[:,1] = y_sm
    return traj_sm

def judge_action(traj, sth=0.1):
    
    #traj shape (50,2)
    num_step = len(traj)
    traj = np.squeeze(traj)
    traj = smooth(traj)
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