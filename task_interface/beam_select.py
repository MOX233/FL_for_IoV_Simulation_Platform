import torch
from torch.utils.data import DataLoader, Dataset
import time
import numpy as np
import copy
import torch.nn.functional as F
from ignite.metrics import TopKCategoricalAccuracy
from utils.FL_utils import FedAvg, DatasetSplit, Trainer_abc, Evaluator_abc
from task_utils.beam_select_utils.nets import Lidar2D
from task_utils.beam_select_utils.dataloader import LidarDataset2D
from task_utils.beam_select_utils.update import LocalUpdate_for_beam_select
from task_utils.beam_select_utils.utils_for_beam_select import plot_for_beam_select_task


def get_beam_select_task(args):
    net = get_net_for_beam_select(args)
    dataset_train, dataset_val = get_dataset_for_beam_select(args)
    evaluator_for_beam_select = Evaluator_for_beam_select(args, dataset_val)
    if args.v2v == 0:
        trainer_for_beam_select = Trainer_for_beam_select(args, dataset_train)
    elif args.v2v == -1:
        trainer_for_beam_select = Trainer_for_beam_select(args, dataset_train)
    else:
        trainer_for_beam_select = Trainer_for_beam_select_with_V2V(args, dataset_train)
    plot_func_for_beam_select = plot_for_beam_select_task
    return dataset_train, net, generate_split_dict_for_beam_select, evaluator_for_beam_select, trainer_for_beam_select, plot_func_for_beam_select

def get_net_for_beam_select(args):
    return Lidar2D(args).to(args.device)


def get_dataset_for_beam_select(args):
    print("Loading dataset")
    start_time = time.time()
    args.lidar_training_data = '/home/ubuntu/Raymobtime_Dataset/Raymobtime_s008/baseline_data/lidar_input/lidar_train.npz'
    args.beam_training_data = '/home/ubuntu/Raymobtime_Dataset/Raymobtime_s008/baseline_data/beam_output/beams_output_train.npz'
    args.lidar_validation_data = '/home/ubuntu/Raymobtime_Dataset/Raymobtime_s008/baseline_data/lidar_input/lidar_validation.npz'
    args.beam_validation_data = '/home/ubuntu/Raymobtime_Dataset/Raymobtime_s008/baseline_data/beam_output/beams_output_validation.npz'
    dataset_train = LidarDataset2D(lidar_data_path=args.lidar_training_data, beam_data_path=args.beam_training_data)
    dataset_val = LidarDataset2D(lidar_data_path=args.lidar_validation_data, beam_data_path=args.beam_validation_data)
    end_time = time.time()
    print("Complete dataset loading with running time {:.3f}s".format(end_time-start_time))
    
    return dataset_train, dataset_val


###################################################
################ data allocator ###################

def generate_split_dict_for_beam_select(args, dataset):
    #TODO
    return None


################ data allocator ###################
###################################################


###################################################
################     trainer    ###################
class Trainer_for_beam_select(Trainer_abc):
    # local_iter and local_batch_size can be given flexibly
    def __init__(self, args, dataset_train):
        super().__init__(args, dataset_train)
        #self.args = args
        #self.dataset_train = dataset_train

    def train_a_round(self, idxs_list, net_glob):
        loss_locals = []
        w_locals = []
        for idxs in idxs_list:
            local = LocalUpdate_for_beam_select(args=self.args, dataset=self.dataset_train, idxs=idxs, local_bs=self.args.local_bs)
            w, loss = local.train(net=copy.deepcopy(net_glob).to(self.args.device), local_iter=self.args.local_iter)
            w_locals.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))
        # update global weights
        w_glob = FedAvg(w_locals)

        # copy weight to netlob
        net_glob.load_state_dict(w_glob)

        # print loss
        train_loss = sum(loss_locals) / len(loss_locals)
        return train_loss


class Trainer_for_beam_select_with_V2V(Trainer_abc):
    pass

################     trainer    ###################
###################################################


###################################################
################   evaluation   ###################

class Evaluator_for_beam_select(Evaluator_abc):
    def __init__(self, args, dataset_val) -> None:
        super().__init__(args, dataset_val)
        self.val_loader = DataLoader(
            dataset_val,
            #batch_size=args.val_batch_size, # To be added
            batch_size=1,
            shuffle=True,
            pin_memory=True,
        )


    def eval_a_round(self, net, round):
        net.eval()
        # testing
        test_loss = 0
        loss_func = lambda y_pred, y_true: -torch.sum(torch.mean(y_true[y_pred>0] * torch.log(y_pred[y_pred>0]), axis=0))
        preds_all = torch.empty((len(self.val_loader), 256))
        top_1 = TopKCategoricalAccuracy(k=1)
        top_5 = TopKCategoricalAccuracy(k=5)
        top_10 = TopKCategoricalAccuracy(k=10)
        for idx, (lidar, beams) in enumerate(self.val_loader):
            lidar, beams = lidar.to(self.args.device), beams.to(self.args.device)
            pred_beams = net(lidar)
            pred_beams = F.softmax(pred_beams, dim=1)

            preds_all[idx, :] = pred_beams
            # sum up batch loss
            test_loss += loss_func(pred_beams, beams).item()
            # get the index of the max log-probability
            top_1.update((pred_beams, torch.argmax(beams)))
            top_5.update((pred_beams, torch.argmax(beams)))
            top_10.update((pred_beams, torch.argmax(beams)))
        net.train()
        test_loss /= len(self.val_loader.dataset)
        top1_acc, top5_acc, top10_acc = top_1.compute(), top_5.compute(), top_10.compute()
        eval_metrices = {"top1_acc":top1_acc, "top5_acc":top5_acc, "top10_acc":top10_acc}
        return test_loss, eval_metrices

################   evaluation   ###################
###################################################