import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
import time
import numpy as np
from torch import nn
import copy
from utils.FL_utils import FedAvg, DatasetSplit, Trainer_abc, Evaluator_abc
from task_utils.digit_cls_utils.model import Net
from task_utils.digit_cls_utils.update import LocalUpdate_for_digit_cls
from task_utils.digit_cls_utils.utils_for_digit_cls import generate_split_dict, plot_for_digit_cls_task


def get_digit_cls_task(args):
    net = get_net_for_digit_cls(args)
    dataset_train, dataset_val = get_dataset_for_digit_cls(args)
    evaluator_for_digit_cls = Evaluator_for_digit_cls(args, dataset_val)
    trainer_for_digit_cls = Trainer_for_digit_cls(args, dataset_train)
    plot_for_digit_cls = plot_for_digit_cls_task
    return dataset_train, net, generate_split_dict_for_digit_cls, evaluator_for_digit_cls, trainer_for_digit_cls, plot_for_digit_cls


def get_net_for_digit_cls(args):
    net = Net().to(args.device)
    return net


def get_dataset_for_digit_cls(args):
    print("Loading dataset")
    start_time = time.time()
    dataset_train = torchvision.datasets.MNIST(root='~/Dataset/MNISTdata',
                                           train=True,
                                           transform=torchvision.transforms.ToTensor(),
                                           download=False)
    dataset_val = torchvision.datasets.MNIST(root='~/Dataset/MNISTdata',
                                           train=False,
                                           transform=torchvision.transforms.ToTensor(),
                                           download=False)
    end_time = time.time()
    print("Complete dataset loading with running time {:.3f}s".format(end_time-start_time))
    #print('dataset size',len(dataset_train),len(dataset_val))
    if args.simple_eval:
        simple_val_idxs = np.random.choice(list(range(len(dataset_val))),args.simple_eval_num,replace=False)
        dataset_val = DatasetSplit(dataset_val,simple_val_idxs)
    return dataset_train, dataset_val


###################################################
################ data allocator ###################

def generate_split_dict_for_digit_cls(args, dataset):
    if args.split_type in list(range(1, 11)):
        split_dict = generate_split_dict(dataset, type_num=args.split_type)
    else:
        exit('Error:undefined split type')
    return split_dict


################ data allocator ###################
###################################################


###################################################
################     trainer    ###################
class Trainer_for_digit_cls(Trainer_abc):
    # local_iter and local_batch_size can be given flexibly
    def __init__(self, args, dataset_train):
        super().__init__(args, dataset_train)

    def train_a_round(self, idxs_list, net_glob):
        loss_locals = []
        w_locals = []
        for idxs in idxs_list:
            local = LocalUpdate_for_digit_cls(args=self.args, dataset=self.dataset_train, idxs=idxs, local_bs=self.args.local_bs)
            w, loss = local.train(net=copy.deepcopy(net_glob), local_iter=self.args.local_iter)
            w_locals.append(copy.deepcopy(w))

            loss_locals.append(copy.deepcopy(loss))
        # update global weights
        if self.args.non_iid:
            #w_glob = FedAvg_digit_weighted(w_locals, skew=self.args.skew) #TODO
            w_glob = FedAvg(w_locals)
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

class Evaluator_for_digit_cls(Evaluator_abc):
    def __init__(self, args, dataset_val) -> None:
        super().__init__(args, dataset_val)
        self.val_loader = DataLoader(
            dataset_val,
            #batch_size=args.val_batch_size, # To be added
            batch_size=1,
            shuffle=False
        )
        self.Loss = nn.CrossEntropyLoss()


    def eval_a_round(self, net, round):
        start_time = time.time()
        metrics = dict()
        total = 0
        loss = 0
        correct = 0
        for i, data in enumerate(self.val_loader):
            imgs, labels = data
            imgs = imgs.to(self.args.device)
            labels = labels.to(self.args.device)
            with torch.no_grad():
                outputs = net(imgs)
                loss += self.Loss(outputs, labels).item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum()
        dt = time.time() - start_time
        loss = loss / total
        eval_metrices = {"acc": correct.item() / total}
        return loss, eval_metrices

################   evaluation   ###################
###################################################