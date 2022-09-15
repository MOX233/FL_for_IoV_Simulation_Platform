

import torch
from torch import nn
from utils.FL_utils import DatasetSplit
from torch.utils.data import DataLoader

class LocalUpdate_for_digit_cls(object):
    def __init__(self, args, dataset=None, idxs=None, local_bs=1) -> None:
        self.args = args
        self.ldr_train = DataLoader(
            DatasetSplit(dataset, idxs),
            batch_size=local_bs,
            shuffle=True,
            drop_last=True,
        )
        self.Loss = nn.CrossEntropyLoss()

    def train(self, net, local_iter=1):
        net.train()
        # train and update
        optimizer = torch.optim.SGD(
            net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        iter_loss = []
        local_epoch = int(local_iter/len(self.ldr_train))
        metrics = dict()
        for epoch in range(local_epoch):
            for i, data in enumerate(self.ldr_train):
                imgs, labels = data
                imgs = imgs.to(self.args.device)
                labels = labels.to(self.args.device)
                #forward pass
                outputs = net(imgs)
                loss = self.Loss(outputs, labels)

                #backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                iter_loss.append(loss.item())
        for i, data in enumerate(self.ldr_train):
            if i < local_iter-local_epoch*len(self.ldr_train):
                imgs, labels = data
                imgs = imgs.to(self.args.device)
                labels = labels.to(self.args.device)
                #forward pass
                outputs = net(imgs)
                loss = self.Loss(outputs, labels)

                #backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                iter_loss.append(loss.item())
            else:
                break
        # return net.state_dict(), sum(iter_loss) / len(iter_loss)
        return net.state_dict(), sum(iter_loss) / len(iter_loss)


class LocalUpdate_for_digit_cls_with_V2V(object):
    def __init__(self, args, dataset=None, car_info_list=None, local_bs=1) -> None:
        self.args = args
        self.car_info_list = car_info_list
        #idxs = [(car_id, car_iter_num, car_dataset_idxs),...]
        self.ldr_train_list = []
        for car_info in self.car_info_list:
            self.ldr_train_list.append(
                DataLoader(
                    #dataset,
                    DatasetSplit(dataset, car_info[2]),
                    batch_size=local_bs,
                    shuffle=True,
                    drop_last=True,
                )
            )
        self.Loss = nn.CrossEntropyLoss()

    def train(self, net, local_iter=1):
        net.train()
        # train and update
        optimizer = torch.optim.SGD(
            net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        iter_loss = []
        metrics = dict()

        for car_info, car_ldr_train in zip(self.car_info_list, self.ldr_train_list):
            car_local_iter = car_info[1]
            car_local_epoch = int(car_local_iter/len(car_ldr_train))
            for epoch in range(car_local_epoch):
                for i, data in enumerate(car_ldr_train):
                    imgs, labels = data
                    imgs = imgs.to(self.args.device)
                    labels = labels.to(self.args.device)
                    #forward pass
                    outputs = net(imgs)
                    loss = self.Loss(outputs, labels)

                    #backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    iter_loss.append(loss.item())
            for i, data in enumerate(car_ldr_train):
                #import ipdb;ipdb.set_trace()
                if i < local_iter-car_local_epoch*len(car_ldr_train):
                    imgs, labels = data
                    imgs = imgs.to(self.args.device)
                    labels = labels.to(self.args.device)
                    #forward pass
                    outputs = net(imgs)
                    loss = self.Loss(outputs, labels)

                    #backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    iter_loss.append(loss.item())
                else:
                    break

        return net.state_dict(), sum(iter_loss) / len(iter_loss)