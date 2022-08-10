

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
        print_interval = int(local_iter/2)
        for epoch in range(local_epoch):
            for i, data in enumerate(self.ldr_train):
                """if (epoch*len(self.ldr_train)+i+1)%print_interval == 0:
                    print("Local training {}/{}".format(epoch*len(self.ldr_train)+i+1, local_iter))"""
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
            #import ipdb;ipdb.set_trace()
            if i < local_iter-local_epoch*len(self.ldr_train):
                """if (local_epoch*len(self.ldr_train)+i+1)%print_interval == 0:
                    print("Local training {}/{}".format(local_epoch*len(self.ldr_train)+i+1, local_iter))"""
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