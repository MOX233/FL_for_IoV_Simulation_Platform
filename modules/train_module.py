class Trainer():
    def __init__(self, args, FL_table, dataset_train, dict_users, trainer_for_task) -> None:
        self.args = args
        self.FL_table = FL_table
        self.dataset_train = dataset_train
        self.dict_users = dict_users
        self.train_loss_list = []
        self.trainer = trainer_for_task

    def train_a_round(self, net_glob, round):
        net_glob.train()
        idxs_users = [int(car.split('_')[-1]) for car in self.FL_table[round].keys()]
        print("Round {:3d}, Car num {}, Training start".format(round, len(idxs_users)))
        if idxs_users == []:
            train_loss = None
        else:
            idxs_list = [self.dict_users[i] for i in idxs_users]
            train_loss = self.trainer.train_a_round(idxs_list, net_glob)

            print('Average Training Loss {:.5f}'.format(train_loss))
        self.train_loss_list.append(train_loss)
        return net_glob, train_loss