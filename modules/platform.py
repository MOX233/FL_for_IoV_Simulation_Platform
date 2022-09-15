import os
from utils.log_utils import save_ckpt
from modules import FLtable_module, data_allocate_module, task_module, traffic_module, train_module, evaluation_module

def run_platform(args):
    dataset_train, net_glob, generate_split_dict_for_task, evaluator_for_task, trainer_for_task, plot_func_for_task = task_module.get_task(args)
    car_tripinfo = traffic_module.get_tripinfo(args)
    if args.v2v == 0:
        FL_table = FLtable_module.generate_FLtable(args, car_tripinfo)
    elif args.v2v == 1:
        FL_table = FLtable_module.generate_FLtable_with_V2V_1(args, car_tripinfo)
    elif args.v2v == 2:
        FL_table = FLtable_module.generate_FLtable_with_V2V_2(args, car_tripinfo)
    elif args.v2v == 3:
        FL_table = FLtable_module.generate_FLtable_with_V2V_3(args, car_tripinfo)
    elif args.v2v == 4:
        FL_table = FLtable_module.generate_FLtable_with_V2V_4(args, car_tripinfo)
    elif args.v2v == -1:
        FL_table = FLtable_module.generate_FLtable_upperbound(args, car_tripinfo)
    else:
        exit("Unrecognized v2v type")
    dict_users = data_allocate_module.data_allocate(args, car_tripinfo, dataset_train, generate_split_dict_for_task)
    evaluator = evaluation_module.Evaluator(args, evaluator_for_task, plot_func_for_task)
    if args.v2v == 0:
        trainer = train_module.Trainer(args, FL_table, dataset_train, dict_users, trainer_for_task)
    elif args.v2v == -1:
        trainer = train_module.Trainer(args, FL_table, dataset_train, dict_users, trainer_for_task)
    else:
        trainer = train_module.Trainer_v2v(args, FL_table, dataset_train, dict_users, trainer_for_task)

    rounds = len(FL_table.keys())
    for round in range(rounds):
        #debug
        """print('before train a round:')
        print(net_glob.state_dict())
        import ipdb; ipdb.set_trace()"""

        net_glob, train_loss = trainer.train_a_round(net_glob, round)
        if train_loss == None:
            evaluator.eval_for_None(net_glob, round)
        else:
            evaluator.eval(net_glob, round)
        evaluator.record_eval_result(trainer.train_loss_list, round)
    
    save_ckpt(net_glob, args.ckpt_save_path, round)
    return