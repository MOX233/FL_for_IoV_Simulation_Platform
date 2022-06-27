from utils.sampling import sample_iid, sample_noniid

def data_allocate(args, car_tripinfo, dataset_train, generate_split_dict_for_task):
    num_users = len(car_tripinfo)
    if not args.non_iid:
        dict_users = sample_iid(dataset_train, args.num_items, num_users)
    else:
        if generate_split_dict_for_task != None:
            split_dict = generate_split_dict_for_task(args, dataset_train)
            dict_users = sample_noniid(dataset_train, split_dict, args.num_items, num_users)
        else:
            exit("non-i.i.d. has not been realized for task: "+args.task)
    return dict_users
