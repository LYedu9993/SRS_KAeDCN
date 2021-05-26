import random
import numpy as np
from collections import defaultdict
from random import shuffle
import torch
import pandas as pd
from tqdm import tqdm
from sampler import Dataset


def split_list(list_part, lens):
    
    starts = 0
    ends = starts + lens
    out = list()
    
    while(ends < len(list_part)):
        out.append(list_part[starts:ends])
        starts = ends
        ends = starts + lens
        if ends >= len(list_part):
            out.append(list_part[starts:])

    return out



def data_partition(Path_name, percentage=[0.1, 0.2], Max_lens = 50, rating = 1):
    isNew = True
    if Path_name.split('.')[-1] == 'csv':
        isNew = False
    itemnum = 0

    
    sessions = list()
    session_train = []
    session_valid = []
    session_test = []
    # assume user/item index starting from 1
    session_id = 0
    
    f = open(Path_name, 'r')
    total_length = 0
    max_length = 0
    for line in f:
        if isNew:
            items = [int(l) for l in line.strip().split(' ')]
        else:
            items = [int(l) for l in line.rstrip().split(',')]


        total_length += len(items)

        if max_length < len(items):
            max_length = len(items)                     # Use max_length to count the maximum length of all interaction data

        itemnum = max(max(items), itemnum)              # Count the total number of items
        
        sessions.append(items)
        session_id += 1
    
    len_sessions = len(sessions)
    
    count_new_add = 0
    indexs = 0
    Threshold = int(rating * Max_lens)
    while(indexs < len(sessions)):
        if indexs >= len(sessions):
            raise ValueError('indexs = ',indexs,'lensession = ',len(sessions),'len_session = ',len_sessions)
        temp_sessions = sessions[indexs]
       
        if len(temp_sessions) > Max_lens:
            
            new_lists = split_list(temp_sessions, Threshold)
            if len(new_lists) != 0:
                del sessions[indexs]
                count_new_add = count_new_add - 1 
                for e in new_lists:
                    sessions.insert(indexs,e)
                    count_new_add = count_new_add + 1 
                    indexs = indexs + 1
                    len_sessions = len_sessions + 1
                indexs = indexs - 1
        indexs = indexs + 1
    session_id_new = len(sessions)
    if session_id + count_new_add != session_id_new:
        raise ValueError('session_id + count_new_add != session_id_new:')
    else:
        session_id = session_id_new
        
    valid_perc = percentage[0]               # 0.1 validation
    test_perc = percentage[1]                # 0.2 test

    total_sessions = session_id              # Number of all sessions

    shuffle_indices = np.random.permutation(range(total_sessions))  

    train_index = int(total_sessions * (1 - valid_perc - test_perc))  
    valid_index = int(total_sessions * (1 - test_perc))  
    if (train_index == valid_index): valid_index += 1  # break the tie

    train_indices = shuffle_indices[:train_index]
    valid_indices = shuffle_indices[train_index:valid_index]
    test_indices = shuffle_indices[valid_index:]
    for i in train_indices:
        session_train.append(sessions[i])
    for i in valid_indices:
        session_valid.append(sessions[i])
    for i in test_indices:
        session_test.append(sessions[i])

    return [np.asarray(session_train), np.asarray(session_valid), np.asarray(session_test), itemnum]



def evaluate(model, test_sessions, itemnum, args, num_workers=4):
    # set the environment
    model.eval()

    MRR = 0.0
    NDCG = 0.0
    HT = 0.0

    MRR_plus_10 = 0.0
    NDCG_plus_10 = 0.0
    HT_plus_10 = 0.0

    valid_sessions = 0.0

    all_items = np.array(range(1, itemnum + 1))  
    all_items_tensor = torch.LongTensor(all_items).to(args.computing_device, non_blocking=True)  # 将对象放入cuda 

    dataset = Dataset(test_sessions, args, itemnum, False)


    sampler = torch.utils.data.DataLoader(dataset,
                                          batch_size=args.batch_size,
                                          num_workers=num_workers,
                                          pin_memory=True)

    with torch.no_grad():

        for step, (seq, grouth_truth) in tqdm(enumerate(sampler), total=len(sampler)):

            seq = torch.LongTensor(seq).to(args.computing_device, non_blocking=True)

            _, rank_20 = model.forward(seq, test_item=all_items_tensor)

            rank_20 = rank_20.cpu().detach().numpy()  # 128 * 20
            grouth_truth = grouth_truth.view(-1, 1).numpy()

            try:
                ranks = np.where(rank_20 == grouth_truth)

                try:
                    ranks = ranks[1]
                except:
                    ranks = ranks[0]

                for rank in ranks:

                    if rank < args.top_k:
                        MRR += 1.0 / (rank + 1)
                        NDCG += 1 / np.log2(rank + 2)
                        HT += 1

                    if rank < args.top_k + 10:
                        MRR_plus_10 += 1.0 / (rank + 1)
                        NDCG_plus_10 += 1 / np.log2(rank + 2)
                        HT_plus_10 += 1

            except:
                continue  # where rank returns none

        valid_sessions = len(dataset)

    return MRR / valid_sessions, \
           NDCG / valid_sessions, \
           HT / valid_sessions, \
           MRR_plus_10 / valid_sessions, \
           NDCG_plus_10 / valid_sessions, \
           HT_plus_10 / valid_sessions

