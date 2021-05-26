import os
import argparse
import csv
import sys
import numpy as np
import datetime
import torch
import torch.optim as optim
from tqdm import tqdm
import pickle

sys.path.append("..")
os.chdir(sys.path[0])
from Model_class.model_KAeDCN import ConvRec
from util_New import *

NOW_TIME = datetime.datetime.now()
NOW_TIME_Str = str(NOW_TIME.year) + '-' + str(NOW_TIME.month) + '-' + str(NOW_TIME.day) + '-' + str(
            NOW_TIME.hour) + '.' + str(NOW_TIME.minute)

def analysis_the_data(dataset):
    [train, valid, test, itemnum] = dataset  # Get data according to ratio
    print('------Analysis the data:-------')
    print("Number of sessions  :", len(train) + len(valid) + len(test))  
    print("Number of items  :", itemnum)
    action = 0
    for i in train:
        action += np.count_nonzero(i)
    for i in valid:
        action += np.count_nonzero(i)
    for i in test:
        action += np.count_nonzero(i)
    print("Number of actions  :", action)
    print("Average length of sessions  :", action / (len(train) + len(valid) + len(test)))
    print('------Analysis the data complete:-------')
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset',   default='ml_items_all_user_final')           # the name of data
    parser.add_argument('--Root_data', default='./Data/')                           # the root path of data
    parser.add_argument('--retrain',   default=False, type=bool)                    # whether to continue training the previous model
    parser.add_argument('--test',      default=False, type=bool)                    # whether a model needs to be tested


    # ------------KG Parameters--------------------
    parser.add_argument('--isKG', default=True, type=bool)                          # Whether to use KG
    parser.add_argument('--KGE_path', default='./Data/KG_data/entity_embeddings_')  # path of KG Embedding 
    parser.add_argument('--KGE', default='TransE', type=str)                        # the name of KG representation method
    parser.add_argument('--KGtype', default='CPU', type=str)                        
    parser.add_argument('--entity_dim', default=50, type=int)                       # the dimension of KG embeddings 
    
    # ------------Other Parameters------------------------
    parser.add_argument('--top_k', default=10, type=int)

    parser.add_argument('--train_dir', default='default')                           # default
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--lr', default=0.001, type=float)                          # the learning rate

    parser.add_argument('--maxlen', default=50, type=int)                           # the maxlens
    parser.add_argument('--embed_dim', default=250, type=int)                       # the dimension of RS embeddings 
    parser.add_argument('--spatial_kernel_size', default=5, type=int)               # the kernel size in SAM
    parser.add_argument('--channel_reduction', default=1, type=int)                 # the reduction ratio in CAM

    parser.add_argument('--ffn_embed_dim', default=250, type=int)                   # the dimension of FFN in DCN
    parser.add_argument('--dropout', default=0.2, type=float)
    parser.add_argument('--weight_dropout', default=0.2, type=float)

    parser.add_argument('--layers', default=2, type=int)                            # the number of layers 
    parser.add_argument('--heads', default=1, type=int)                             # default

    parser.add_argument('--decoder_kernel_size_list', default=[15, 15])               # the kernel size of DCN ; the length of the list depends on the number of layer

    parser.add_argument('--num_epochs', default=100, type=int)                      # train epoch
    parser.add_argument('--num_neg_samples', default=400, type=int)                 # Note: 100 is sufficient
    parser.add_argument('--eval_epoch', default=5, type=int)


    # Check if your system supports CUDA
    use_cuda = torch.cuda.is_available()

    # Setup GPU optimization if CUDA is supported
    if use_cuda:
        computing_device = torch.device("cuda")
        extras = {"num_workers": 1, "pin_memory": True}
        print("CUDA is supported")
        print()
    else:  # Otherwise, train on the CPU
        computing_device = torch.device("cpu")
        extras = False
        print("CUDA NOT supported")
        print()

    parser.add_argument('--computing_device', default=computing_device)

    # Get the arguments
    try:
        # if running from command line
        args = parser.parse_args()
    except:
        # if running in IDEs
        args = parser.parse_known_args()[0]

    Origial_data_path = args.Root_data + args.dataset + '.txt'  
    data_path         = args.Root_data + args.dataset + '.pkl'  
    
    retrain_model_path = '../Model_Retrain/'                                                                    # Save the path of the model to be retrained.

    if args.retrain:
        result_path = 'KAeDCN_results/' + 'Retrain_' + args.dataset + '_' + args.train_dir + NOW_TIME_Str      # the path of retrain result 
    else:
        result_path = 'KAeDCN_results/' + args.dataset + '_' + args.train_dir + NOW_TIME_Str                   
    if not os.path.isdir(result_path):
        os.makedirs(result_path)

    # Write the parameters line by line to a file
    Name_args_logs_file = NOW_TIME_Str + '_args_log.txt'
    with open(os.path.join(result_path, Name_args_logs_file), 'w') as f:
        f.write('Write parameters：')
        f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))
        f.write('\n')
    f.close()

    # Start loading data
    if os.path.exists(data_path):               # pkl data already exists
        pickle_in = open(data_path, "rb")
        dataset = pickle.load(pickle_in)
    else:
        dataset = data_partition(Origial_data_path, Max_lens=args.maxlen)
        pickle_out = open(data_path, "wb")
        pickle.dump(dataset, pickle_out)
        pickle_out.close()

    [train, valid, test, itemnum] = dataset  

    analysis_the_data(dataset)

    num_batch = len(train) // args.batch_size
    print("The batch size is:", num_batch)
    # Counting the number of batch

    f = open(os.path.join(result_path, Name_args_logs_file), 'a')



    conv_model = ConvRec(args, itemnum)                                     
    conv_model = conv_model.to(args.computing_device, non_blocking=True)    

    # Note: testing a pretrained model
    if args.retrain:
        if os.path.exists(retrain_model_path + "pretrained_model.pth"):
            print('Start to continue the model training, the directory exists target model:')
            conv_model.load_state_dict(torch.load(result_path + "pretrained_model.pth"))  # Loading Models
            t_test = evaluate(conv_model, test, itemnum, args, num_workers=4)             # Start testing
            model_performance = "Model performance on test:\n " + str(t_test) + '\n'               # Print test results
            print(model_performance)
        else:
            print('No target training model exists in the catalog:')
    # Note: testing a pretrained model
    if args.test:
        if os.path.exists(result_path + "/pretrained_model.pth"):
            print('Start testing the model, the directory exists for the target model:')
            conv_model.load_state_dict(torch.load(result_path + "pretrained_model.pth"))  # Loading the model
            t_test = evaluate(conv_model, test, itemnum, args, num_workers=4)             # Begin testing
            model_performance = "Model performance on test:\n " + str(t_test) + '\n'               # Print results
            print(model_performance)
            print('Test complete!')
        else:
            print('No target test model exists in the catalog!')
        sys.exit(0)

    # Define Optimizer
    optimizer = optim.Adam(conv_model.parameters(), lr=args.lr, betas=(0.9, 0.98), weight_decay=0.0)

    f.write('\n The model parameters are:\n')
    f.write('\n' + str(args) + '\n')  # Write the parameters to the created log file
    f.write('\n')
    f.flush()

    best_val_loss = 1e6
    train_losses = []
    val_losses = []

    best_ndcg = 0
    best_hit = 0
    model_performance = None

    stop_count = 0
    total_epochs = 1

    dataset = Dataset(train, args, itemnum, train=True)  # Initialization Dataset
    sampler = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, num_workers=4, pin_memory=True)
  

    for epoch in range(1, args.num_epochs + 1):
        conv_model.train()  # Switching model states

        epoch_losses = []

        for step, (seq, pos) in tqdm(enumerate(sampler), total=len(sampler)):

            optimizer.zero_grad()           # Clear gradient
            seq = torch.LongTensor(seq).to(args.computing_device, non_blocking=True)  # cuda
            pos = torch.LongTensor(pos).to(args.computing_device, non_blocking=True)  # cuda

            loss, _ = conv_model.forward(seq, pos=pos)      
            epoch_losses.append(loss.item())

            # Compute gradients
            loss.backward()

            # Update the parameters
            optimizer.step()

        if total_epochs % args.eval_epoch == 0:
            str_evaluate = '---------Start testing---------' + '' +'\n'
            print(str_evaluate)
            t_valid = evaluate(conv_model, valid, itemnum, args, num_workers=4)
            str_t_valid = \
                'num of steps:' + str(total_epochs) + \
                '\nvalid (MRR@' + str(args.top_k) + ': ' + str(t_valid[0]) +\
                ', NDCG@' + str(args.top_k) + ':' + str(t_valid[1]) + \
                ', HR@' + str(args.top_k) + ':' + str(t_valid[2]) + \
                '\nvalid (MRR@' + str(args.top_k + 10) + ': ' + str(t_valid[3]) +\
                ', NDCG@' + str(args.top_k + 10) + ':' + str(t_valid[4]) + \
                ', HR@' + str(args.top_k + 10) + ':' + str(t_valid[5])
            print('num of steps:%d, '
                  'valid (MRR@%d: %.4f, NDCG@%d: %.4f, HR@%d: %.4f), '
                  'valid (MRR@%d: %.4f, NDCG@%d: %.4f, HR@%d: %.4f)'
                  % (total_epochs,
                     args.top_k, t_valid[0],
                     args.top_k, t_valid[1],
                     args.top_k, t_valid[2],
                     args.top_k + 10, t_valid[3],
                     args.top_k + 10, t_valid[4],
                     args.top_k + 10, t_valid[5]))

            f.write(str_t_valid + '\n')
            f.flush()

            if t_valid[0] > best_ndcg:
                best_ndcg = t_valid[0]
                torch.save(conv_model.state_dict(), result_path + "/pretrained_model.pth")
                temp_str = 'Save the currently acquired best model，total_epochs = ' + str(total_epochs) + ' epoch = ' + str(epoch) + '\n' + 'end\n'
                print(temp_str)
                f.write(temp_str)
                f.flush()
                stop_count = 1
            else:
                stop_count += 1
                temp_str = '---------The results of this round of testing were unsatisfactory, with the best current results being:' + str(best_ndcg) + ',---------'
                print(temp_str)
                f.write(temp_str)
                f.flush()
            if stop_count == 5:  # model did not improve 5 consequetive times
                temp_str = '---------No better results, end of training.---------\n'
                print(temp_str)
                f.write(temp_str)
                f.flush()
                break

        total_epochs += 1

        train_loss = np.mean(epoch_losses)
        print(str(epoch) + "epoch, loss = ", train_loss, 'total_epochs = ', str(total_epochs))
        f.write(str(epoch) + "epoch, loss = " + str(train_loss) + '\n')
        f.flush()
        torch.cuda.empty_cache()

    conv_model = ConvRec(args, itemnum)
    conv_model.load_state_dict(torch.load(result_path + "/pretrained_model.pth"))

    conv_model = conv_model.to(args.computing_device)

    t_test = evaluate(conv_model, test, itemnum, args, num_workers=4)

    str_t_test = \
        '\nModel performance on test (easy) :' \
        '\nvalid (MRR@' + str(args.top_k) + ': ' + str(t_test[0]) + \
        ', NDCG@' + str(args.top_k) + ':' + str(t_test[1]) + \
        ', HR@' + str(args.top_k) + ':' + str(t_test[2]) + \
        '\nvalid (MRR@' + str(args.top_k + 10) + ': ' + str(t_test[3]) + \
        ', NDCG@' + str(args.top_k + 10) + ':' + str(t_test[4]) + \
        ', HR@' + str(args.top_k + 10) + ':' + str(t_test[5])

    print('\nModel performance on test (easy) :'
          'valid (MRR@%d: %.4f, NDCG@%d: %.4f, HR@%d: %.4f), '
          'valid (MRR@%d: %.4f, NDCG@%d: %.4f, HR@%d: %.4f)'
          % (args.top_k, t_test[0],
             args.top_k, t_test[1],
             args.top_k, t_test[2],
             args.top_k + 10, t_test[3],
             args.top_k + 10, t_test[4],
             args.top_k + 10, t_test[5]))

    f.write(str_t_test + '\n')
    f.flush()
    f.close()

    print("Done！！")
