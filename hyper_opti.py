from argparse import ArgumentParser, Namespace
from hyperopt import fmin, tpe, hp,Trials
from hyperopt.early_stop import no_progress_loss
import numpy as np
import os
from copy import deepcopy
from args import set_hyper_argument
from tool import set_log
from train_all import training
import matplotlib.pyplot as plt

space = {
         'fp_2_dim':hp.quniform('fp_2_dim', low=300, high=600, q=50),
         'hidden_size':hp.choice('hidden_size', [64,128,256,512]),
         'emb_dim_gnn':hp.choice('emb_dim_gnn', [32,64,128,256]),
         'emb_egnn':hp.choice('emb_egnn', [32,64,128,256]),
         'hidden_gnn':hp.choice('hidden_gnn', [64,128,256]),
         'dropout_fpn':hp.quniform('dropout_fpn', low=0.0, high=0.5, q=0.05),
         'dropout_gnn':hp.quniform('dropout_gnn', low=0.0, high=0.5, q=0.05),
         'dropout_egnn':hp.quniform('dropout_egnn', low=0.0, high=0.5, q=0.05),
         'linear_dim':hp.choice('linear_dim', [32,64,128,256])
}

def fn(space):
    search_no = args.search_now
    log_name = 'train'+str(search_no)
    log = set_log(log_name,args.log_path)
    result_path = os.path.join(args.log_path, 'hyper_para_result.txt')
    
    list = ['fp_2_dim','hidden_size','emb_dim_gnn','emb_egnn','hidden_gnn','linear_dim']
    for one in list:
        space[one] = int(space[one])
    hyperp = deepcopy(args)
    name_list = []
    change_args = []
    for key,value in space.items():
        name_list.append(str(key))
        name_list.append('-')
        name_list.append((str(value))[:5])
        name_list.append('-')
        setattr(hyperp,key,value)
    dir_name = "".join(name_list)
    dir_name = dir_name[:10]
    hyperp.save_path = os.path.join(hyperp.model_path, dir_name)
    
    ave,std = training(hyperp,log)
    
    with open(result_path,'a') as file:
        file.write(str(space)+'\n')
        file.write('Result '+str(hyperp.metric)+' : '+str(ave)+' +/- '+str(std)+'\n')
    
    if ave is None:
        if hyperp.dataset_type == 'classification':
            ave = 0
        else:
            raise ValueError('Result of model is error.')
    
    args.search_now += 1

    #print('search_now', args.search_now)

    if hyperp.dataset_type == 'classification':
        return -ave
    else:
        return ave

def hyper_searching(args):
    result_path = os.path.join(args.log_path,'hyper_para_result.txt')
    #trial_path = os.path.join(args.log_path,'trails')
    trials = Trials()

    result = fmin(fn,space,tpe.suggest,args.search_num,trials=trials)

    with open(result_path, 'a') as file:
        file.write('Best Hyperparameters : \n')
        file.write(str(result) + '\n')


    #print('result:', result)

    #print('trials:')

    #for trial in trials.trials[:2]:
        #print(trial)

    return result, trials

if __name__ == '__main__':
    args = set_hyper_argument()
    result, trials=hyper_searching(args)
