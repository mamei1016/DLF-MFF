from argparse import Namespace
from logging import Logger
import numpy as np
import os
from train import fold_train
#from train_3cl import fold_train
from args import set_train_argument
from tool import set_log,  get_task_name, mkdir

import warnings
warnings.filterwarnings("ignore")


def training(args,log):
    info = log.info
    
    seed_first = args.seed
    data_path = args.data_path
    save_path = args.save_path
    args.task_names = get_task_name(data_path)
    
    score = []
    
    for num_fold in range(args.num_folds):
        info(f'Seed {args.seed}')
        args.seed = seed_first + num_fold
        args.save_path = os.path.join(save_path,f'Seed_{args.seed}')
        mkdir(args.save_path)
        
        fold_score = fold_train(args,log)  # 每个task的分数

        score.append(fold_score)
    score = np.array(score)   #n-折下的所有分数
    
    info(f'Running {args.num_folds} folds in total.')
    if args.num_folds > 1:
        for num_fold, fold_score in enumerate(score):
            info(f'Seed {seed_first + num_fold} : test {args.metric} = {np.nanmean(fold_score):.6f}')
            if args.task_num > 1:
                for one_name, one_score in zip(args.task_names, fold_score):
                    info(f' Task {one_name} {args.metric} = {one_score:.6f}')

    ave_task_score = np.nanmean(score, axis=1)  #每个任务在N——折下的分数
    score_ave = np.nanmean(ave_task_score)  #所有task的平均分
    score_std = np.nanstd(ave_task_score)
    info(f'final_all-task average test {args.metric} = {score_ave:.6f} +/- {score_std:.6f}') # 最终的分数，n-折-ntask下的一个平均值
    
    if args.task_num > 1:
        for i,one_name in enumerate(args.task_names):
            info(f'final every task Average test {one_name} {args.metric} = {np.nanmean(score[:, i]):.6f} +/- {np.nanstd(score[:, i]):.6f}')
            #每个task的平均值
    
    return score_ave, score_std

if __name__ == '__main__':
    args = set_train_argument()
    log = set_log('train',args.log_path)
    training(args,log)
