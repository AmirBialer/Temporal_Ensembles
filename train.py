import argparse
import os
import shutil
import time
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data as data
import torchvision.transforms as transforms
from keras.utils import np_utils
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split,StratifiedKFold, StratifiedShuffleSplit,RandomizedSearchCV
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler,Dataset, TensorDataset

import torchvision.datasets as datasets


import pdb
import bisect


import math
from math import ceil
import torch.nn.functional as F
from Pi_Model_Pytorch import Pi_Model
from methods import  Train_Pi_nepochs, Evaluate_Model 

parser = argparse.ArgumentParser(description='PyTorch Semi-supervised learning Training')
parser.add_argument('--arch', '-a', metavar='ARCH', default='wideresnet',
                    help='model architecture: '+ ' (default: wideresnet)')
parser.add_argument('--model', '-m', metavar='MODEL', default='baseline',
                    help='model: '+' (default: baseline)', choices=['baseline', 'pi', 'mt'])
parser.add_argument('--optim', '-o', metavar='OPTIM', default='adam',
                    help='optimizer: '+' (default: adam)', choices=['adam', 'sgd'])
parser.add_argument('--dataset', '-d', metavar='DATASET', default='cifar10_zca',
                    help='dataset: '+' (default: cifar10)', choices=['cifar10', 'cifar10_zca', 'svhn'])
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 225)')
parser.add_argument('--lr', '--learning-rate', default=0.003, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--weight_l1', '--l1', default=1e-3, type=float,
                    metavar='W1', help='l1 regularization (default: 1e-3)')
parser.add_argument('--print-freq', '-p', default=10000, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--num_classes',default=10, type=int, help='number of classes in the model')
parser.add_argument('--ckpt', default='ckpt', type=str, metavar='PATH',
                    help='path to save checkpoint (default: ckpt)')
parser.add_argument('--boundary',default=0, type=int, help='different label/unlabel division [0,9]')
parser.add_argument('--path',default="/home/amirbial/Computational_Learning/Scratch/datasets/", type=str)
parser.add_argument('--label_ratio',default=0.9, type=float)
parser.add_argument('--improved_model',default=False, type=bool)

parser.add_argument('--data_name',default="wine-quality-white.csv", type=str)
parser.add_argument('--gpu',default=0, type=str, help='cuda_visible_devices')
args = parser.parse_args()

def Load_Data_And_Preprocess(debug=False):
    data=pd.read_csv(args.path+args.data_name)
    if debug and len(data)>500: 
        data=data.sample(500)

    data=data.apply(pd.to_numeric)
    class_col=data.columns[-1]
    X=data.drop(columns=[class_col])
    y=data[class_col]
    num_features=len(X.columns.values)
    num_classes=y.nunique()
    args.num_classes=num_classes
    y=LabelEncoder().fit_transform(y)
    one_hot_y=np_utils.to_categorical(y)
    return X, y, one_hot_y, num_features, num_classes

def Train_Test_Split(X,y,train_index, test_index,one_hot_y, label_percentage=0.9, batch_size=100, train_size=0.7, val_size=0.8):

    x_train=X.iloc[train_index]
    x_labeled_test=X.iloc[test_index]
    y_train=y[train_index]
    y_labeled_test=y[test_index]  
    x_labeled, x_unlabeled, y_labeled, y_unlabeled = train_test_split(x_train, y_train, train_size=label_percentage)
    x_labeled_train, x_labeled_val, y_labeled_train, y_labeled_val = train_test_split(x_labeled, y_labeled, train_size=val_size)

    train_labeled_dataset=TensorDataset(torch.tensor(x_labeled_train.values, dtype=torch.float), torch.tensor(y_labeled_train, dtype=torch.long))
    val_labeled_dataset=TensorDataset(torch.tensor(x_labeled_val.values, dtype=torch.float), torch.tensor(y_labeled_val, dtype=torch.long))
    train_unlabeled_dataset = TensorDataset(torch.tensor(x_unlabeled.values, dtype=torch.float),torch.tensor(y_unlabeled, dtype=torch.long))
    test_dataset = TensorDataset(torch.tensor(x_labeled_test.values, dtype=torch.float),torch.tensor(y_labeled_test, dtype=torch.long)) 

    train_labeled_dataloader = DataLoader(
        train_labeled_dataset,  
        sampler=RandomSampler(train_labeled_dataset), 
        batch_size=batch_size  
    )
    train_unlabeled_dataloader = DataLoader(
        train_unlabeled_dataset,  
        sampler=RandomSampler(train_unlabeled_dataset),  
        batch_size=batch_size  
    )
    val_labeled_dataloader = DataLoader(
        val_labeled_dataset,  
        sampler=RandomSampler(val_labeled_dataset),
        batch_size=batch_size  
    )
    test_dataloader = DataLoader(
        test_dataset,  
        sampler=SequentialSampler(test_dataset), 
        batch_size=batch_size  
    )
    return train_labeled_dataloader, train_unlabeled_dataloader,val_labeled_dataloader, test_dataloader

def main():
    print("start main")
    improved=args.improved_model
    debug=False
    label_percentage=args.label_ratio
    print("Improve ="+str(improved))
    print(label_percentage)
    results_df=pd.DataFrame()
    reader_files_list=open("/home/amirbial/Computational_Learning/Scratch/datasets/files.txt")
    files_list=reader_files_list.read().splitlines()
    global args

    criterion = nn.CrossEntropyLoss(size_average=False).cuda()
    criterion_mse = nn.MSELoss(size_average=False).cuda()
    criterion_kl = nn.KLDivLoss(size_average=False).cuda()    
    criterion_l1 = nn.L1Loss(size_average=False).cuda()    
    criterions = (criterion, criterion_mse, criterion_kl, criterion_l1)
    drop_rate_list=[0.1,0.15,0.2]
    noise_list=[0.1,0.15,0.2]
    drop_rate = 0.15
    noise=0.15
    batch_size = 100


    for file_path in files_list:#iterate each file
        print(file_path)
        args.data_name=file_path
        # Data loading code
        X, y, one_hot_y, num_features, num_classes = Load_Data_And_Preprocess(debug=debug)
        print(num_classes)
        fold_results=pd.DataFrame()
        if debug:
            cv_outer = StratifiedKFold(n_splits=2,  random_state=42, shuffle=True)   
            cv_inner= StratifiedKFold(n_splits=2,  random_state=42, shuffle=True) # shorter time runining for debugging perposes
        else:
            cv_outer = StratifiedKFold(n_splits=10,  random_state=42, shuffle=True)    #10 fold cross validation:
            cv_inner= StratifiedKFold(n_splits=3,  random_state=42, shuffle=True) #3 fold inner validation for hyperparameters random search
        for train_index, test_index in cv_outer.split(X, y):
            start_train_time=time.time()

            best_acc=0
            best_param={"Accuracy":0, "noise":0.05, "drop_rate":0.1}

            #Train for random search of hyper parameters:
            for train_index_inner, test_index_inner in cv_inner.split(X, y):
                r=np.random.randint(3,size=2)
                noise=noise_list[r[0]]
                drop_rate=drop_rate_list[r[1]]

                train_labeled_dataloader, train_unlabeled_dataloader,val_labeled_dataloader, test_dataloader = Train_Test_Split(
                X, y,train_index_inner, test_index_inner,  one_hot_y, batch_size=batch_size, label_percentage=label_percentage)
                results=Train_Pi_nepochs(train_labeled_dataloader, train_unlabeled_dataloader,val_labeled_dataloader, test_dataloader, criterions, args, drop_rate, noise, num_features, num_classes, start_train_time, improved=False)

                if results["Accuracy"]>best_param["Accuracy"]:
                    best_param["Accuracy"]=results["Accuracy"]
                    best_param["noise"]=results["noise"]
                    best_param["drop_rate"]=results["drop_rate"]


            #Now fit model with best parameters:
            train_labeled_dataloader, train_unlabeled_dataloader,val_labeled_dataloader, test_dataloader= Train_Test_Split(
                X, y,train_index, test_index,  one_hot_y, batch_size=batch_size,label_percentage=label_percentage)
            results=Train_Pi_nepochs(train_labeled_dataloader, train_unlabeled_dataloader, val_labeled_dataloader,test_dataloader, criterions, args,  best_param["drop_rate"],  best_param["noise"],num_features, num_classes,start_train_time, improved=improved,data=( X, y,train_index, test_index))
            print(results)       
            fold_results=fold_results.append(results, ignore_index=True)
        fold_results.to_csv("/home/amirbial/Computational_Learning/Scratch/results/"+str(improved)+str(label_percentage)+args.data_name, index=False)
        mean_results=fold_results.mean()
        mean_results["dataset name"]=file_path
        results_df=results_df.append(mean_results, ignore_index=True)
    print("finish")
    results_df.to_csv("/home/amirbial/Computational_Learning/Scratch/results/"+str(improved)+str(label_percentage)+"All_Summary.csv", index=False)




if __name__ == '__main__':
    main()
