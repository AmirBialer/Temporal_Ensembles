import math
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, roc_curve, precision_score, average_precision_score, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
import warnings
import numpy as np
import time
from Pi_Model_Pytorch import Pi_Model
import bisect
import xgboost
learning_rate, weights_cl = [], []

np.seterr(divide='ignore', invalid='ignore')

def Train_Pi_nepochs(train_labeled_dataloader, train_unlabeled_dataloader,val_labeled_dataloader, test_dataloader, criterions, args, drop_rate, noise,num_features, num_classes, start_train_time, improved, data=None):
    global weights_cl

    # create model
    model=Pi_Model(num_features, num_classes, dropout=drop_rate, noise=noise)
    model = torch.nn.DataParallel(model).cuda()
    best_model=model
    best_acc=0

    # deifine loss function (criterion) and optimizer

    optimizer = torch.optim.Adam(model.parameters(), args.lr,
                                    betas=(0.9, 0.999),
                                    weight_decay=args.weight_decay)
    for epoch in range(args.start_epoch, args.epochs):
        #train
        lr = adjust_learning_rate_adam(optimizer, epoch, args)
        model,prec1_tr, loss_tr, loss_cl_tr, weight_cl = train_pi_1epoch(train_labeled_dataloader, train_unlabeled_dataloader, model, criterions,
                                                                optimizer, epoch, args)

        # evaluate on validation set
        results_val=Evaluate_Model(val_labeled_dataloader, model, num_classes)
        if results_val["Accuracy"]>=best_acc:
            best_acc=results_val["Accuracy"]
            best_model=model

        #print("accuracy val: {}, roc-auc val: {}".format(results["Accuracy"], results["ROC_AUC"]))
        weights_cl.append(weight_cl)
        learning_rate.append(lr)


    #Evaluate Model
    end_train_time=time.time()
    if improved==False:
        results_test=Evaluate_Model(test_dataloader, best_model, num_classes)
    if improved:
        X, y,train_index, test_index=data
        x_train, y_train, x_test,y_labeled_test=X.iloc[train_index], y[train_index], X.iloc[test_index], y[test_index]
        if num_classes>2:
            model_xgb=xgboost.XGBClassifier(**{"num_class":num_classes, "objective":"multi:softprob"})#.set_params()
        else:
            model_xgb=xgboost.XGBClassifier()
        model_xgb.fit(x_train, y_train)
        results_test=Evaluate_Improved_Model(test_dataloader, best_model,model_xgb, num_classes)

    results_test["train_time [seconds]"]=end_train_time-start_train_time
    results_test["noise"]=noise
    results_test["drop_rate"]=drop_rate
    return results_test



def train_pi_1epoch(label_loader, unlabel_loader, model, criterions,optimizer, epoch, args, weight_pi=20.0):
    end = time.time()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_pi = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    weights_cl = AverageMeter()

    # switch to train mode
    model.train()
    criterion, criterion_mse, _, criterion_l1 = criterions

    label_iter = iter(label_loader)
    unlabel_iter = iter(unlabel_loader)
    len_iter = len(unlabel_iter)
    for i in range(len_iter):
        # set weights for the consistency loss
        weight_cl = cal_consistency_weight(epoch * len_iter + i, end_ep=(args.epochs // 2) * len_iter, end_w=1.0)

        try:
            #input, target, input1 = next(label_iter)# Amir Change
            #I ignore all input1 because different implementation of noise- they put the noise in the transformation dataloader, while I do it in the network
            input, target = next(label_iter)
        except StopIteration:
            label_iter = iter(label_loader)
            input, target = next(label_iter)
            #input, target, input1 = next(label_iter)

        #input_ul, _, input1_ul = next(unlabel_iter)
        input_ul, _ = next(unlabel_iter)
        sl = input.shape
        su = input_ul.shape
        batch_size = sl[0] + su[0]


        # measure data loading time
        data_time.update(time.time() - end)
        target = target.cuda(non_blocking=True)
        input_var = torch.autograd.Variable(input)
        #input1_var = torch.autograd.Variable(input1)
        input_ul_var = torch.autograd.Variable(input_ul)
        #input1_ul_var = torch.autograd.Variable(input1_ul)
        input_concat_var = torch.cat([input_var, input_ul_var])
        #input1_concat_var = torch.cat([input1_var, input1_ul_var])

        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_concat_var) #z1
        with torch.no_grad():
            output1 = model(input_concat_var)# z1 tilda 


        pred = F.softmax(output, 1) 
        pred1 = F.softmax(output1, 1)
        output_prob=pred[:sl[0]]
        output_logit = output[:sl[0]]# consistency loss on logit is better


        loss_ce = criterion(output_logit, target_var) / float(sl[0])
        #loss_pi = criterion_mse(output, output1) / float(args.num_classes * batch_size)
        loss_pi = criterion_mse(pred, pred1) / float(args.num_classes * batch_size)

        reg_l1 = cal_reg_l1(model, criterion_l1)

        loss = loss_ce + args.weight_l1 * reg_l1 + weight_cl * weight_pi * loss_pi

        # measure accuracy and record loss
        #prec1, prec5 = accuracy(output_label.data, target, topk=(1, 5))
        prec1= accuracy(output_prob.data, target)
        #prec1= accuracy(output_label.data, target)


        prec5=1
        losses.update(loss_ce.item(), input.size(0))
        losses_pi.update(loss_pi.item(), input.size(0))
        #top1.update(prec1.item(), input.size(0)) returning list instead of tensor so I changes
        top1.update(prec1[0], input.size(0))
        #top5.update(prec5.item(), input.size(0)) I change because prec5 not working for now
        top5.update(prec5, input.size(0))
        weights_cl.update(weight_cl, input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    return model,top1.avg, losses.avg, losses_pi.avg, weights_cl.avg



class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()

    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

    


def cal_consistency_weight(epoch, init_ep=0, end_ep=150, init_w=0.0, end_w=20.0):
    """Sets the weights for the consistency loss"""
    if epoch > end_ep:
        weight_cl = end_w
    elif epoch < init_ep:
        weight_cl = init_w
    else:
        T = float(epoch - init_ep) / float(end_ep - init_ep)
        # weight_mse = T * (end_w - init_w) + init_w #linear
        weight_cl = (math.exp(-5.0 * (1.0 - T) * (1.0 - T))) * (end_w - init_w) + init_w  # exp
    # print('Consistency weight: %f'%weight_cl)
    return weight_cl


def cal_reg_l1(model, criterion_l1):
    reg_loss = 0
    np = 0
    for param in model.parameters():
        reg_loss += criterion_l1(param, torch.zeros_like(param))
        np += param.nelement()
    reg_loss = reg_loss / np
    return reg_loss

def Evaluate_Model(test_loader, model,num_classes):
    model.eval()    # switch to evaluate mode

    start_time = time.time()

    pred_list=[]
    pred_prob_list=[]
    labels=[]
    with torch.no_grad():
        #for i, (input, target, _) in enumerate(val_loader):
        for i, (input, target) in enumerate(test_loader):
            sl = input.shape
            batch_size = sl[0]
            target = target.cuda(non_blocking=True)
            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)

            # compute output
            output = model(input_var)
            pred=F.softmax(output, 1)
            prob=pred.detach().cpu().numpy()
            #save results
            pred_prob_list.append(prob)
            labels.append(target.detach().cpu().numpy())


    flat_predictions_prob = np.array([item for sublist in pred_prob_list for item in sublist])
    flat_predictions = np.argmax(flat_predictions_prob, axis=1).flatten()
    flat_true_labels = [item for sublist in labels for item in sublist]
    results=Get_Results(flat_predictions_prob, flat_predictions, flat_true_labels, num_classes, start_time)
    return results

def Evaluate_Improved_Model(test_loader, model,model_xgb,num_classes, ensemble_method="average"):
    model.eval()    # switch to evaluate mode

    start_time = time.time()

    pred_list=[]
    pred_prob_list=[]
    labels=[]
    with torch.no_grad():
        #for i, (input, target, _) in enumerate(val_loader):
        for i, (input, target) in enumerate(test_loader):
            sl = input.shape
            batch_size = sl[0]
            target = target.cuda(non_blocking=True)
            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)

            # compute output
            output = model(input_var)
            pred=F.softmax(output, 1)
            prob=pred.detach().cpu().numpy()
            #XGBoost prob
            prob_xgb=model_xgb.predict_proba(input.detach().cpu().numpy())
            if ensemble_method=="average":#try average:
                if(prob.shape[0]==prob_xgb.shape[0] and prob.shape[1]==prob_xgb.shape[1]):
                    average_prob=0.5*(prob_xgb+prob)
                else:
                    average_prob=prob
            #save results
            pred_prob_list.append(average_prob)
            labels.append(target.detach().cpu().numpy())


    flat_predictions_prob = np.array([item for sublist in pred_prob_list for item in sublist])
    flat_predictions = np.argmax(flat_predictions_prob, axis=1).flatten()
    flat_true_labels = [item for sublist in labels for item in sublist]
    results=Get_Results(flat_predictions_prob, flat_predictions, flat_true_labels, num_classes, start_time)
    return results


def Get_Results(pred_prob, pred, y_true, num_classes, start_time):
    #np.seterr(divide='ignore', invalid='ignore') #ignore divide by zero warnings

    unique_labels=np.arange(num_classes)
    cnf_matrix = confusion_matrix(y_true, pred)

    FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)  
    FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
    TP = np.diag(cnf_matrix)
    TN = cnf_matrix.sum() - (FP + FN + TP)

    FP = FP.astype(float)
    FN = FN.astype(float)
    TP = TP.astype(float)
    TN = TN.astype(float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        TPR = np.nanmean(TP/(TP+FN)) #TPR
        #PPV = TP/(TP+FP) #Precision
        FPR = np.nanmean(FP/(FP+TN)) #FPR
        accuracy=accuracy_score(y_true, pred)
        roc_auc_count=0
        roc_list=[]
        pr_list=[]
        for i, label in enumerate(unique_labels):
            pr_list.append(average_precision_score((y_true==label)*1,pred_prob[:,i]))
            try:
                roc_list.append(roc_auc_score((y_true==label)*1,pred_prob[:,i]))
            except:
                continue
        pr_auc= np.nanmean(pr_list)
        roc_auc= np.nanmean(roc_list)
    prec=precision_score(y_true,pred,average="macro",zero_division=1)#precision
    end_time=time.time()
    diff=end_time-start_time
    inference_time=diff*1000/len(y_true)
    results={"Accuracy":accuracy, "ROC_AUC":roc_auc,"PR_AUC":pr_auc,"Precision":prec,"TPR":TPR,"FPR":FPR, "inference_time [seconds]":inference_time}
    return results

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 at [150, 225, 300] epochs"""

    boundary = [args.epochs // 2, args.epochs // 4 * 3, args.epochs]
    lr = args.lr * 0.1 ** int(bisect.bisect_left(boundary, epoch))
    # print(epoch, lr, bisect.bisect_left(boundary, epoch))
    # lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr


def adjust_learning_rate_adam(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 5 at [240] epochs"""

    boundary = [args.epochs // 5 * 4]
    lr = args.lr * 0.2 ** int(bisect.bisect_left(boundary, epoch))
    # print(epoch, lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr

