import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import auc, confusion_matrix, f1_score, roc_curve


def accuary_(y_pred, y_true):
    '''
    return the accuary of the test
    '''
    y_p = y_pred.reshape(-1)
    y_t = y_true.reshape(-1)
    return (y_p == y_t).sum()/y_p.shape[0]


def f1_score_(y_pred, y_true):
    '''
    return the f1-score(macro and micro)
    '''
    return f1_score(y_true, y_pred, average='macro', zero_division=1),\
        f1_score(y_true, y_pred, average='micro', zero_division=1)


def confusion_matrix_(y_pred, y_true):
    '''
    return the confusion_matrix of this test
    '''
    return confusion_matrix(y_true.ravel(), y_pred.ravel(), labels=[0, 1, 2])


def roc_auc(y_pred, y_true, n_classes=3):
    '''
    依次为class0 ~ class(n-1) 与总micro
    total shape is fpr/tpr->(1,n_classes+1,)
    '''
    y_pred = y_pred
    y_true = F.one_hot(torch.LongTensor(y_true).squeeze(),
                       n_classes).numpy()
    # one hot -> be some two classes questions
    fpr, tpr, thresholds, roc_auc_micro = [], [], [], []
    for var in range(n_classes):
        # get each class roc and auc
        fpr_t, tpr_t, thresholds_t = roc_curve(
            y_true[:, var], y_pred[:, var], pos_label=1)
        fpr.append(fpr_t), tpr.append(tpr_t), thresholds.append(thresholds_t)
        roc_auc_micro.append(auc(fpr_t, tpr_t))
    # get total roc and auc
    fpr_t, tpr_t, thresholds_t = roc_curve(
        y_true.ravel(), y_pred.ravel(), pos_label=1)
    fpr.append(fpr_t), tpr.append(tpr_t), thresholds.append(thresholds_t)
    roc_auc_micro.append(auc(fpr_t, tpr_t))
    fpr = np.array(fpr, dtype=object)
    tpr = np.array(tpr, dtype=object)
    thresholds = np.array(thresholds, dtype=object)
    roc_auc_micro = np.array(roc_auc_micro)
    return fpr, tpr, thresholds, roc_auc_micro

