from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import auc, confusion_matrix, f1_score, roc_curve


def accuary_(y_pred, y_true):
    y_p = y_pred.reshape(-1)
    y_t = y_true.reshape(-1)
    return (y_p == y_t).sum()/y_p.shape[0]


def f1_score_(y_pred, y_true):
    return f1_score(y_true, y_pred, average='macro', zero_division=1),\
        f1_score(y_true, y_pred, average='micro', zero_division=1)


def confusion_matrix_(y_pred, y_true):
    return confusion_matrix(y_true.ravel(), y_pred.ravel(), labels=[0, 1, 2])


def roc_auc(y_pred, y_true, n_classes=3):
    '''
    依次为class0 ~ class(n-1) 与总micro
    '''

    y_pred = y_pred
    y_true = F.one_hot(torch.LongTensor(y_true).squeeze(),
                       n_classes).numpy()
    fpr, tpr, thresholds, roc_auc_micro = [], [], [], []
    for var in range(n_classes):
        fpr_t, tpr_t, thresholds_t = roc_curve(
            y_true[:, var], y_pred[:, var], pos_label=1)
        fpr.append(fpr_t), tpr.append(tpr_t), thresholds.append(thresholds_t)
        roc_auc_micro.append(auc(fpr_t, tpr_t))

    fpr_t, tpr_t, thresholds_t = roc_curve(
        y_true.ravel(), y_pred.ravel(), pos_label=1)
    fpr.append(fpr_t), tpr.append(tpr_t), thresholds.append(thresholds_t)
    roc_auc_micro.append(auc(fpr_t, tpr_t))
    fpr = np.array(fpr, dtype=object)
    tpr = np.array(tpr, dtype=object)
    thresholds = np.array(thresholds, dtype=object)
    roc_auc_micro = np.array(roc_auc_micro)
    return fpr, tpr, thresholds, roc_auc_micro


def Mereics_score(y_pred, y_true, n_classes=3):
    import warnings
    warnings.filterwarnings("ignore")
    """
    Getting the score of model, include these element
    accuracy, sensitivity, Specificity, precision F1-score
    AP, dice, iou and mAP
    more Mereics_score will append in the future
    """
    mereics_dict = OrderedDict()
    y_pred = y_pred.numpy()
    y_true = F.one_hot(y_true.squeeze(),
                       n_classes).numpy()
    mAP = 0
    fpr, tpr, thresholds = roc_curve(y_true, y_pred, pos_label=1)

    for idx in range(y_pred.shape[0]):

        prediction = y_pred[idx]
        groundtruth = y_true[idx]
        # intersection_area = ((prediction+groundtruth) > 0).sum()
        # mereics_dict['accuracy_' +
        #              str(idx+1)] = accuracy_score(groundtruth, prediction)
        # mereics_dict['precision_score_'+str(idx+1)] = precision_score(
        #     groundtruth, prediction, average='weighted', zero_division=1)
        # mereics_dict['recall_score_'+str(idx+1)] = recall_score(
        #     groundtruth, prediction, average='weighted', zero_division=1)
        mereics_dict['f1_score_macro_'+str(idx+1)] = f1_score(
            groundtruth, prediction, average='macro', zero_division=1)
        mereics_dict['f1_score_mirco_'+str(idx+1)] = f1_score(
            groundtruth, prediction, average='micro', zero_division=1)
    #     if intersection_area == 0:
    #         mereics_dict['AP_'+str(idx+1)] = 1
    #     else:
    #         mereics_dict['AP_'+str(idx+1)] = average_precision_score(
    #             groundtruth, prediction, average='weighted')
    #     mAP += mereics_dict['AP_'+str(idx+1)]
    # mereics_dict['mAP'] = mAP/y_pred.shape[0]
    return mereics_dict


if __name__ == '__main__':
    a = np.random.randint(0, 3, (512, 512))
    b = a
    print(Mereics_score(a, b))
