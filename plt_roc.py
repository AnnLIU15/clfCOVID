import argparse

import matplotlib.pyplot as plt
import numpy as np
from scipy import interp
from sklearn.metrics import auc


def main(args):
    acc_ = np.load(args.auc_path)
    f_tpr = np.load(args.roc_path, allow_pickle=True)
    name_list = ['normal', 'cp', 'ncp', 'total_micro', 'total_macro']
    color_list = ['b-', 'r--', 'y-.', 'g:','m^']
    print('acc class', acc_.shape)
    print('f_tpr class', f_tpr.shape)
    fig = plt.figure(1, figsize=(10, 10))
    plt.grid(1)
    plt.xlabel('fpr')
    plt.ylabel('tpr')
    plt.title('multiclass ROC curve')
    for i in range(f_tpr.shape[1]):
        print(f_tpr[0, i].shape)
        plt.plot(f_tpr[0, i], f_tpr[1, i], color_list[i],
                 label=name_list[i]+' AUC=%.4f' % acc_[i])
    all_fpr = np.unique(np.concatenate([f_tpr[0, i] for i in range(f_tpr.shape[1]-1)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(f_tpr.shape[1]-1):
        mean_tpr += interp(all_fpr, f_tpr[0, i], f_tpr[1, i])
    # Finally average it and compute AUC
    mean_tpr /= f_tpr.shape[1]-1
    fpr = all_fpr
    tpr = mean_tpr
    roc_auc = auc(fpr, tpr)
    print(all_fpr.shape)
    plt.plot(fpr, tpr, color_list[f_tpr.shape[1]],
                 label=name_list[f_tpr.shape[1]]+' AUC=%.4f' % roc_auc)
    
    plt.legend()
    #    plt.legend()要在show前一句
    plt.savefig(args.save_pic_dir)
    plt.close()


if __name__ == '__main__':
    parser_ = argparse.ArgumentParser()
    parser_.add_argument('--auc_path', type=str,
                         default='output/clfResult/resnet/auc.npy')
    parser_.add_argument('--roc_path', type=str,
                         default='output/clfResult/resnet/roc.npy')
    parser_.add_argument('--save_pic_dir', type=str,
                         default='./output/clfResult/resnet/ROC.eps')
    args = parser_.parse_args()

    main(args)
