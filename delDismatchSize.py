from glob import glob
import os
from tqdm import tqdm
import numpy as np
from multiprocessing import Process
import argparse
def finderror(imgs_path,name,idx_name):
    file_list=[]
    data_list=[]
    for var in imgs_path:
        shape_var=np.load(var).shape
        if not shape_var==(512,512):
            file_list.append(var)
            data_list.append(shape_var)
            print('%d detect %s!'%(idx_name,var))
    with open(name,'w+') as f:
        for idx,_ in enumerate(file_list):
            print(file_list[idx],data_list[idx],file=f)

if __name__ == '__main__':
    parser_ = argparse.ArgumentParser()
    parser_.add_argument('--train_path', type=str,
                         default='./data/process_clf/train/imgs/*.npy', help='save rename pic')
    parser_.add_argument('--val_path', type=str,
                         default='./data/process_clf/val/imgs/*.npy', help='save rename pic')
    parser_.add_argument('--test_path', type=str,
                         default='./data/process_clf/test/imgs/*.npy', help='save rename pic')
    parser_.add_argument('--save_path', type=str,
                         default='./output/error_data/', help='save rename pic')
    args = parser_.parse_args()
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    
    train_path=sorted(glob(args.train_path))
    val_path=sorted(glob(args.train_path))
    test_path=sorted(glob(args.train_path))
    process_list=[]
    total_list=train_path+val_path+test_path
    del train_path,val_path,test_path
    l=len(total_list)

    process_list.append(Process(target=finderror, args=(
            total_list[:l//4],args.save_path+'/error_1.txt',1,)))
    process_list.append(Process(target=finderror, args=(
            total_list[l//4:l//4*2],args.save_path+'/error_2.txt',2,)))
    process_list.append(Process(target=finderror, args=(
            total_list[l//4*2:l//4*3],args.save_path+'/error_3.txt',3,)))
    process_list.append(Process(target=finderror, args=(
            total_list[l//4*3:],args.save_path+'/error_4.txt',4,)))
    
    
    for idx,process_ in enumerate(process_list):
        print('process %d start!'%idx)
        process_.start()
    # file_list,data_list=finderror(imgs_path)
    # print(file_list)
    # print(data_list)