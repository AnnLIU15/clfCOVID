from glob import glob
import torch
import numpy as np
from torch.utils.data import Dataset,DataLoader

class clfDataSet(Dataset):
    '''
    CNCB数据库
    dataset_path为数据路径
    目录结构为
    >dataset_path
        >imgs
            >xxxx.jpg
        >masks
            >xxxx.png
    '''

    def __init__(self, dataset_path,match=True,radiomics_feature=1032):
        super(clfDataSet, self).__init__()
        self.dataset_path = dataset_path
        self.match=match
        if match:
            self.imgs_path,self.radiomics_path,self.imgs_name=self.__get_dataset_match()
        else:
            self.imgs_path,self.imgs_name=self.__get_dataset_match()
    def __getitem__(self, idx):
        imgs_data=torch.FloatTensor(np.load(self.imgs_path[idx])).unsqueeze(0)
        imgs_name_=self.imgs_name[idx]
        labels=int(imgs_name_[0])
        if self.match:
            radiomics_data=torch.FloatTensor(np.load(self.radiomics_path[idx]))
        else:
            radiomics_data=torch.zeros(size=(1032,1)).squeeze().float()
        return imgs_data,labels,radiomics_data, imgs_name_

    def __len__(self):
        return len(self.imgs_name)

    def __get_dataset_match(self):
        imgs_path_dir=self.dataset_path+'/imgs'
        radiomics_path_dir=self.dataset_path+'/radiomics'
        imgs_path=sorted(glob(imgs_path_dir+'/*.npy'))
        radiomics_path=sorted(glob(radiomics_path_dir+'/*.npy'))
        imgs_name=[var[len(imgs_path_dir)+1:] for var in imgs_path]
        radiomics_name=[var[len(radiomics_path_dir)+1:] for var in radiomics_path]
        if self.match:
            sum_of_error=0
            try:
                if not len(imgs_name)==len(radiomics_name):
                    raise ValueError("图片与影像组学数量对应不上")
            except ValueError as e:
                print("引发异常：",repr(e))
                sum_of_error+=1
            try:
                if not imgs_name==radiomics_name:
                    raise ValueError("图片与掩膜对应不上")
            except ValueError as e:
                print("引发异常：",repr(e))
                sum_of_error+=1
            if not sum_of_error==0:
                raise RuntimeError('存在错误')
            return imgs_path,radiomics_path,imgs_name
        else:
            return imgs_path,imgs_name
      
if __name__ == '__main__':
    dataset = clfDataSet('data/process_clf/train',0)
    data_loader = DataLoader(
        dataset=dataset, batch_size=16, num_workers=8, shuffle=True, drop_last=False)
    try:
        for batch_idx, (data, pic_class,radiomics_,name_) in enumerate(data_loader):
            print(batch_idx, data.shape,pic_class.shape, radiomics_.shape)
    except RuntimeError as e:
        print(batch_idx,name_)