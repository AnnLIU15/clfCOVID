from glob import glob

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


def getImage(dataset_path):
    '''
    获取当前目录下的所有pic_type格式的图片
    '''
    pics = sorted(glob(dataset_path+'/*.npy'))
    pic_data = []
    pic_name = []
    length_path = len(dataset_path)+1

    for pic in pics:
        pic_ = np.load(pic)
        pic_data.append(pic_)
        pic_name.append(pic[length_path:-4])

    return pic_data, pic_name


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

    def __init__(self, dataset_path, match=True, radiomics_feature=[1032, 1]):
        super(clfDataSet, self).__init__()
        self.dataset_path = dataset_path
        self.radiomics_feature = radiomics_feature
        self.match = match
        if match:
            self.imgs_data, self.radiomics_path, self.imgs_name = self.__get_dataset_match()
        else:
            self.imgs_data, self.imgs_name = self.__get_dataset_match()

    def __getitem__(self, idx):
        '''
        return imgs,labels,radiomics,imgs_name
        '''
        imgs_data = torch.FloatTensor(self.imgs_data[idx]).unsqueeze(0)
        imgs_name_ = self.imgs_name[idx]
        # acquire labels
        labels = int(imgs_name_[0])
        if self.match:
            radiomics_data = torch.FloatTensor(
                np.load(self.radiomics_path[idx]))
        else:
            radiomics_data = torch.zeros(
                size=(self.radiomics_feature[0], self.radiomics_feature[1])).squeeze().float()
        return imgs_data, labels, radiomics_data, imgs_name_

    def __len__(self):
        return len(self.imgs_name)

    def __get_dataset_match(self):

        imgs_path_dir = self.dataset_path+'/imgs'
        radiomics_path_dir = self.dataset_path+'/radiomics'
        imgs_pic, imgs_name = getImage(imgs_path_dir)
        print('imgs dir:',imgs_path_dir,'imgs_num:',len(imgs_name))
        radiomics_path = sorted(glob(radiomics_path_dir+'/*.npy'))
        radiomics_name = [var[len(radiomics_path_dir)+1:-4]
                          for var in radiomics_path]
        print('radiomics dir:',radiomics_path_dir,'radiomics num:',len(radiomics_name))
        if self.match:
            sum_of_error = 0
            try:
                if not len(imgs_name) == len(radiomics_name):
                    raise ValueError("图片与影像组学数量对应不上")
            except ValueError as e:
                print("引发异常：", repr(e))
                sum_of_error += 1
            try:
                if not imgs_name == radiomics_name:
                    raise ValueError("图片与影像组学对应不上")
            except ValueError as e:
                print("引发异常：", repr(e))
                sum_of_error += 1
            if not sum_of_error == 0:
                raise RuntimeError('存在错误')
            return imgs_pic, radiomics_path, imgs_name
        else:
            return imgs_pic, imgs_name


if __name__ == '__main__':
    dataset = clfDataSet('data/process_clf/train', 0)
    dataset_v = clfDataSet('data/process_clf/val', 0)
    data_loader = DataLoader(
        dataset=dataset, batch_size=16, num_workers=8, shuffle=True, drop_last=False)
    print('loaded')
    data_loader_v = DataLoader(
        dataset=dataset_v, batch_size=16, num_workers=8, shuffle=True, drop_last=False)
    print('loaded')
    try:
        for batch_idx, (data, pic_class, radiomics_, name_) in enumerate(data_loader):
            print(batch_idx, data.shape, pic_class.shape, radiomics_.shape)
    except RuntimeError as e:
        print('error')
