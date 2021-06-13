from glob import glob

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


def getImageName(dataset_path):
    '''
    获取当前目录下的所有pic_type格式的图片
    '''
    pics = sorted(glob(dataset_path+'/*.npy'))
    pic_name = []
    length_path = len(dataset_path)+1

    for pic in pics:
        pic_name.append(pic[length_path:-4])

    return pic_name, pics


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

    def __init__(self, dataset_path):
        super(clfDataSet, self).__init__()
        self.dataset_path = dataset_path
        self.imgs_name, self.imgs_path = self.__get_dataset_match()

    def __getitem__(self, idx):
        '''
        return imgs,labels,radiomics,imgs_name
        '''
        imgs_data = torch.FloatTensor(
            np.load(self.imgs_path[idx])).unsqueeze(0)
        imgs_name_ = self.imgs_name[idx]
        # acquire labels
        labels = int(imgs_name_[0])

        return imgs_data, labels, imgs_name_

    def __len__(self):
        return len(self.imgs_name)

    def __get_dataset_match(self):
        imgs_name, imgs_path = getImageName(self.dataset_path)
        print('imgs dir:', self.dataset_path, 'imgs_num:', len(imgs_name))

        return imgs_name, imgs_path
if __name__ == '__main__':
    dataset = clfDataSet('data/extend')
    data_loader = DataLoader(
        dataset=dataset, batch_size=1, num_workers=8, shuffle=True, drop_last=False)
    print('loaded')
    
    for batch_idx, (imgs_data, labels, imgs_name_) in enumerate(data_loader):
        print(batch_idx, imgs_data.shape, labels, imgs_name_)
