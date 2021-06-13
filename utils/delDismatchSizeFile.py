import os
import re
from glob import glob


def readfileAndDel(file, f1):
    '''
    read the utils/findDismatchSize.py's output file,
    del the file(include imgs masks radiomics) which match the key words in the txt
    '''
    with open(file, 'r') as f:
        file_content = f.readlines()
    imgs_files = []
    for file_c in file_content:
        for m in re.finditer('npy', file_c):
            imgs_files.append(file_c[:m.end()])
    masks_files = [var.replace('imgs', 'masks') for var in imgs_files]
    radiomics_files = [var.replace('imgs', 'radiomics') for var in imgs_files]
    for idx, _ in enumerate(imgs_files):
        if os.path.exists(imgs_files[idx]):
            # remove the imgs
            os.remove(imgs_files[idx])
            print('idx: %d  file_type: img   file_name: %s' %
                  (idx, imgs_files[idx]), file=f1)
        if os.path.exists(masks_files[idx]):
            os.remove(masks_files[idx])
            # remove the masks
            print('idx: %d  file_type: msk   file_name: %s' %
                  (idx, masks_files[idx]), file=f1)
        if os.path.exists(radiomics_files[idx]):
            # remove the radiomics
            os.remove(radiomics_files[idx])
            print('idx: %d  file_type: rad   file_name: %s' %
                  (idx, radiomics_files[idx]), file=f1)


if __name__ == '__main__':
    files = glob('output/error_data/*.txt')
    del_file = './output/error_data/del_file.txt'
    if os.path.exists(del_file):
        del_file = './output/error_data/del_file1.txt'
    with open(del_file, 'w+') as f1:
        for file in files:
            readfileAndDel(file, f1)
