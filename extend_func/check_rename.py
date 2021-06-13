import os
import numpy as np
from glob import glob


if __name__ == '__main__':
    file_path='2A_images_process/'
    files=glob(file_path+'/*.npy')
    cnt=0
    for file in files:
        if '0'==file[file.rfind('/')+1] or '1'==file[file.rfind('/')+1] or '2'==file[file.rfind('/')+1] :
            cnt+=1
    print(len(files)==cnt)