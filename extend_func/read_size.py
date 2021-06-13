import cv2
from glob import glob
from tqdm import tqdm

if __name__ == '__main__':
    files=glob('2A_images/*')
    save_file_dir='./file_size.txt'
    cnt=0
    with open(save_file_dir,'w+') as f:
        for file in tqdm(files,desc='Reading',total=len(files)):
            pic=cv2.imread(file, cv2.IMREAD_GRAYSCALE)
            print(file,pic.shape,file=f)
            if not (512,512)==pic.shape:
                cnt+=1
        print('files len:',len(files),'del_len:',cnt)
