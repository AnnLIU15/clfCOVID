import shutil
from glob import glob


if __name__ == '__main__':
    files=glob('2A_images/*')
    print('files len:',len(files))
    cnt=0
    for file in files:
        if ('CP' in file) or ('Normal' in file):
            shutil.move(src=file,dst='./2019nCOV')
            cnt+=1
            print('src:',file,'dst:','./2019nCOV')
    print('total numbers:',cnt)
    