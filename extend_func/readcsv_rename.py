import os
import numpy as np
from glob import glob


if __name__ == '__main__':
    file_path='2A_images_process/'
    with open('pre.csv','r') as f:
        vars_from_file=f.readlines()
    vars_from_file=[var.replace('\n','').split(',') for var in vars_from_file][1:]
    data_name_list,type_pic_list=[],[]
    type_ndarray=np.array([],dtype=np.uint8)
    for data_name,type_pic in vars_from_file:
        data_name_list.append(data_name)
        type_pic_list.append(type_pic)
    for var in type_pic_list:
        if 'Normal'==var:
            type_ndarray=np.hstack((type_ndarray,0))
        elif 'Pneumonia'==var:
            type_ndarray=np.hstack((type_ndarray,1))
        else:
            type_ndarray=np.hstack((type_ndarray,2))
    files=sorted(glob(file_path+'/*.npy'))
    cnt=0
    for file in files:
        # print(file[len(file_path):])

        for idx,data_name in enumerate(data_name_list):
            if data_name == file[len(file_path):len(file_path)+len(data_name)]:
                dst_name=file[:file.rfind('/')+1]+str(type_ndarray[idx])+'_'+file[file.rfind('/')+1:]
                print('src:',file,'->','dst:',dst_name)
                cnt+=1
                os.rename(src=file,dst=dst_name)
                break
                        
            elif idx==len(data_name_list)-1:
                if 'normal' in file:
                    dst_name=file[:file.rfind('/')+1]+'0_'+file[file.rfind('/')+1:]
                elif 'covid' in file:
                    dst_name=file[:file.rfind('/')+1]+'2_'+file[file.rfind('/')+1:]
                else:
                    assert 1
                print('src:',file,'->','dst:',dst_name)
                os.rename(src=file,dst=dst_name)
                cnt+=1

    print('cnt',cnt,'len',len(files))
