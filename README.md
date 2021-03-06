# 新冠肺炎分类

**Verion2.0 Author: ZhaoY**

## 运行环境

| Version  | v1.0    20210611           |
| -------- | ------------------------------- |
| 编程语言 | Python                          |
| Cuda版本 | 10.0                            |
| 库       | [requirements](./requirement.txt) |

## 分割神经网络模型

| 网络名称     | 原始文章                                                     |
| ------------ | ------------------------------------------------------------ |
| [ResNet18](https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py) | [Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385.pdf) |

```flow
st=>start: start
rs=>operation: utils/rename_clf_pic
ps=>operation: utils/preprocessClf
fe=>operation: utils/findDismatchSize
de=>operation: utils/delDismatchSizeFile
tr=>operation: clfTrain
ts=>operation: clfTest
e=>end: end

st->rs->ps->fe->de->tr->ts->e
```



### 1. 分类(先看分割代码)

#### 预处理分类图片

```
python ./utils/preprocessClf.py --in_dir /home/e201cv/Desktop/covid_data/clf/train /home/e201cv/Desktop/covid_data/clf/val /home/e201cv/Desktop/covid_data/clf/test --out_dir /home/e201cv/Desktop/covid_data/process_clf/train /home/e201cv/Desktop/covid_data/process_clf/val /home/e201cv/Desktop/covid_data/process_clf/test
```

#### 获取掩模

```
python infer.py --infer_data_dirs /home/e201cv/Desktop/covid_data/process_clf/train /home/e201cv/Desktop/covid_data/process_clf/val /home/e201cv/Desktop/covid_data/process_clf/test --pth /home/e201cv/Desktop/covid_seg/output/saved_models/U2Net/bestSeg.pth --num_classes 3 --device cuda
```

#### 训练1

```
/home/e201cv/.conda/envs/pt171/bin/python --device cuda --num_classes 3 --radiomics_require False --match False --batch_size 64 --num_epochs 155 --lrate 1e-3 --model_name resnet18

/home/e201cv/.conda/envs/pt171/bin/python clfTrain.py --device cuda --num_classes 3 --radiomics_require False --match False --batch_size 32 --num_epochs 155 --lrate 1e-3 --model_name resnet34

/home/e201cv/.conda/envs/pt171/bin/python clfTrain.py --device cuda --num_classes 3 --radiomics_require False --match False --batch_size 16 --num_epochs 155 --lrate 1e-3 --model_name resnet50

/home/e201cv/.conda/envs/pt171/bin/python clfTrain.py --device cuda --num_classes 3 --radiomics_require False --match False --batch_size 64 --num_epochs 155 --lrate 1e-3 --model_name vgg11

python clfTrain.py --device cuda --num_classes 3 --radiomics_require False --match False --batch_size 12 --num_epochs 80 --lrate 1e-3  --train_data_dir data/process_clf/train/ --val_data_dir data/process_clf/val/ --model_name vgg19
```

#### 测试

```
python clfTest.py --num_classes 3 --device cuda --pth output/saved_models/ClfBestModel.pth --batch_size 256 --radiomics_require False --test_data_dir ./data/process_clf/test --save_clf ./output/clfResult/

python clfTest.py --num_classes 3 --device cuda --pth output/saved_models/resnet18/epoch_150_model.pth --batch_size 256 --radiomics_require False --test_data_dir ./data/process_clf/test --save_clf ./output/clfResult/ --model_name resnet18
```

#### ROC曲线与具体评测效果显示
```
python plt_roc.py
python cmp2file.py
```

#### log

```
tensorboard --logdir='log/clf/resnet0607-1902'
```

#### 外源测试(x)

```
python extendTest.py --num_classes 3 --device cuda --pth output/saved_models/ClfBestModel.pth --batch_size 1 --test_data_dir ./data/extend --save_clf ./output/clfResult/ --model_name resnet_ex
```

#### 看学习率
```
/home/e201cv/.conda/envs/pt171/bin/python /home/e201cv/Desktop/COVID_prj/Code/covid_clf/watchLrate.py -m output/saved_models/resnet/epoch_140_model.pth
```
