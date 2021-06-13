# 新冠肺炎分类

**Verion1.0 Author: ZhaoY**

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

#### 获取影像组学
```
python Radiomics/exact_radiomics.py --imgs_dir data/clf/train/imgs data/clf/val/imgs data/clf/test/imgs --masks_dir data/clf/train/masks data/clf/val/masks data/clf/test/masks --out_dir data/clf/train/radiomics data/clf/val/radiomics data/clf/test/radiomics
```

#### 训练1

```
python /home/e201cv/Desktop/covid_clf/clfTrain.py --device cuda --num_classes 3 --model_name resnet --radiomics_require False --match False --batch_size 256 --num_epochs 200
```

#### 测试
```
 python clfTest.py --num_classes 3 --device cuda --pth output/saved_models/ClfBestModel.pth --batch_size 256 --radiomics_require False --test_data_dir ./data/process_clf/test --save_clf ./output/clfResult/
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



#### 性能指标

confusion matrix



| Normal | 9381   | 68   | 1    |
| ------ | ------ | ---- | ---- |
| CP     | 50     | 7238 | 18   |
| NCP    | 22     | 63   | 4261 |
|        | Normal | CP   | NCP  |

heatmap

| 0.992 | 0.009 | 0.0002  |
| ----- | ----- | ------- |
| 0.005 | 0.982 | 0.0042  |
| 0.002 | 0.008 | 0.99556 |