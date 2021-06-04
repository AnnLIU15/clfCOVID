# 新冠肺炎分类

## 运行环境

| Version  | v0.1    20210604           |
| -------- | ------------------------------- |
| 编程语言 | Python                          |
| Cuda版本 | 10.0                            |
| 库       | [requirements](./requirement.txt) |

## 分割神经网络模型

| 网络名称     | 原始文章                                                     |
| ------------ | ------------------------------------------------------------ |
| EfficientNet | [EfficientNetV2: Smaller Models and Faster Training](https://arxiv.org/abs/2104.00298) |

| 网络代码 |
| -------- |
|          |

### 1. 分类(先看分割代码)

#### 预处理分类图片

```
python ./utils/preprocessClf.py --in_dir /home/e201cv/Desktop/covid_data/clf/train /home/e201cv/Desktop/covid_data/clf/val /home/e201cv/Desktop/covid_data/clf/test --out_dir /home/e201cv/Desktop/covid_data/process_clf/train /home/e201cv/Desktop/covid_data/process_clf/val /home/e201cv/Desktop/covid_data/process_clf/test
```

#### 获取掩模

```
python infer.py --infer_data_dirs /home/e201cv/Desktop/covid_data/process_clf/train /home/e201cv/Desktop/covid_data/process_clf/val /home/e201cv/Desktop/covid_data/process_clf/test --pth output/saved_models/U2Net_n/1_20_20_2_best_in_150/epoch_150_model.pth --num_classes 3 --device cuda
```

#### 获取影像组学
```
python Radiomics/exact_radiomics.py --imgs_dir data/clf/train/imgs data/clf/val/imgs data/clf/test/imgs --masks_dir data/clf/train/masks data/clf/val/masks data/clf/test/masks --out_dir data/clf/train/radiomics data/clf/val/radiomics data/clf/test/radiomics
```

#### 训练1

```
python /home/e201cv/Desktop/covid_clf/clfTrain.py --device cuda --num_classes 3 --model_name EfficientNet_s --radiomics_require False --match False --batch_size 8 --num_epochs 200
```

