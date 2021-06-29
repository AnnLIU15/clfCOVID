## 指标

| model name | accuarcy | f1 macro | f1 micro | AUC marco | AUC mirco |
| ---------- | -------- | -------- | -------- | --------- | --------- |
| ResNet18   | 0.9953   | 0.9945   | 0.9953   | 0.9998    | 0.9998    |
| ResNet34   | 0.9916   | 0.9906   | 0.9916   | 0.9998    | 0.9998    |
| ResNet50   | 0.9927   | 0.9921   | 0.9927   | 0.9992    | 0.9994    |
| vgg11      | 0.9919   | 0.9914   | 0.9919   | 0.9994    | 0.9996    |
| vgg19      | 0.971    | 0.970    | 0.971    | 0.9975    | 0.9979    |

### 混淆矩阵

#### ResNet18

| Normal    | 9440   | 10   | 0    |
| --------- | ------ | ---- | ---- |
| CP        | 20     | 7268 | 18   |
| NCP       | 3      | 49   | 4294 |
| true/pred | Normal | CP   | NCP  |

heatmap

| 0.9975   | 0.00136 | 0       |
| -------- | ------- | ------- |
| 0.00214  | 0.9919  | 0.00417 |
| 0.000317 | 0.00669 | 0.9958  |



#### ResNet34

| Normal    | 9394   | 44   | 12   |
| --------- | ------ | ---- | ---- |
| CP        | 21     | 7246 | 39   |
| NCP       | 5      | 56   | 4285 |
| true/pred | Normal | CP   | NCP  |

heatmap

| 0.997    | 0.00599 | 0.00277 |
| -------- | ------- | ------- |
| 0.00223  | 0.9864  | 0.00899 |
| 0.000531 | 0.00762 | 0.9882  |



#### ResNet50

| Normal    | 9413   | 37   | 0    |
| --------- | ------ | ---- | ---- |
| CP        | 36     | 7255 | 15   |
| NCP       | 5      | 61   | 4280 |
| true/pred | Normal | CP   | NCP  |

heatmap

| 0.9957  | 0.00503 | 0      |
| ------- | ------- | ------ |
| 0.00381 | 0.9867  | 0.0035 |
| 0.00053 | 0.0083  | 0.9965 |



#### Vgg11(with batchnorm)

| Normal    | 9423   | 27   | 0    |
| --------- | ------ | ---- | ---- |
| CP        | 56     | 7243 | 7    |
| NCP       | 24     | 57   | 4265 |
| true/pred | Normal | CP   | NCP  |

heatmap

| 0.9916 | 0.0037 | 0      |
| ------ | ------ | ------ |
| 0.0059 | 0.9885 | 0.0016 |
| 0.0025 | 0.0078 | 0.9983 |





#### Vgg19(with batchnorm)

| Normal    | 9210   | 236  | 4    |
| --------- | ------ | ---- | ---- |
| CP        | 118    | 7075 | 113  |
| NCP       | 37     | 112  | 4197 |
| true/pred | Normal | CP   | NCP  |

heatmap

| 0.9834  | 0.0318 | 0.000927 |
| ------- | ------ | -------- |
| 0.0126  | 0.9531 | 0.0262   |
| 0.00395 | 0.0151 | 0.9729   |
