
import numpy as np

if __name__ == '__main__':
    '''
    判断预测值和真实值，找出那些图片错误(基本没有用，可删)
    '''
    pred = np.load('./output/pred.npy')
    true_ = np.load('./output/true.npy')
    with open('./output/cmp.txt', 'w+') as f:
        print('true', 'pred', '\t', 'equal', file=f)
        for idx, _ in enumerate(pred):
            print(true_[idx], '\t', pred[idx], '\t\t', 500 *
                  (not true_[idx] == pred[idx]), file=f)
