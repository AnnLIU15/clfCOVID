import argparse
from glob import glob


def main(args):
    train_dir, val_dir, test_dir = args.train_dir, args.val_dir, args.test_dir
    train_file = glob(train_dir+'*.npy')
    val_file = glob(val_dir+'*.npy')
    test_file = glob(test_dir+'*.npy')
    train_file = [var[len(train_dir):] for var in train_file]
    val_file = [var[len(val_dir):] for var in val_file]
    test_file = [var[len(test_dir):] for var in test_file]
    total_list = []
    total_list.extend(train_file)
    total_list.extend(val_file)
    total_list.extend(test_file)
    total_list = set(total_list)
    print(len(train_file))
    print(len(val_file))
    print(len(test_file))
    print(len(total_list))
    print(len(train_file)+len(val_file)+len(test_file))


if __name__ == '__main__':
    parser_ = argparse.ArgumentParser()
    parser_.add_argument('--train_dir', type=str,
                         default='data/process_clf/train/imgs/', help='train data dir')
    parser_.add_argument(
        '--val_dir', type=str, default='data/process_clf/val/imgs/', help='val data dir')

    parser_.add_argument('--test_dir', type=str,
                         default='data/process_clf/test/imgs/', help='test data dir')
    args = parser_.parse_args()
    main(args)
