import argparse


def getConfig(stage):
    parser_ = argparse.ArgumentParser(
        description='训练的参数')

    parser_.add_argument("--device", type=str, default='cuda')
    parser_.add_argument("--num_classes", type=int, default=3,
                         help="pic classes,3->ncov(2) cp(1) normal(0)")
    parser_.add_argument("--model_name", type=str, default='resnet')
    parser_.add_argument("--pth", type=str, default=None,
                         help="训练好的pth路径，模型必须包含以下参数"
                         "model_weights, optimizer_state")
    parser_.add_argument("--radiomics_require", type=bool, default=False)
    parser_.add_argument("--match", type=bool, default=False)
    parser_.add_argument("--batch_size", type=int,
                         default=16, help='batch_size')
    if stage == "train":

        parser_.add_argument("--start_epoch", type=str, default=1)
        parser_.add_argument("--num_epochs", type=int, default=100)

        parser_.add_argument("--save_dir", type=str, default="./output/saved_models",
                             help="Directory to save checkpoints")
        parser_.add_argument("--train_data_dir", type=str, default='./data/process_clf/train',
                             help="Path to the training data.")
        parser_.add_argument("--val_data_dir", type=str, default='./data/process_clf/val',
                             help="Path to the validation data.")
        # parser_.add_argument("--batch_size", type=int, default=8, help="Implemented only for batch size = 1")
        parser_.add_argument("--save_every", type=int, default=10)
        parser_.add_argument("--lrate", type=float,
                             default=1e-3, help="init Learning rate")
        parser_.add_argument('--log_name', type=str,
                             default=None, help='中断后继续训练记载')

    elif stage == "test":
        parser_.add_argument("--test_data_dir", type=str, default='./data/process_clf/test',
                             help="Path to the test data.")
        parser_.add_argument("--save_clf", type=str,
                             default='./output/clfResult/')
    elif stage == "infer":
        parser_.add_argument("--infer_data_dirs", type=str, nargs='+', default=['/home/e201cv/Desktop/covid_data/process_clf/train',
                                                                                '/home/e201cv/Desktop/covid_data/process_clf/val', 
                                                                                '/home/e201cv/Desktop/covid_data/process_clf/test'],
                             help="Path to the test data.")
    model_args = parser_.parse_args()
    return model_args
