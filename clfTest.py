import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from clfConfig import getConfig
from datasets.clfDataSet import clfDataSet
from models.resnet import resnet18,resnet34,resnet50
from utils.Metrics import accuary_, confusion_matrix_, f1_score_, roc_auc
from models.vgg import vgg11_bn,vgg19_bn


def test(model, test_loader, device, radiomics_require=False):
    model.eval()
    labels_list = np.array([], dtype=np.uint8)
    argmax_output_list = np.array([], dtype=np.uint8)
    with torch.no_grad():
        for idx, (imgs, labels, _) in tqdm(enumerate(test_loader), desc='Train', total=len(test_loader)):
            imgs, labels = imgs.to(device), labels.to(device)
            output = model(imgs)

            # origin label
            labels_list = np.hstack(
                (labels_list, labels.clone().detach().cpu().numpy().reshape(-1)))

            # predict score
            if idx == 0:
                output_list = output.clone().detach().cpu().numpy()
            else:
                output_list = np.vstack(
                    (output_list, output.clone().detach().cpu().numpy()))

            # score's argmax ->0 1 2
            argmax_output_list = np.hstack((argmax_output_list, (torch.nn.Softmax(
                dim=1)(output)).argmax(dim=1).clone().detach().cpu().numpy().reshape(-1)))

            # release cache
            torch.cuda.empty_cache()
    # acquire accuracy f1-score confusion_matrix and ROC(AUC)
    total_accuracy = accuary_(argmax_output_list, labels_list)

    total_f1_score_macro, total_f1_score_mirco = f1_score_(
        argmax_output_list, labels_list)
    total_cm = confusion_matrix_(argmax_output_list, labels_list)
    fpr, tpr, thresholds, roc_auc_micro = roc_auc(output_list, labels_list, 3)
    '''
    save result
    '''
    np.save('./output/pred.npy', argmax_output_list)
    np.save('./output/true.npy', labels_list)
    return fpr, tpr, thresholds, roc_auc_micro, total_accuracy, total_f1_score_macro, total_f1_score_mirco, total_cm


def main(args):

    match, device, num_classes, batch_size,  model_name =\
        args.match, args.device, args.num_classes, args.batch_size, args.model_name
    radiomics_require, pth, save_clf, test_data_dir = \
        args.radiomics_require, args.pth, args.save_clf, args.test_data_dir
    save_dir = save_clf+'/'+model_name
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    ng = torch.cuda.device_count()
    print("Available cuda Devices:{}".format(ng))
    for i in range(ng):
        print('device%d:' % i, end='')
        print(torch.cuda.get_device_properties(i))

    if device == 'cuda':
        torch.cuda.set_device(0)
        if not torch.cuda.is_available():
            print('Cuda is not available, use CPU to train.')
            device = 'cpu'
    device = torch.device(device)
    print('===>device:', device)
    torch.cuda.manual_seed_all(0)

    print('===>Setup Model')
    # model = EfficientNet(tf=1, in_channels=1,
    #                      num_class=num_classes).to(device)
    if model_name=='resnet34':
        model = resnet34(pretrained=False, num_classes=num_classes).to(device)
    elif model_name=='resnet50':
        model = resnet50(pretrained=False, num_classes=num_classes).to(device) 
    elif model_name=='vgg11':
        model=vgg11_bn(pretrained=False, num_classes=num_classes).to(device) 
    elif model_name=='vgg19':
        model=vgg19_bn(pretrained=False, num_classes=num_classes).to(device)        
    else:
        model = resnet18(pretrained=False, num_classes=num_classes).to(device)
    # a = summary(model=model, input_size=(
    #     1, 512, 512), batch_size=4, device='cuda')
    # model=efficientnetv2_s(in_channels=1,num_classes=num_classes).to(device)
    # print('*'*30)
    # summary(model=model,input_size=(1,512,512),batch_size=4,device='cuda')

    print('===>Setting optimizer and scheduler')

    if not pth == None:
        print('===>Loading Pretrained Model')
        checkpoint = torch.load(pth)
        model.load_state_dict(checkpoint['model_weights'])
    else:
        exit('no trained model')

    print('===>Loading dataset')

    require = True if (match and radiomics_require) else False

    test_data_loader = DataLoader(
        dataset=clfDataSet(test_data_dir, match=require), batch_size=batch_size,
        num_workers=8, shuffle=True, drop_last=False)
    print('test_data_dir:', test_data_dir)
    print('test_data_loader data:', len(test_data_loader))
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    test_begin_time = time.time()
    # val_loss =
    fpr, tpr, _, roc_auc_micro, total_accuracy, total_f1_score_macro, total_f1_score_mirco, total_cm = test(
        model=model, test_loader=test_data_loader, device=device, radiomics_require=radiomics_require)
    roc_mt = np.vstack((fpr, tpr))

    np.save(file=save_dir+'/roc.npy', arr=roc_mt)
    np.save(file=save_dir+'/auc.npy', arr=roc_auc_micro)
    with open(save_dir+'/metrics.txt', 'w+') as f:
        print('accuracy', 'f1_score_macro', 'f1_score_mirco', file=f)
        print(total_accuracy, total_f1_score_macro,
              total_f1_score_mirco, file=f)

    with open(save_dir+'/confusion_matrix.txt', 'w+') as f:
        print(total_cm, file=f)
    print('This test cost %ds.' % (time.time()-test_begin_time))


if __name__ == '__main__':
    '''
    similar to the clfTrain's comment 
    '''
    args = getConfig('test')
    main(args)
