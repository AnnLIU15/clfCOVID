import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchsummary import summary
from tqdm import tqdm

from clfConfig import getConfig
from datasets.clfDataSet import clfDataSet
from models.EfficientNet import EfficientNet
from models.EfficientNetV2 import efficientnetv2_s
from models.resnet import resnet18
from utils.Metrics import Mereics_score, accuary_,f1_score_,confusion_matrix_,roc_


def test(model, test_loader, device, radiomics_require=False):
    total_accuracy = 0
    total_f1_score_macro=0
    total_f1_score_mirco=0
    total_cm=np.zeros((3,3))
    model.eval()
    with torch.no_grad():
        for idx, (imgs, labels, radiomics_data, _) in tqdm(enumerate(test_loader), desc='Train', total=len(test_loader)):
            imgs, labels = imgs.to(device), labels.to(device)
            predict_label = model(imgs)
            
            roc_(torch.nn.Softmax(
                dim=1)(predict_label).clone().detach(
            ).cpu().numpy())
            
            predict_label = (torch.nn.Softmax(
                dim=1)(predict_label)).argmax(dim=1)
            in_predict_label, in_labels = predict_label.clone().detach(
            ).cpu().numpy(), labels.clone().detach().cpu().numpy()
            total_accuracy += accuary_(in_predict_label, in_labels)
            tmp1,tmp2=f1_score_(in_predict_label, in_labels)
            total_f1_score_macro+=tmp1
            total_f1_score_mirco+=tmp2
            total_cm+=confusion_matrix_(in_predict_label, in_labels)
            torch.cuda.empty_cache()
    total_accuracy = total_accuracy / len(test_loader)
    total_f1_score_macro = total_f1_score_macro / len(test_loader)

    total_f1_score_mirco = total_f1_score_mirco / len(test_loader)

    print(total_accuracy,total_f1_score_macro,total_f1_score_mirco)
    print(total_cm)
    exit()
    return total_accuracy


def main(args):

    match, device, num_classes, batch_size,  model_name =\
        args.match, args.device, args.num_classes, args.batch_size, args.model_name
    radiomics_require, pth, save_seg, test_data_dir = \
        args.radiomics_require, args.pth, args.save_seg, args.test_data_dir
    save_dir = save_seg+'/'+model_name
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
    print('===>Loading dataset')

    require = True if (match and radiomics_require) else False

    test_data_loader = DataLoader(
        dataset=clfDataSet(test_data_dir, match=require), batch_size=batch_size,
        num_workers=8, shuffle=True, drop_last=False)
    print('test_data_dir:',test_data_dir)
    print('test_data_loader data:', len(test_data_loader))
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    test_begin_time = time.time()
    val_loss = test(
        model=model, test_loader=test_data_loader, device=device, radiomics_require=radiomics_require)

    print('This test cost %ds.' % (time.time()-test_begin_time))


if __name__ == '__main__':
    args = getConfig('test')
    main(args)
