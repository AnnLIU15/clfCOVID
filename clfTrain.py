import os
import time

import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
from torchsummary import summary
from tqdm import tqdm

from clfConfig import getConfig
from datasets.clfDataSet import clfDataSet
from models.resnet import resnet18,resnet34,resnet50


def train(model, train_loader, optimizer, device, radiomics_require=False):
    epoch_loss = 0
    model.train()
    for idx, (imgs, labels, _) in tqdm(enumerate(train_loader), desc='Train', total=len(train_loader)):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        predict_label = model(imgs)
        # print(one_hot_mask_.shape,masks.shape)
        loss = CrossEntropyLoss()(predict_label, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.clone().detach().cpu().numpy()
        torch.cuda.empty_cache()
    epoch_loss = epoch_loss / len(train_loader)
    return epoch_loss


def val(model, val_loader, device, radiomics_require=False):
    epoch_loss = 0
    model.eval()
    with torch.no_grad():
        for idx, (imgs, labels, _) in tqdm(enumerate(val_loader), desc='Train', total=len(val_loader)):
            imgs, labels = imgs.to(device), labels.to(device)
            predict_label = model(imgs)
            loss = CrossEntropyLoss()(predict_label, labels)
            epoch_loss += loss.clone().detach().cpu().numpy()
            torch.cuda.empty_cache()
    epoch_loss = epoch_loss / len(val_loader)
    return epoch_loss


def main(args):
    '''
    loading arguements
    '''
    match, device, lrate, num_classes, num_epochs, log_name, batch_size,  model_name =\
        args.match, args.device, args.lrate, args.num_classes, args.num_epochs, args.log_name, args.batch_size, args.model_name
    radiomics_require, pth, save_dir, save_every, start_epoch, train_data_dir, val_data_dir = \
        args.radiomics_require, args.pth, args.save_dir, args.save_every, args.start_epoch, args.train_data_dir, args.val_data_dir
    
    '''
    dir of saving result
    '''
    save_dir = save_dir+'/'+model_name
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
    
    if model_name=='resnet34':
        model = resnet34(pretrained=False, num_classes=num_classes).to(device)
    elif model_name=='resnet50':
        model = resnet50(pretrained=False, num_classes=num_classes).to(device) 
    elif model_name=='vgg':
        pass   
    else:
        model = resnet18(pretrained=False, num_classes=num_classes).to(device)
    # summary(model=model, input_size=(
    #     1, 512, 512), batch_size=4, device='cuda')

    print('===>Setting optimizer and scheduler')
    optimizer = Adam(model.parameters(), lr=lrate, weight_decay=1e-3)
    scheduler = CosineAnnealingWarmRestarts(
        optimizer, T_0=10, eta_min=1e-6, last_epoch=-1, T_mult=2)

    if not pth == None:
        print('===>Loading Pretrained Model')
        checkpoint = torch.load(pth)
        model.load_state_dict(checkpoint['model_weights'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch']+1
    print('===>Making log')
    if not os.path.exists('./log/clf/'):
        os.makedirs('./log/clf/')
    if log_name == None:
        writer = SummaryWriter(
            './log/clf/'+model_name+time.strftime('%m%d-%H%M', time.localtime(time.time())))
    else:
        writer = SummaryWriter('./log/clf/'+log_name)
    print('===>Loading dataset')
    '''
    whether use radiomics_data
    '''
    require = True if (match and radiomics_require) else False

    train_data_loader = DataLoader(
        dataset=clfDataSet(train_data_dir, match=require), batch_size=batch_size,
        num_workers=8, shuffle=True, drop_last=False)
    val_data_loader = DataLoader(
        dataset=clfDataSet(val_data_dir, match=require), batch_size=batch_size,
        num_workers=8, shuffle=True, drop_last=False)
    print('train_data_loader data:', len(train_data_loader))
    print('val_data_loader data:', len(val_data_loader))
    best_train_performance = [0, np.Inf]
    best_val_performance = [0, np.Inf]
    train_start_time = time.time()
    for epoch in range(start_epoch, start_epoch+num_epochs):
        epoch_begin_time = time.time()
        print("\n"+"="*20+"Epoch[{}:{}]".format(epoch, start_epoch+num_epochs-1)+"="*20 +
              '\nlr={}\tweight_decay={}'.format(optimizer.state_dict()['param_groups'][0]['lr'],
                                                optimizer.state_dict()['param_groups'][0]['weight_decay']))
        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        '''
        train and val
        '''
        train_loss = train(
            model=model, train_loader=train_data_loader, optimizer=optimizer, device=device, radiomics_require=radiomics_require)
        val_loss = val(
            model=model, val_loader=val_data_loader, device=device, radiomics_require=radiomics_require)
        scheduler.step()
        print('Epoch %d Train Loss:%.4f\t\t\tValidation Loss:%.4f' %
              (epoch, train_loss, val_loss))
        '''
        save the model if it satisfied the conditions
        '''
        if best_train_performance[1] > train_loss and train_loss > 0:
            state = {'epoch': epoch, 'model_weights': model.state_dict(
            ), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict()}
            torch.save(state, os.path.join(
                save_dir, 'best_train_model.pth'.format(epoch)))
            best_train_performance = [epoch, train_loss]

        if best_val_performance[1] > val_loss and val_loss > 0:
            state = {'epoch': epoch, 'model_weights': model.state_dict(
            ), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict()}
            torch.save(state, os.path.join(
                save_dir, 'best_val_model.pth'.format(epoch)))
            best_val_performance = [epoch, val_loss]

        if epoch % save_every == 0:
            state = {'epoch': epoch, 'model_weights': model.state_dict(
            ), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict()}
            torch.save(state, os.path.join(
                save_dir, 'epoch_{}_model.pth'.format(epoch)))
        print('Best train loss epoch:%d\t\t\tloss:%.4f' %
              (best_train_performance[0], best_train_performance[1]))
        print('Best val loss epoch:%d\t\t\tloss:%.4f' %
              (best_val_performance[0], best_val_performance[1]))
        '''
        tensorboard visualize
        ---------------------
        train_loss
        val_loss
        '''
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('val/loss', val_loss, epoch)
        epoch_time = time.time()-epoch_begin_time
        print('This epoch cost %.4fs, predicting it will take another %.4fs'
              % (epoch_time, epoch_time*(start_epoch+num_epochs-epoch-1)))
    train_end_time = time.time()
    print('This train total cost %.4fs' % (train_end_time-train_start_time))
    writer.close()


if __name__ == '__main__':
    args = getConfig('train')
    main(args)
