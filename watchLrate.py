import torch
import argparse
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from models.resnet import resnet18


def main(args):
    if args.model_path==None:
        raise ValueError('undifine path')
    else:
        model = resnet18(pretrained=False, num_classes=3).to('cuda')
        optimizer = Adam(model.parameters(), lr=1e-4, weight_decay=1e-3)
        scheduler = CosineAnnealingWarmRestarts(
        optimizer, T_0=10, eta_min=1e-6, last_epoch=-1, T_mult=2)
        ckpt=torch.load(args.model_path)
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
    lrate_list=[]
    for i in range(100):
        lrate_list.append(optimizer.state_dict()['param_groups'][0]['lr'])
        scheduler.step()
    print(lrate_list[0],max(lrate_list),min(lrate_list))

if __name__ == '__main__':
    parse_=argparse.ArgumentParser()
    parse_.add_argument('--model_path','-m',type=str,default=None)

    args=parse_.parse_args()
    main(args)
    