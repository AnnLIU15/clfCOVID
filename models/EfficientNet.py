import torch

import torch.nn as nn
import torch.nn.functional
import numpy as np

def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            std_var=(2.0 / (fan_out // m.groups))**0.5
            m.weight.data.normal_(mean=0, std=std_var)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1.0)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.uniform_(-1.0 / np.sqrt(m.weight.size()[0]), 1.0 / np.sqrt(m.weight.size()[0]))
            m.bias.data.zero_()


class SiLU(nn.Module):
    """
    [https://arxiv.org/pdf/1710.05941.pdf]
    """

    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = inplace

    def forward(self, x):
        return x.mul_(x.sigmoid()) if self.inplace else x.mul(x.sigmoid())


class Conv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=False):
        super().__init__(in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias)

    def forward(self, x):
        s = self.stride
        d = self.dilation
        k = self.weight.shape[-2:]
        h, w = x.size()[-2:]
        
        pad_h =int(max((np.ceil(h / s[0]) - 1) * s[0] + (k[0] - 1) * d[0] + 1 - h, 0))
        pad_w =int(max((np.ceil(w / s[1]) - 1) * s[1] + (k[1] - 1) * d[1] + 1 - w, 0))
        
        if pad_h > 0 or pad_w > 0:
            x = nn.functional.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2], value=0)

        return nn.functional.conv2d(x, self.weight, self.bias, self.stride, (0, 0), self.dilation, self.groups)


class Conv(nn.Module):
    def __init__(self, tf, in_channels, out_channels, activation, k=1, s=1, g=1):
        super().__init__()
        if tf:
            self.conv = Conv2d(in_channels, out_channels, k, s, 1, g, bias=False)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, k, s, k // 2, 1, g, bias=False)
        self.norm = nn.BatchNorm2d(out_channels, 0.001, 0.01)
        self.silu = activation

    def forward(self, x):
        return self.silu(self.norm(self.conv(x)))


class SE(nn.Module):
    """
    [https://arxiv.org/pdf/1709.01507.pdf]
    """

    def __init__(self, ch, r):
        super().__init__()
        self.se = nn.Sequential(nn.Conv2d(ch, ch // (4 * r), 1),
                                      nn.SiLU(),
                                      nn.Conv2d(ch // (4 * r), ch, 1),
                                      nn.Sigmoid())

    def forward(self, x):
        return x * self.se(x.mean((2, 3), keepdim=True))


class Residual(nn.Module):
    """
    [https://arxiv.org/pdf/1801.04381.pdf]
    """

    def __init__(self, tf, in_channels, out_channels, s, r, fused=True):
        super().__init__()
        identity = nn.Identity()
        if fused:
            if tf and r == 1:
                features = [Conv(tf, in_channels, r * in_channels, nn.SiLU(), 3, s)]
            else:
                features = [Conv(tf, in_channels, r * in_channels, nn.SiLU(), 3, s),
                            Conv(tf, r * in_channels, out_channels, identity)]
        else:
            if r == 1:
                features = [Conv(tf, r * in_channels, r * in_channels, nn.SiLU(), 3, s, r * in_channels),
                            SE(r * in_channels, r),
                            Conv(tf, r * in_channels, out_channels, identity)]
            else:
                features = [Conv(tf, in_channels, r * in_channels, nn.SiLU()),
                            Conv(tf, r * in_channels, r * in_channels, nn.SiLU(), 3, s, r * in_channels),
                            SE(r * in_channels, r),
                            Conv(tf, r * in_channels, out_channels, identity)]
        self.add = s == 1 and in_channels == out_channels
        self.res = nn.Sequential(*features)

    def forward(self, x):
        return x + self.res(x) if self.add else self.res(x)


class EfficientNet(nn.Module):
    def __init__(self, tf,in_channels=1, num_class=1000,softmax_require=False) -> None:
        super().__init__()
        self.softmax_require=softmax_require
        gate_fn = [True, False]
        filters = [24, 48, 64, 128, 160, 272, 1792]
        feature = [Conv(tf, in_channels, filters[0], nn.SiLU(), 3, 2)]
        if tf:
            filters[5] = 256
            filters[6] = 1280
        for i in range(2):
            if i == 0:
                feature.append(Residual(tf, filters[0], filters[0], 1, 1, gate_fn[0]))
            else:
                feature.append(Residual(tf, filters[0], filters[0], 1, 1, gate_fn[0]))

        for i in range(4):
            if i == 0:
                feature.append(Residual(tf, filters[0], filters[1], 2, 4, gate_fn[0]))
            else:
                feature.append(Residual(tf, filters[1], filters[1], 1, 4, gate_fn[0]))

        for i in range(4):
            if i == 0:
                feature.append(Residual(tf, filters[1], filters[2], 2, 4, gate_fn[0]))
            else:
                feature.append(Residual(tf, filters[2], filters[2], 1, 4, gate_fn[0]))

        for i in range(6):
            if i == 0:
                feature.append(Residual(tf, filters[2], filters[3], 2, 4, gate_fn[1]))
            else:
                feature.append(Residual(tf, filters[3], filters[3], 1, 4, gate_fn[1]))

        for i in range(9):
            if i == 0:
                feature.append(Residual(tf, filters[3], filters[4], 1, 6, gate_fn[1]))
            else:
                feature.append(Residual(tf, filters[4], filters[4], 1, 6, gate_fn[1]))

        for i in range(15):
            if i == 0:
                feature.append(Residual(tf, filters[4], filters[5], 2, 6, gate_fn[1]))
            else:
                feature.append(Residual(tf, filters[5], filters[5], 1, 6, gate_fn[1]))
        feature.append(Conv(tf, filters[5], filters[6], nn.SiLU()))

        self.feature = nn.Sequential(*feature)
        self.fc = nn.Sequential(nn.Dropout(0.3, True),
                                      nn.Linear(filters[6], num_class))

        initialize_weights(self)

    def forward(self, x):
        x = self.feature(x)
        x = self.fc(x.mean((2, 3)))
        if self.softmax_require:
            x=nn.Softmax(dim=1)(x)
        return x

    def export(self):
        for m in self.modules():
            if type(m) is Conv and hasattr(m, 'silu'):
                if isinstance(m.silu, nn.SiLU):
                    m.silu = SiLU()
            if type(m) is SE:
                if isinstance(m.se[1], nn.SiLU):
                    m.se[1] = SiLU()
        return self
if __name__ == '__main__':
    device='cuda'
    x=torch.rand(8,1,512,512).to(device)
    model=EfficientNet(tf=1,num_class=3).to(device)
    y=model(x).to(device)
    print(x.shape,y.shape)
    print(y)
    del y,model
    model=EfficientNet(tf=0,num_class=3).to(device)
    y=model(x).to(device)
    print(x.shape,y.shape)
    print(y)