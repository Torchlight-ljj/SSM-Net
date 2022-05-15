""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F
# from resnet import resnet50
from model.resnet import resnet50
import torch.nn.functional as F
from math import sqrt
import numpy  as np
import time
import parameters
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.conv(x)
        return x1

class SSM_Net(nn.Module):
    def __init__(self, n_channels, n_classes, pretrained_model_path = None, bilinear=True, num_classes =4):
        super(UNet_Res, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.context_path = resnet50(pretrained=False,num_classes=num_classes)

        if pretrained_model_path:
        #     # freeze all layers but the last fc
            for name, param in self.context_path.named_parameters():
                if name not in ['outConv.weight', 'outConv.bias', 'outBN.weight',
                 'outBN.bias', 'outBN.running_mean', 'outBN.running_var', 
                 'outBN.num_batches_tracked', 'fc.weight', 'fc.bias', 
                 'attention.query_projection.weight', 'attention.query_projection.bias', 
                 'attention.key_projection.weight', 'attention.key_projection.bias', 
                 'attention.value_projection.weight', 'attention.value_projection.bias', 
                 'attention.out_projection.weight', 'attention.out_projection.bias',]:
                    param.requires_grad = False
   
            self.context_path.load_state_dict(torch.load(pretrained_model_path),False)

        self.refine = nn.Conv2d(2048, 1024, kernel_size=3, padding=1)
        self.up1 = Up(2048, 512, bilinear)
        self.up2 = Up(1024, 256, bilinear)
        self.up3 = Up(512, 128, bilinear)
        self.outc = OutConv(128, n_classes)

    def forward(self, x):
        # x = self.SR(x)
        context_blocks,y_cla,features = self.context_path(x)
        context_blocks.reverse()
        y = self.refine(context_blocks[0])
        y = self.up1(y, context_blocks[1])
        y = self.up2(y, context_blocks[2])
        y = self.up3(y, context_blocks[3])
        y = self.outc(y)
        y = F.interpolate(y, scale_factor=4,
                          mode='bilinear',
                          align_corners=True)
        return y,y_cla,features

    def load_weights(self, weights_path):
        weights = torch.load(weights_path)
        del weights["fc.weight"]
        del weights["fc.bias"]

        names = []
        for key, value in self.context_path.state_dict().items():
            if "num_batches_tracked" in key:
                continue
            names.append(key)

        for name, dict in zip(names, weights.items()):
            self.weights_new[name] = dict[1]

        self.context_path.load_state_dict(self.weights_new,False)
