# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from PIL import ImageFilter
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""
    def __init__(self, base_transform):
        self.base_transform = base_transform
        self.SR = SRlayer_(1)

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        seed = random.random()
        if seed > 0.8:
            q = self.SR(q)
        seed = random.random()
        if seed > 0.8:
            k = self.SR(k)
        return [q, k]

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

class SRlayer_(nn.Module):
    def __init__(self,channel):
        super(SRlayer_,self).__init__()
        self.channel = channel
        self.batch = 1
        self.kernalsize = 3
        self.amp_conv = nn.Conv2d(self.channel, self.channel, kernel_size=self.kernalsize, stride=1,padding=1, bias=False)
        self.fucker = np.zeros([self.kernalsize,self.kernalsize])
        for i in range(self.kernalsize):
            for j in range(self.kernalsize):
                self.fucker[i][j] = 1/np.square(self.kernalsize)
        self.aveKernal = torch.Tensor(self.fucker).unsqueeze(0).unsqueeze(0).repeat(self.batch,self.channel,1,1)
        self.amp_conv.weight = nn.Parameter(self.aveKernal, requires_grad=False)
        
        self.amp_relu = nn.ReLU()
        self.gaussi = nn.Conv2d(self.channel, self.channel, kernel_size=3, stride=1,padding=1, bias=False)
        self.gauKernal = torch.Tensor([[1/16, 1/8, 1/16], [1/8, 1/4, 1/8], [1/16, 1/8, 1/16]]).unsqueeze(0).unsqueeze(0).repeat(self.batch,self.channel,1,1)
        self.gaussi.weight = nn.Parameter(self.gauKernal,requires_grad=False)

    def forward(self,x):
        out = []
        for batch in range(x.shape[0]):
            x1 = x[batch,:,:].unsqueeze(0).unsqueeze(0)
            rfft = torch.fft.fftn(x1)
            amp = torch.abs(rfft) + torch.exp(torch.tensor(-10))
            log_amp = torch.log(amp)
            phase = torch.angle(rfft)
            amp_filter = self.amp_conv(log_amp)
            amp_sr = log_amp - amp_filter
            SR = torch.fft.ifftn(torch.exp(amp_sr+1j*phase))
            SR = torch.abs(SR)
            SR = self.gaussi(SR)
            out.append(SR.squeeze(0))
        return torch.cat(out,dim=0)
