import torch
import torch.nn as nn
import numpy as np

from meld.util.pytorch_complex import *
from meld.model.pytorch_transforms import Wavelet2

from meld.model import pbn_layer

mul_c  = ComplexMul().apply
div_c  = ComplexDiv().apply
abs_c  = ComplexAbs().apply
abs2_c = ComplexAbs2().apply 
exp_c  = ComplexExp().apply

class Conv2dSame(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias=True, padding_layer=torch.nn.ReflectionPad2d):
        super().__init__()
        ka = kernel_size // 2
        kb = ka - 1 if kernel_size % 2 == 0 else ka
        self.net = torch.nn.Sequential(
            padding_layer((ka,kb,ka,kb)),
            torch.nn.Conv2d(in_channels, out_channels, kernel_size, bias=bias)
        )
    def forward(self, x):
        return self.net(x)

class ResNet3(pbn_layer):
    def __init__(self, num_filters=32, filter_size=3, T=4):
        super(ResNet3, self).__init__()
        self.model = nn.Sequential(
            Conv2dSame(1,num_filters,filter_size),
            nn.ReLU(),
            Conv2dSame(num_filters,num_filters,filter_size),
            nn.ReLU(),
            Conv2dSame(num_filters,1,filter_size)
        )
        self.T = T
        
    def forward(self,x,device='cpu'):
        return x + self.step(x,device=device)
    
    def step(self,x,device='cpu'):
        x = x.unsqueeze(0).unsqueeze(0)
        return self.model(x).squeeze(0).squeeze(0)
    
    def reverse(self, x, device='cpu'):
        z = x
        for _ in range(self.T):
            z = x - self.step(z)
        return z
    
class ResNet4(pbn_layer):
    def __init__(self, num_filters=32, filter_size=3, T=4):
        super(ResNet4, self).__init__()
        self.model = nn.Sequential(
            Conv2dSame(2,num_filters,filter_size),
            nn.ReLU(),
            Conv2dSame(num_filters,num_filters,filter_size),
            nn.ReLU(),
            Conv2dSame(num_filters,2,filter_size)
        )
        self.T = T
        
    def forward(self,x,device='cpu'):
        return x + self.step(x,device=device)
    
    def step(self,x,device='cpu'):
        # reshape (batch,x,y,channel=2) -> (batch,channel=2,x,y)
        x = x.permute(0, 3, 1, 2)
        y = self.model(x)
        # reshape (batch,channel=2,x,y) -> (batch,x,y,channel=2)
        return y.permute(0, 2, 3, 1)
    
    def reverse(self, x, device='cpu'):
        z = x
        for _ in range(self.T):
            z = x - self.step(z)
        return z