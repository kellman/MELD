import torch
import torch.nn as nn
import numpy as np

from meld.util.pytorch_complex import *
from meld.model.pytorch_transforms import Wavelet2

from meld.model import pbn_layer

from collections import OrderedDict

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
    
    
class ConvSame(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dims, bias=True):
        super().__init__()
        ka = kernel_size // 2
        kb = ka - 1 if kernel_size % 2 == 0 else ka
        if dims == 3:
            padding_layer=torch.nn.ReplicationPad3d
            conv_layer=torch.nn.Conv3d
            pad_params=(ka,kb,ka,kb,ka,kb)
        elif dims == 2:
            padding_layer=torch.nn.ReflectionPad2d
            conv_layer=torch.nn.Conv2d
            pad_params=(ka,kb,ka,kb)
            
        conv_params={'in_channels':in_channels, 'out_channels':out_channels, 'kernel_size':kernel_size, 'bias':bias}
        self.net = torch.nn.Sequential(
            padding_layer(pad_params),
            conv_layer(**conv_params)
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
    def __init__(self, dims, num_channels=32, kernel_size=3, T=4, num_layers=3):
        super(ResNet4, self).__init__()
        
        conv = lambda in_channels, out_channels, filter_size: ConvSame(in_channels, out_channels, filter_size, dims)
        self.dims = dims
        
        layer_dict = OrderedDict()
        layer_dict['conv1'] = conv(2,num_channels,kernel_size)
        layer_dict['relu1'] = nn.ReLU()
        for i in range(num_layers-2):
            layer_dict[f'conv{i+2}'] = conv(num_channels, num_channels, kernel_size)
            layer_dict[f'relu{i+2}'] = nn.ReLU()
        layer_dict[f'conv{num_layers}'] = conv(num_channels,2,kernel_size)
        
        self.model = nn.Sequential(layer_dict)
        self.T = T
        
    def forward(self,x,device='cpu'):
        return x + self.step(x,device=device)
    
    def step(self,x,device='cpu'):
        # reshape (batch,x,y,channel=2) -> (batch,channel=2,x,y)
        if self.dims == 2:
            p1 = (0, 3, 1, 2)
            p2 = (0, 2, 3, 1)
        elif self.dims == 3:
            p1 = (0, 4, 1, 2, 3)
            p2 = (0, 2, 3, 4, 1)
        x = x.permute(*p1)
        y = self.model(x)
        # reshape (batch,channel=2,x,y) -> (batch,x,y,channel=2)
        return y.permute(*p2)
    
    def reverse(self, x, device='cpu'):
        z = x
        for _ in range(self.T):
            z = x - self.step(z)
        return z