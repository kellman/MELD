import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import sys

sys.path.append('/home/kellman/Workspace/PYTHON/Pytorch_Physics_Library/Utilities/')
from utility import *
from optics import *
from pytorch_complex import *
from model import *
sys.path.append('/home/kellman/Workspace/PYTHON/Pytorch_Physics_Library/Denoisers/')
from pytorch_proximal import *
from pytorch_transforms import *

mul_c  = ComplexMul().apply
div_c  = ComplexDiv().apply
abs_c  = ComplexAbs().apply
abs2_c = ComplexAbs2().apply 
exp_c  = ComplexExp().apply

class MRI(nn.Module):
    def __init__(self, Np, img, maps, sampling, alpha = 1e-2, noise_level=1e-2, T=4, testFlag=False, device='cpu'):
        super(MRI,self).__init__()
    
        self.Np = Np
        self.T = T
        
        self.truth = img
        self.maps = maps
        self.sampling = sampling
      
        # measurements
        self.y = self.model(img)
        self.noise = torch.randn_like(self.y).to(device)
        self.y += self.noise * noise_level
        self.y = self.y.to(device)
        
        # parameters
        self.alpha = nn.Parameter(alpha)
        self.testFlag = testFlag
        if testFlag:
            self.alpha.requires_grad_(False)
        
    def forward(self, x, device='cpu'):
        return x + self.step(x)
    
    def reverse(self, x, device='cpu'):
        z = x
        for _ in range(self.T):
            z = x - self.step(z)
        return z

    def model(self,x,device='cpu'):
        # Apply maps
        PFSx = torch.zeros_like(self.maps)
        for ii in range(self.maps.shape[0]):
            PFSx[ii,...] = mul_c(self.maps[ii,...],x)
            PFSx[ii,...] = torch.fft(PFSx[ii,...],2)
            PFSx[ii,...] = mul_c(self.sampling,PFSx[ii,...])
        return PFSx
    
    def step(self,x):
        return -1 * self.alpha * self.grad(x)
    
    def grad(self,x,device='cpu'):
        AHAx = torch.zeros_like(self.maps)
        
        for ii in range(self.maps.shape[0]):
            
            # forward model
            AHAx[ii,...] = mul_c(self.maps[ii,...],x)
            AHAx[ii,...] = torch.fft(AHAx[ii,...],2)
            AHAx[ii,...] = mul_c(self.sampling,AHAx[ii,...])
            
            # compute measurement residual
            res = AHAx[ii,...] - self.y[ii,...]
            
            # adjoint model
            AHAx[ii,...] = mul_c(conj(self.sampling), res)
            AHAx[ii,...] = torch.ifft(AHAx[ii,...], 2)
            AHAx[ii,...] = mul_c(conj(self.maps[ii,...]), AHAx[ii,...])
            
        # coil combination
        output = torch.sum(AHAx,dim=0)
        
        return output
    
    def loss(self, x):
        return (1/2) * torch.sum((self.y - self.model(x))**2)
        
    def mse(self,x):
        with torch.no_grad():
            return torch.sum((x - self.truth)**2)