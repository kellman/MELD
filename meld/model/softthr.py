import torch
import torch.nn as nn
import numpy as np
from meld.util.pytorch_complex import *
from meld.model import pbn_layer
from meld.util import getAbs, getPhase

np_dtype = np.float32
dtype = torch.float32

mul_c  = ComplexMul().apply
div_c  = ComplexDiv().apply
abs_c  = ComplexAbs().apply
abs2_c = ComplexAbs2().apply 
exp_c  = ComplexExp().apply

    
class isoftthr(pbn_layer):
    def __init__(self, thr, alpha, testFlag = False, device='cpu'):
        super(isoftthr, self).__init__()
        self.testFlag = testFlag
        self.alpha = alpha
#         self.thr = nn.Parameter(torch.from_numpy(np.asarray([thr],dtype=np_dtype))).to(device)
        self.thr = torch.from_numpy(np.asarray([thr],dtype=np_dtype)).to(device)
#         self.thr.requires_grad_(not self.testFlag)
#         self.thr.requires_grad_(False)
        
    def forward(self,x,device='cpu'):
        z = (torch.abs(x) - self.thr*(1-self.alpha)) * (torch.abs(x) > self.thr).type(dtype) * torch.sign(x)
        z = z + self.alpha * x * (torch.abs(x) <= self.thr).type(dtype)
        return z
    
    def reverse(self,x,device='cpu'):
        z = (torch.abs(x) + self.thr * (1 - self.alpha)) * (torch.abs(x) > self.thr*self.alpha).type(dtype) * torch.sign(x)
        z += x * (1 / self.alpha) * (torch.abs(x) <= self.thr * self.alpha).type(dtype)
        return z
    
# class inv_soft_thr2(pbn_layer):
#     def __init__(self, thr, alpha, testFlag = False):
#         super(inv_soft_thr2, self).__init__()
#         self.testFlag = testFlag
#         self.alpha = alpha
#         self.thr = nn.Parameter(torch.from_numpy(np.asarray([thr],dtype=np_dtype)))
#         self.thr.requires_grad_(not self.testFlag)
        
#     def forward(self,x,device='cpu'):
#         return self.alpha * x * (torch.abs(x) <= self.thr).type(dtype) + (x - self.thr*(1-self.alpha)) * (x > self.thr).type(dtype) + (x + self.thr*(1-self.alpha)) * (x < -1 * self.thr).type(dtype)
    
#     def reverse(self,x,device='cpu'):
#         thr2 = self.thr*self.alpha
#         invslope = (1 / self.alpha)
#         return x * invslope * (torch.abs(x) <= thr2).type(dtype) + (x + self.thr*(1-self.alpha)) * (x > thr2).type(dtype) + (x - self.thr*(1-self.alpha)) * (x < -1 * thr2).type(dtype)
    