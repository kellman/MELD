import torch
import torch.nn as nn
import numpy as np
from meld.util.pytorch_complex import *
from meld.util.utility import * 

from meld.model import pbn_layer

mul_c  = ComplexMul().apply
div_c  = ComplexDiv().apply
abs_c  = ComplexAbs().apply
abs2_c = ComplexAbs2().apply 
exp_c  = ComplexExp().apply

class Wavelet2(pbn_layer):
    def __init__(self, Np, device='cpu'):
        super(Wavelet2, self).__init__()
        self.Np = Np
        self.c = torch.from_numpy(np.asarray([1/np.sqrt(4)])).type(dtype).to(device)
        
    def forward(self,x,shiftx=False,shifty=False,device='cpu'):
        if shiftx: x = roll2(x,1)
        if shifty: x = roll2(x.permute(1,0),1)
            
        a0 = self.c*(x[0::2,0::2] + x[0::2,1::2] + x[1::2,0::2] + x[1::2,1::2])
        a1 = self.c*(x[0::2,0::2] + x[0::2,1::2] - x[1::2,0::2] - x[1::2,1::2])
        a2 = self.c*(x[0::2,0::2] - x[0::2,1::2] + x[1::2,0::2] - x[1::2,1::2])
        a3 = self.c*(x[0::2,0::2] - x[0::2,1::2] - x[1::2,0::2] + x[1::2,1::2])
        return [a0,a1,a2,a3]
    
    def reverse(self,a,shiftx=False,shifty=False,device='cpu'): # inverse
        x = torch.zeros(self.Np,dtype=dtype,device=device)
        x[0::2,0::2] = self.c*(a[0] + a[1] + a[2] + a[3])
        x[0::2,1::2] = self.c*(a[0] + a[1] - a[2] - a[3])
        x[1::2,0::2] = self.c*(a[0] - a[1] + a[2] - a[3])
        x[1::2,1::2] = self.c*(a[0] - a[1] - a[2] + a[3])
        
        if shifty: x = roll2(x,-1).permute(1,0)
        if shiftx: x = roll2(x,-1)
        return x