import torch
import torch.nn as nn
import numpy as np
from meld.util.pytorch_complex import *
from meld.model.pytorch_transforms import Wavelet2

np_dtype = np.float32
dtype = torch.float32

mul_c  = ComplexMul().apply
div_c  = ComplexDiv().apply
abs_c  = ComplexAbs().apply
abs2_c = ComplexAbs2().apply 
exp_c  = ComplexExp().apply

class wavelet_soft_thr(nn.Module):
    def __init__(self, Np, thr, alpha, testFlag = False,device='cpu'):
        super(wavelet_soft_thr, self).__init__()
        self.testFlag = testFlag
        self.alpha = alpha
        self.thr = thr
        self.prox = inv_soft_thr2(thr, alpha, testFlag)
        self.trans = Wavelet2(Np,device=device)
        self.device = device
        
    def forward(self, x, shiftx=False, shifty=False, device='cpu'):
        a = self.trans.forward(x,shiftx,shifty,device=self.device)
        a_s = [a[0],self.prox(a[1]),self.prox(a[2]),self.prox(a[3])]
        x = self.trans.adjoint(a_s,shiftx,shifty,device=self.device)        
        return x
    
    def reverse(self, z, shiftx=False, shifty=False, device='cpu'):
        a = self.trans.forward(z,shiftx,shifty,device=self.device)
        a_s = [a[0],self.prox.reverse(a[1]),self.prox.reverse(a[2]),self.prox.reverse(a[3])]
        z = self.trans.adjoint(a_s,shiftx,shifty,device=self.device)        
        return z
    
    def loss(self,z,lamb):
        a = self.trans.forward(x,shiftx,shifty,device=self.device)
        return lamb * torch.sum(torch.abs(a[1]) + torch.abs(a[2]) + torch.abs(a[3]))

    
class abs_wavelet_soft_thr(nn.Module):
    def __init__(self, Np, thr, alpha, testFlag = False,device='cpu'):
        super(abs_wavelet_soft_thr, self).__init__()
        self.testFlag = testFlag
        self.alpha = alpha
        self.thr = thr
        self.prox = inv_soft_thr(thr,alpha,testFlag)
        self.trans = Wavelet2(Np,device=device)
        self.device = device
        
    def forward(self, x, shiftx=False, shifty=False, device='cpu'):
        amp = getAbs(x)
        phase = getPhase(x)

        a = self.trans.forward(amp,shiftx,shifty,device=self.device)
        a_s = [a[0],self.prox(a[1]),self.prox(a[2]),self.prox(a[3])]
        x = self.trans.adjoint(a_s,shiftx,shifty,device=self.device)        

        x = self.getExp(x,phase)
        return x

    def reverse(self, z, shiftx=False, shifty=False, device='cpu'):
        amp = getAbs(z)
        phase = getPhase(z)

        a = self.trans.forward(amp,shiftx,shifty,device=self.device)
        a_s = [a[0],self.prox.inverse(a[1]),self.prox.inverse(a[2]),self.prox.inverse(a[3])]
        z = self.trans.adjoint(a_s,shiftx,shifty,device=self.device)  

        z = self.getExp(z,phase)

        return z

    def loss(self,z,lamb):
        a = self.trans.forward(x,shiftx,shifty,device=self.device)
        return lamb * torch.sum(torch.abs(a[1]) + torch.abs(a[2]) + torch.abs(a[3]))
    
    def getExp(self,a_amp,a_angle):
        return torch.stack((a_amp * torch.cos(a_angle), a_amp * torch.sin(a_angle)),2)
    
    
class inv_soft_thr(nn.Module):
    def __init__(self, thr, alpha, testFlag = False):
        super(inv_soft_thr, self).__init__()
        self.testFlag = testFlag
        self.alpha = alpha
        self.thr = nn.Parameter(torch.from_numpy(np.asarray([thr],dtype=np_dtype)))
        self.thr.requires_grad_(not self.testFlag)
        
    def forward(self,x,device='cpu'):
        z = (torch.abs(x) - self.thr*(1-self.alpha)) * (torch.abs(x) > self.thr).type(dtype) * torch.sign(x)
        z = z + self.alpha * x * (torch.abs(x) <= self.thr).type(dtype)
        return z
    
    def reverse(self,x,device='cpu'):
        z = (torch.abs(x) + self.thr * (1 - self.alpha)) * (torch.abs(x) > self.thr*self.alpha).type(dtype) * torch.sign(x)
        z += x * (1 / self.alpha) * (torch.abs(x) <= self.thr * self.alpha).type(dtype)
        return z
    
class inv_soft_thr2(nn.Module):
    def __init__(self, thr, alpha, testFlag = False):
        super(inv_soft_thr2, self).__init__()
        self.testFlag = testFlag
        self.alpha = alpha
        self.thr = nn.Parameter(torch.from_numpy(np.asarray([thr],dtype=np_dtype)))
        self.thr.requires_grad_(not self.testFlag)
        
    def forward(self,x,device='cpu'):
        return self.alpha * x * (torch.abs(x) <= self.thr).type(dtype) + (x - self.thr*(1-self.alpha)) * (x > self.thr).type(dtype) + (x + self.thr*(1-self.alpha)) * (x < -1 * self.thr).type(dtype)
    
    def reverse(self,x,device='cpu'):
        thr2 = self.thr*self.alpha
        invslope = (1 / self.alpha)
        return x * invslope * (torch.abs(x) <= thr2).type(dtype) + (x + self.thr*(1-self.alpha)) * (x > thr2).type(dtype) + (x - self.thr*(1-self.alpha)) * (x < -1 * thr2).type(dtype)
    
class shrinkage(nn.Module):
    def __init__(self, thr, testFlag = False):
        super(shrinkage, self).__init__()
        self.testFlag = testFlag
        self.thr = nn.Parameter(torch.from_numpy(np.asarray([thr],dtype=np_dtype)))
        self.thr.requires_grad_(not self.testFlag)
        
    def forward(self,z,device='cpu'):
        return z / (1 + self.thr)
    
    def reverse(self,z,device='cpu'):
        return z * (1 + self.thr)
    
class soft_thr(nn.Module):
    def __init__(self, thr, testFlag = False):
        super(soft_thr, self).__init__()
        self.testFlag = testFlag
        self.thr = nn.Parameter(torch.from_numpy(np.asarray([thr],dtype=np_dtype)))
        self.thr.requires_grad_(not self.testFlag)
        
    def forward(self,x,device='cpu'):
        return (torch.abs(x) - self.thr) * (torch.abs(x) > self.thr).type(dtype) * torch.sign(x)
    
    def reverse(self,x,device='cpu'):
        print('Inverse does not exist, use invSoftThr')
        return