import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import sys

sys.path.append('/home/kellman/Workspace/PYTHON/MELD/utilities/')
from utility import *
from pytorch_complex import *

mul_c  = ComplexMul().apply
div_c  = ComplexDiv().apply
abs_c  = ComplexAbs().apply
abs2_c = ComplexAbs2().apply 
exp_c  = ComplexExp().apply

class Deblur(nn.Module):
    def __init__(self, Np, img, kernel, alpha, lamb=1e-1, noise_level=1e-2, T=4, testFlag=False, device='cpu'):
        super(Deblur,self).__init__()
    
        self.Np = Np
        self.T = T
      
        # point spread function (put design parameters in here)
#         self.psf = torch.from_numpy(kernel).type(dtype)
        self.psf = kernel/torch.sum(kernel)    
        tmp = torch.stack((self.psf ,torch.zeros_like(self.psf)),2)
        self.fpsf = torch.fft(tmp,2)
        self.fpsf = self.fpsf.to(device)
        
        # measurements
        self.xtruth = img
        self.y = self.blur(img)
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

    def blur(self,img):
        img = torch.stack((img,torch.zeros_like(img)),2)
        fimg = torch.fft(img,2)
        output = torch.ifft(mul_c(self.fpsf,fimg),2)
        return output[...,0]
    
    def step(self,x):
        return -1 * self.alpha * self.grad(x)
    
    def grad(self,x):
        ys = torch.stack((self.y,torch.zeros_like(self.y)),2)
        Fy = torch.fft(ys,2)
        AHy = mul_c(conj(self.fpsf),Fy)
        
        xs = torch.stack((x,torch.zeros_like(x)),2)
        Fx = torch.fft(xs,2)
        AHA = mul_c(conj(self.fpsf),self.fpsf)
        AHAx = mul_c(AHA,Fx)
        
        return torch.ifft(AHAx - AHy,2)[...,0]
    
    def pinv(self,eps=1e-8):
        ys = torch.stack((self.y,torch.zeros_like(self.y)),2)
        Fys = torch.fft(ys,2)
        output = torch.ifft(div_c(Fys,self.fpsf + eps),2)
        return output[...,0]
    
    def verifyAdjoint(self,device='cpu',printFlag=False):
        x = torch.randn(Np[0],Np[1],2,device=device)
        y = torch.randn(Np[0],Np[1],2,device=device)
        
        # <Ax,y>
        test1 = torch.sum(mul_c(conj(mul_c(self.fpsf,x)),y))
        
        # <x,AHy>
        test2 = torch.sum(mul_c(conj(x),mul_c(conj(self.fpsf),y)))
        
        if printFlag:
            print(test1,test2,torch.abs(test1-test2))
        
        assert torch.abs(test1-test2) < 1e-3 , 'Adjoint Verification Failed!'
    
    def loss(self, x):
        return (1/2) * torch.sum((self.y - self.blur(x))**2)
        
    def mse(self,x):
        with torch.no_grad():
            return torch.sum((x - self.xtruth)**2)
        
#     def recon(self,x0,N,device='cpu'):
#         x = x0
#         ll = torch.zeros(N)
#         for ii in range(N):
            
#             if ii % 4 == 0:
#                 shiftx = False
#                 shifty = False
#             elif ii % 4 == 1:
#                 shiftx = True
#                 shifty = False
#             elif ii % 4 == 2:
#                 shiftx = False
#                 shifty = True
#             elif ii % 4 == 3:
#                 shiftx = True
#                 shifty = True 
                
#             x = x + self.forward(x)
#             a = self.trans.forward(x,shiftx,shifty)
#             a_s = [a[0],self.nonlin(a[1]),self.nonlin(a[2]),self.nonlin(a[3])]
# #             a_s = [self.nonlin(a[0]),self.nonlin(a[1]),self.nonlin(a[2]),self.nonlin(a[3])]
#             x = self.trans.adjoint(a_s,shiftx,shifty,device=device)
#             with torch.no_grad():
#                 ll[ii] = self.loss(x)
#         return x,ll

#     def recon_storage(self,x0,N,device='cpu'):
#         x = x0
#         ll = torch.zeros(N)
#         X = torch.zeros(N,x0.shape[0],x0.shape[1])
#         for ii in range(N):
            
#             if ii % 4 == 0:
#                 shiftx = False
#                 shifty = False
#             elif ii % 4 == 1:
#                 shiftx = True
#                 shifty = False
#             elif ii % 4 == 2:
#                 shiftx = False
#                 shifty = True
#             elif ii % 4 == 3:
#                 shiftx = True
#                 shifty = True 
                
#             x = x + self.forward(x)
#             a = self.trans.forward(x,shiftx,shifty)
# #             a_s = [self.nonlin(a[0]),self.nonlin(a[1]),self.nonlin(a[2]),self.nonlin(a[3])]
#             a_s = [a[0],self.nonlin(a[1]),self.nonlin(a[2]),self.nonlin(a[3])]
#             x = self.trans.adjoint(a_s,shiftx,shifty,device=device)
            
#             with torch.no_grad():
#                 ll[ii] = self.loss(x)
#                 X[ii,...] = x
#         return x, ll, X

#     def fullinvstep(self,xk):
#         with torch.no_grad():
#             ys = torch.stack((self.y,torch.zeros_like(self.y)),2)
#             Fy = torch.fft(ys,2)
#             AHy = torch.ifft(mul_c(conj(self.fpsf),Fy),2)[...,0]
#             aAHy = self.alpha * AHy
            
#             xkpaAHy = xk - aAHy

#             ts = torch.stack((xkpaAHy,torch.zeros_like(xkpaAHy)),2)
#             Ft = torch.fft(ts,2)
#             AHA = mul_c(conj(self.fpsf),self.fpsf)
#             I = torch.zeros_like(AHA)
#             I[...,0] = 1
#             ImaAHA = I - self.alpha*AHA
#             out = torch.ifft(div_c(Ft,ImaAHA),2)
#             return out[...,0]