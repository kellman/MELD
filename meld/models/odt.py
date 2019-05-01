import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../utilities/')
from pytorch_complex import *
import scipy.io as sio

dtype = torch.float32
np_dtype = np.float32

mul_c  = ComplexMul().apply
div_c  = ComplexDiv().apply
abs_c  = ComplexAbs().apply
abs2_c = ComplexAbs2().apply 
exp_c  = ComplexExp().apply

def roll2(x,n):
    return torch.cat((x[:, -n:], x[:, :-n]), dim=1)

def fftshift2_tmp(x):
    N = [x.shape[0]//2, x.shape[1]//2]
    x = roll2(x,N[1])
    x = roll2(x.transpose(1,0),N[0]).transpose(1,0)
    return x

def fftshift2(x):
    real = x[:,:,0]
    imag = x[:,:,1]
    realout = fftshift2_tmp(real)
    imagout = fftshift2_tmp(imag)
    return torch.stack((realout,imagout),dim=2)

def ifftshift2_tmp(x):
    N = [x.shape[0]//2, x.shape[1]//2]
    x = roll2(x,N[1]+1)
    x = roll2(x.transpose(1,0),N[0]+1).transpose(1,0)
    return x

def ifftshift2(x):
    real = x[:,:,0]
    imag = x[:,:,1]
    realout = fftshift2_tmp(real)
    imagout = fftshift2_tmp(imag)
    return torch.stack((realout,imagout),dim=2)

class Multislice(nn.Module):
    def __init__(self, Np, na, wl, ps, mag, alpha, T=4, testFlag=False, device='cpu'):
        super(Multislice, self).__init__()
        
        self.Np = Np
        self.Npad = Np
        self.na = na
        self.wl = wl
        self.ps = ps
        self.mag = mag
        
        self.dz = self.wl/self.na**2 
        self.dz = 10 * self.ps/self.mag
        
        # setup sampling
        self.xx,self.yy,self.uxx,self.uyy = self.sampling()
        
        # setup pupil
        self.pupil = self.genPupil(self.na,self.wl).type(dtype)
        
        # setup propagation kernel
        self.propkern = self.genFrensel(self.dz).type(dtype).to(device)

        self.k = torch.zeros(self.Npad[0],self.Npad[1],2)
        self.k[...,1] = 2*np.pi*self.dz/self.wl
        self.k = self.k.type(dtype).to(device)
        
        self.T = T
        
        self.testFlag = testFlag
        self.alpha = nn.Parameter(torch.from_numpy(np.asarray([alpha]).astype(np_dtype))).requires_grad_(not self.testFlag)

 
    def sampling(self, device='cpu'):
        self.ps_eff = self.ps/self.mag
        self.FOV = [a*self.ps_eff for a in self.Np]

        # real space sampling
        x = (np.arange(self.Npad[0])-self.Npad[0]//2) * (self.ps_eff)
        y = (np.arange(self.Npad[1])-self.Npad[1]//2) * (self.ps_eff)
        xx,yy = np.meshgrid(x,y)

        # spatial frequency sampling
        ux = (np.arange(self.Npad[0])-self.Npad[0]//2) * (1/self.ps_eff/self.Npad[0])
        uy = (np.arange(self.Npad[1])-self.Npad[1]//2) * (1/self.ps_eff/self.Npad[1])
        uxx,uyy = np.meshgrid(ux,uy)
    
        return torch.from_numpy(xx).to(device),torch.from_numpy(yy).to(device),torch.from_numpy(uxx).to(device),torch.from_numpy(uyy).to(device)
    
    def genPupil(self,na,wl,device='cpu'):
        urr = np.sqrt(self.uxx**2 + self.uyy**2)
        pupil = 1. * (urr**2 < (na/wl)**2)
        pupil = fftshift2(torch.stack((pupil,torch.zeros_like(pupil)),2))
        return pupil.to(device)
    
    def genFrensel(self, dz, device='cpu'):
        urr = self.uxx**2 + self.uyy**2
        urr = urr.numpy()
        p = fftshift2(self.genPupil(self.na,self.wl).type(dtype))[...,0].numpy()
        kern = p * np.exp(p * 1j*2*np.pi* dz *(p*((1/self.wl)**2 - urr))**(1/2))
        kern = fftshift2(torch.stack((torch.from_numpy(np.real(kern)),torch.from_numpy(np.imag(kern))),2))
        return kern.to(device)

    def genPlaneWave(self, na_illum, device='cpu'):
        tilt = [a/self.wl for a in na_illum]
        dfx = (1/self.ps_eff/self.Npad[0])
        tilt = [np.round(a/dfx)*dfx for a in tilt]
        exp = 2 * np.pi * (tilt[0] * self.xx + tilt[1] * self.yy)
        wave = exp_c(torch.stack((torch.zeros_like(exp),exp),2))
        return wave.to(device)

    def measurement(self, O, na_illum, device='cpu'):
        
        # setup prop focus kernel
        self.propkern_focus = self.genFrensel(-1 * self.dz * ((O.shape[3]-1)/2)).type(dtype).to(device)
        
        # Setup Field: complex transmittion function
        T = torch.zeros(self.Npad[0],self.Npad[1],2,O.shape[3],dtype=dtype,device=device)
        for zz in range(O.shape[3]):
            T[...,zz] = exp_c(mul_c(self.k.to(device),O[...,zz])).type(dtype)
        
        # Compute Forward: multislice propagation
        Uconj = torch.zeros(self.Npad[0],self.Npad[1],2,O.shape[3],dtype=dtype)
        u = self.genPlaneWave(na_illum).type(dtype).to(device)
        for zz in range(O.shape[3]):
            # storage for backward
            Uconj[...,zz] = conj(u)
            
            u = mul_c(T[...,zz],u)
            
            if zz < O.shape[3] - 1:
#                 print('not last plane',zz)
                u = torch.fft(u,2)
                u = mul_c(u,self.propkern.to(device))
                u = torch.ifft(u,2)
                
        # Focus
        if O.shape[3] > 1:
#             print('Refocusing')
            u = torch.fft(u,2)
            u = mul_c(self.propkern_focus.to(device),u)
            u = torch.ifft(u,2)
            
        # Microscope's Point Spread Function
        u = torch.fft(u,2)
        u = mul_c(self.pupil.to(device),u)
        u = torch.ifft(u,2)
        
        # Camera Intensity Measurement
        est_intensity = abs2_c(u)
        return est_intensity
    
    def reverse(self, x, meas, na_illum, device='cpu'):
        z = x
        for _ in range(self.T):
            z = x - self.step(z, meas, na_illum, device=device)
        return z
    
    def forward(self, x, meas, na_illum, device='cpu'):
        return x + self.step(x, meas, na_illum, device=device)

    def step(self, x, meas, na_illum, device='cpu'):
        return -1 * self.alpha * self.grad(x, meas, na_illum, device=device)
    
    def grad(self, O, intensity, na_illum, device='cpu'):        
        # to device
        self.pupil = self.pupil.to(device)
        
        # setup prop focus kernel
        self.propkern_focus = self.genFrensel(-1 * self.dz * ((O.shape[3]-1)/2)).type(dtype).to(device)
        
        # Setup Field: complex transmittion function
        T = torch.zeros(self.Npad[0],self.Npad[1],2,O.shape[3],dtype=dtype,device=device)
        for zz in range(O.shape[3]):
            T[...,zz] = exp_c(mul_c(self.k.to(device),O[...,zz])).type(dtype)
        
        # Compute Forward: multislice propagation
#         Uconj = torch.zeros(self.Npad[0],self.Npad[1],2,O.shape[3],dtype=dtype,device=device)
        u = self.genPlaneWave(na_illum).type(dtype).to(device)
        for zz in range(O.shape[3]):
            # storage for backward
#             Uconj[...,zz] = conj(u)
            u = mul_c(T[...,zz],u)
            
            if zz < O.shape[3] - 1:
                u = torch.fft(u,2)
                u = mul_c(u,self.propkern.to(device))
                u = torch.ifft(u,2)
                
        # Focus
        if O.shape[3] > 1:
            u_ref = torch.fft(u,2)
            u_ref = mul_c(self.propkern_focus.to(device),u_ref)
            u_ref = torch.ifft(u_ref,2)
            
        # Microscope's Point Spread Function
        u_pup = torch.fft(u_ref,2)
        u_pup = mul_c(self.pupil.to(device),u_pup)
        u_pup = torch.ifft(u_pup,2)
        
        # Camera Intensity Measurement
        est_intensity = abs2_c(u_pup)
        
        # Compute Residual
        abs_field = abs_c(u_pup)
        sqrt_meas = torch.sqrt(intensity)
        comp = torch.stack((sqrt_meas * u_pup[...,0] / abs_field, sqrt_meas * u_pup[...,1] / abs_field),2)
        residual = u_pup - comp

        # Backpropagate through pupil
        u_bp = torch.fft(residual,2)
        u_bp = mul_c(conj(self.pupil),u_bp)
        u_bp = torch.ifft(u_bp,2)
        
        if O.shape[3] > 1:
            u_bp = torch.fft(u_bp,2)
            u_bp = mul_c(conj(self.propkern_focus.to(device)),u_bp)
            u_bp = torch.ifft(u_bp,2)
        
        # Backpropagate through layers
        obj_grad = torch.zeros(self.Npad[0],self.Npad[1],2,O.shape[3],dtype=dtype,device=device)
        
        for zz in range(O.shape[3]-1,-1,-1):
            obj_grad[...,zz] = mul_c(u_bp,conj(u)) #[...,zz])
            obj_grad[...,zz] = mul_c(obj_grad[...,zz],conj(T[...,zz])) # apply conj to T rather than store
            obj_grad[...,zz] = mul_c(obj_grad[...,zz],conj(self.k.to(device)))
            
            if zz > 0:
                u_bp = mul_c(u_bp,conj(T[...,zz]))
                
                # backpropagate gradient in z
                u_bp = torch.fft(u_bp,2)
                u_bp = mul_c(conj(self.propkern.to(device)),u_bp)
                u_bp = torch.ifft(u_bp,2)
                
                # backpropagate field in z
                u = torch.fft(u,2)
                u = mul_c(conj(self.propkern.to(device)),u)
                u = torch.ifft(u,2)
                u = div_c(u,T[...,zz-1])

        return obj_grad
    
    def loss(self,O,meas,na_list,device='cpu'):
        est = torch.zeros(self.Np[0],self.Np[1],na_list.shape[0],dtype=dtype,device=device)
        for ii in range(na_list.shape[0]):
            a,b,c,m = self.fb(O,est[...,ii],na_list[ii,:])
            est[...,ii] = m
        res = torch.sqrt(meas) - torch.sqrt(est)
        return torch.sum(res**2)