import os
import argparse
import time
import numpy as np
import torch
import scipy
import scipy.fftpack
import h5py
import pathlib
import time
import lib_complex as cp
from torch.utils.data import Dataset
import mri
import math
# from mri import sense_adj

class dataloader(Dataset):
    def __init__(self, path, noise_std, mask_path=None, maps_path=None, device='cpu'):
        super(dataloader,self).__init__()

        self.path = path
#         self.num_batches = num_batches
#         self.batch_size = batch_size
#         self.device = device
        self.noise_std = noise_std
        self.np_dtype = np.float32
        self.mask_path = mask_path
        self.maps_path = maps_path
        self.device = device
        
    def __len__(self,):
        with h5py.File(self.path, 'r') as F:
            return F['imgs'].shape[0]
    
    def __getitem__(self, idx):
        # add caching here?
        # simulation on or off gpu?
#         start_idx = self.batch_size * idx
#         end_idx = self.batch_size * (idx + 1)
#         indices = [idx for idx in range(start_idx,end_idx)]
        imgs, maps, masks = load_data(idx, self.path, mask_path=self.mask_path, maps_path=self.maps_path)
#         print(imgs.shape, maps.shape, masks.shape)
#         meas = self._sim_data(imgs, maps, masks)
        masks = fftshift(masks)
        imgs_0 = torch.tensor(cp.c2r(imgs).astype(self.np_dtype))
        maps_0 = torch.tensor(cp.c2r(maps).astype(self.np_dtype))
        mask_0 = torch.tensor(masks.astype(self.np_dtype))
        if (len(maps_0.size()) == 6):
            maps_0 = maps_0.permute(0, 2, 1, 3, 4, 5)
        meas_0 = self._sim_data(imgs_0, maps_0, mask_0)
#         adj = sense_adj(meas_0, maps_0, mask_0)
#         print(imgs.shape, maps.shape, masks.shape)
#         print(meas.shape)
#         return meas_0, maps_0, masks, imgs_0
        return imgs_0.squeeze(0), maps_0.squeeze(0), meas_0.squeeze(0), mask_0.squeeze(0)
    
    def _sim_data(self, imgs, maps, masks, ksp=None):

        # N, nc, nx, ny
#         print(imgs.size(), maps.size(), masks.size())
       
        with torch.no_grad():
            noise = torch.rand(*maps.size()) * (1/np.sqrt(2))*self.noise_std
            mapped = mri.maps_forw(imgs, maps)
            if (len(maps.size()) == 6):                
                mapped_fft = mri.fft_forw(mapped, ndim=3)
            elif (len(maps.size()) == 5):
                mapped_fft = mri.fft_forw(mapped, ndim=2)                
            mapped_fft_noisy = mapped_fft + noise
            masked_mapped_fft_noisy = mri.mask_forw(mapped_fft_noisy, masks)

#         if self.inverse_crime and ksp is None:
        return masked_mapped_fft_noisy


def load_data(idx, data_file, gen_masks=False, mask_path=None, maps_path=None):
    with h5py.File(data_file, 'r') as F:
        imgs = np.array(F['imgs'][idx,...], dtype=np.complex)
        if maps_path is None:
            maps = np.array(F['maps'][idx,...], dtype=np.complex)
        if not gen_masks:
            try:
                if mask_path is None:
                    masks = np.array(F['masks'][idx,...], dtype=np.float)
            except:
                masks = poisson_mask(imgs.shape, seed=idx)
    if mask_path:
        with h5py.File(mask_path, 'r') as F:
            masks = np.array(F['masks'][idx,...], dtype=np.complex)
            
    if gen_masks:
        masks = poisson_mask(imgs.shape, seed=idx)
        
    if maps_path:    
        with h5py.File(maps_path, 'r') as F:
            maps = np.array(F['maps'][idx,...], dtype=np.complex)

    if len(masks.shape) == 2:
        imgs, maps, masks = imgs[None,...], maps[None,...], masks[None,...]
    return imgs, maps, masks

def fftmod(out):
    return ifftnc(fftshift(fftnc(out)))
    
def fftshift(x):
    if len(x.shape[1:]) == 3:
        axes = (-3, -2, -1)
        return scipy.fftpack.fftshift(x, axes=axes)
    else:
        axes = (-2, -1)
        return scipy.fftpack.fftshift(x, axes=axes)

def ifftshift(x):
    if len(x.shape[1:]) == 3:
        axes = (-3, -2, -1)
        return scipy.fftpack.ifftshift(x, axes=axes)
    else:
        axes = (-2, -1)
        return scipy.fftpack.ifftshift(x, axes=axes)

def fftnc(x):
    return fftshift(fftn(ifftshift(x)))

def ifftnc(x):
    return ifftshift(ifftn(fftshift(x)))

def fftnuc(x):
    factor = len(x.shape[1:])
    return fftnc(x) / np.sqrt(np.prod(x.shape[-1*factor:]))

def ifftnuc(x):
    factor = len(x.shape[1:])
    return ifftnc(x) / np.sqrt(np.prod(x.shape[-1*factor:]))

def fftn(x):
    if len(x.shape[1:]) == 3:
        axes = (-3, -2, -1)
        return scipy.fftpack.fftn(x, axes=axes)
    else:
        axes = (-2, -1)
        return scipy.fftpack.fft2(x, axes=axes)

def ifftn(x):
    if len(x.shape[1:]) == 3:
        axes = (-3, -2, -1)
        return scipy.fftpack.ifftn(x, axes=axes)
    else:
        axes = (-2, -1)
        return scipy.fftpack.ifft2(x, axes=axes)
