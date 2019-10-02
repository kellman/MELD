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
# from mri import sense_adj

class dataloader(Dataset):
    def __init__(self, path, noise_std):
        super(dataloader,self).__init__()

        self.path = path
#         self.num_batches = num_batches
#         self.batch_size = batch_size
#         self.device = device
        self.noise_std = noise_std
        self.np_dtype = np.float32
        
    def __len__(self,):
        with h5py.File(self.path, 'r') as F:
            return F['imgs'].shape[0]
    
    def __getitem__(self, idx):
#         start_idx = self.batch_size * idx
#         end_idx = self.batch_size * (idx + 1)
#         indices = [idx for idx in range(start_idx,end_idx)]
        imgs, maps, masks = load_data(idx, self.path)
        meas = self._sim_data(imgs, maps, masks)
        imgs_0 = torch.tensor(cp.c2r(imgs).astype(self.np_dtype))
        maps_0 = torch.tensor(cp.c2r(maps).astype(self.np_dtype))
        meas_0 = torch.tensor(cp.c2r(meas).astype(self.np_dtype))
        mask_0 = torch.tensor(masks.astype(self.np_dtype))
#         adj = sense_adj(meas_0, maps_0, mask_0)
#         print(imgs.shape, maps.shape, masks.shape)
#         print(meas.shape)
#         return meas_0, maps_0, masks, imgs_0
        return imgs_0.squeeze(0), maps_0.squeeze(0), meas_0.squeeze(0), mask_0.squeeze(0)
    
    def _sim_data(self, imgs, maps, masks, ksp=None):

        # N, nc, nx, ny
        noise = np.random.randn(*maps.shape) + 1j * np.random.randn(*maps.shape)

#         if self.inverse_crime and ksp is None:
        meas = masks[:,None,...] * (fftnuc(imgs[:,None,...] * maps) + 1 / np.sqrt(2) * self.noise_std * noise)
        meas = fftmod(meas)
        return meas


def load_data(idx, data_file, gen_masks=False):
    with h5py.File(data_file, 'r') as F:
        imgs = np.array(F['imgs'][idx,...], dtype=np.complex)
        maps = np.array(F['maps'][idx,...], dtype=np.complex)
        if not gen_masks:
            try:
#                 print(F['masks'].shape)
                masks = np.array(F['masks'][idx,...], dtype=np.float)
            except:
                masks = poisson_mask(imgs.shape, seed=idx)
    if gen_masks:
        masks = poisson_mask(imgs.shape, seed=idx)

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