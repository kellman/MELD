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

class dataloader():
    def __init__(self, path, num_batches, batch_size, noise_std, device='cpu'):
        super(dataloader,self).__init__()

        self.path = path
        self.num_batches = num_batches
        self.batch_size = batch_size
        self.device = device
        self.noise_std = noise_std
        
    def __len__(self,):
        return self.num_batches * self.batch_size
    
    def __getitem__(self, idx):
        start_idx = self.batch_size * idx
        end_idx = self.batch_size * (idx + 1)
        indices = [idx for idx in range(start_idx,end_idx)]
        imgs, maps, masks = load_data(indices, self.path)
        meas = self._sim_data(imgs, maps, masks)
        print(imgs.shape, maps.shape, masks.shape)
        print(meas.shape)
        return meas, maps, masks, imgs
    
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