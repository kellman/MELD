import numpy as np
import torch
from meld.model import waveletsoftthr
from meld.model import pytorch_proximal, waveletsoftthr
from meld.recon import makeNetwork

from mri import MultiChannelMRI

np_dtype = np.float32
dtype = torch.float32

class model():
    def __init__(self, metadata, testFlag=False, device='cpu', debug=False, shrinkage_params=None, num_layers=5, num_filters=64, ndim=2, conj_back=True):
        self.metadata = metadata
        self.num_unrolls = metadata['num_unrolls']
        self.alpha = metadata['alpha']
        self.lamb = metadata['lamb']
        self.testFlag = testFlag
        if shrinkage_params:
            self.thr, self.scale = shrinkage_params
        self.num_layers, self.num_filters= num_layers, num_filters
        self.ndim = ndim
        self.conj_back=conj_back
        
        self._make_model(device=device, debug=debug)
        
        
    def _make_model(self, device='cpu', debug=False):
        if debug:
            self.prox = waveletsoftthr.shrinkage(self.thr, self.scale)
        else:
            self.prox = pytorch_proximal.ResNet4(num_channels=self.num_filters, num_layers=self.num_layers, dims=self.ndim)
        self.prox.to(device)
        
        self.mri_model = MultiChannelMRI(alpha = self.alpha,
                             mu = self.lamb,
                             testFlag = self.testFlag,
                             device = device, ndim=self.ndim, conj_back=self.conj_back)
        self.mri_model.to(device)
        
#         with torch.no_grad():
        self.network = makeNetwork([self.mri_model, self.prox], self.metadata['num_unrolls'])
        
    
    def _initialize_model(self, device='cpu'):
        return

    def projection(self,):
        pass

    def initialize(self, imgs, maps, meas, mask, device='cpu'):
        # set data for model
        adjoint = self.mri_model.batch(imgs, maps, meas, mask, device)
        return adjoint