import numpy as np
import torch
from meld.model import waveletsoftthr
from meld.recon import makeNetwork

from mri import MultiChannelMRI

np_dtype = np.float32
dtype = torch.float32

class model():
    def __init__(self, metadata, testFlag=False, device='cpu'):
        self.metadata = metadata
        self.num_unrolls = metadata['num_unrolls']
        self.alpha = metadata['alpha']
        self.lamb = metadata['lamb']
        self.testFlag = testFlag
        
        self._make_model(device=device)
        
        
    def _make_model(self, device='cpu'):
        self.mri_model = MultiChannelMRI(alpha = self.alpha,
                                         mu = self.lamb,
                                         testFlag = self.testFlag,
                                         device = device)
        
        self.prox = waveletsoftthr.shrinkage(thr = self.lamb,
                                             scale = self.alpha,
                                             testFlag = self.testFlag)
        
        self.grad.to(device)
        self.prox.to(device)
        with torch.no_grad():
            self.network = makeNetwork([self.grad, self.prox], self.metadata['num_unrolls'])
        
    
    def _initialize_model(self, device='cpu'):
        return

    def projection(self,):
        pass

    def initialize(self, device='cpu'):
        # set data for model
        return 