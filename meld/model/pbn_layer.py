import torch
import torch.nn as nn

class pbn_layer(nn.Module):
    def __init__(self, ):
         super(pbn_layer, self).__init__()
    
    def forward(self, x, device='cpu'):
        return
    
    def reverse(self, x, device='cpu'):
        print('Must implement reverse for memory-efficient learning')
        return
           