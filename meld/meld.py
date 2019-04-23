import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import sys

sys.path.append('/home/kellman/Workspace/PYTHON/Pytorch_Physics_Library/Utilities/')
from utility import *

def GD(grad):
    return nn.ModuleList([grad])

def PGD(grad,prox):
    return nn.ModuleList([grad,prox])

def genNetwork(layer,N):
    return nn.ModuleList([layer for _ in range(N)])

def feedforward(network,x0,iterFunc=None,interFlag=False,testFlag = True,device='cpu'):
    if interFlag:
        size = [len(network)] + [a for a in x0.shape]
        X = torch.zeros(size)
    else:
        X = None
        
    for p_ in network.parameters(): p_.requires_grad_(not testFlag)
        
    x = x0
    
    for ii in range(len(network)):
 
        for layer in network[ii]:
            x = layer.forward(x,device=device)
            
            
        if interFlag:
#             print('Saving:',torch.sum(x.cpu()))
            X[ii,...] = x
            
        if iterFunc is not None:
            iterFunc(x)
            
    return x,X
        
def feedbackward(network,xN,qN,interFlag=False,device='cpu'):
    if interFlag:
        size = [len(network)] + [a for a in xN.shape]
        X = torch.zeros(size,device=device)
    else:
        X = None
        
    for p_ in network.parameters(): p_.requires_grad_(True) 
    # gradients should just accumalate? It does :) 

    xkm1 = xN
    qk = qN
    for ii in range(len(network)-1,-1,-1):
        # reverse sequence
        with torch.no_grad():
            if interFlag:
                X[ii,...] = xkm1
            for jj in range(len(network[ii])-1,-1,-1):
                layer = network[ii][jj]
                xkm1 = layer.reverse(xkm1,device=device)
            
        # forward sequece
        xkm1 = xkm1.detach().requires_grad_(True)
        xk = xkm1
        for layer in network[ii]:
            xk = layer.forward(xk,device=device)
        
        # backward call
        xk.backward(qk)
        with torch.no_grad():
            qk = xkm1.grad
    return X

def feedforward_cp(network,x0,cpList=[],iterFunc=None,interFlag=False,testFlag = True, device='cpu'):
    # setup storage (for debugging)
    if interFlag:
        size = [len(network)] + [a for a in x0.shape]
        Xall = torch.zeros(size,device=device)
    else:
        Xall = None
      
    # setup checkpointing 
    if cpList is not []:
        size = [len(cpList)] + [a for a in x0.shape]
        Xcp = torch.zeros(size,device=device)
    else:
        Xcp = None
    cp = 0
        
    for p_ in network.parameters(): p_.requires_grad_(not testFlag)
        
    x = x0
    
    for ii in range(len(network)):
        if cp < len(cpList) and ii == cpList[cp]:
            print(ii,cpList[cp],cp,len(cpList))
            Xcp[cp,...] = x
            cp += 1

        for layer in network[ii]:
            x = layer.forward(x,device=device)
            
        if interFlag:
            Xall[ii,...] = x
            
        if iterFunc is not None:
            iterFunc(x)
            
    return x,Xcp,Xall

def feedbackward_cp(network,xN,qN,cpList=[],Xcp=None,interFlag=False,device='cpu'):
    if interFlag:
        size = [len(network)] + [a for a in xN.shape]
        X = torch.zeros(size, device = device)
    else:
        X = None
        
    for p_ in network.parameters(): p_.requires_grad_(True) 
    # gradients should just accumalate?
    
    # checkpointing flag
    cp = len(cpList)-1
    print(cp)
    
    xkm1 = xN
    qk = qN
    for ii in range(len(network)-1,-1,-1):
        # reverse sequence
        with torch.no_grad():
            
            # checkpointing
            if cp > 0 and ii == cpList[cp]:
                print('Using Checkpoint:',ii)
                xkm1 = Xcp[cp,...]
                cp -= 1
                
            # calculate inverse
            else:
                for jj in range(len(network[ii])-1,-1,-1):
                    layer = network[ii][jj]
                    xkm1 = layer.reverse(xkm1,device=device)
                
            if interFlag:
                X[ii,...] = xkm1
                
        # forward sequece
        xkm1 = xkm1.detach().requires_grad_(True)
        xk = xkm1
        for layer in network[ii]:
            xk = layer.forward(xk,device=device)
        
        # backward call
        xk.backward(qk)
        with torch.no_grad():
            qk = xkm1.grad
            
    return X


def setup(layer,x0,memlimit,N,gpu_device):
    
    for p_ in layer.parameters(): p_.requires_grad_(True)

    # test memory
    torch.cuda.empty_cache()
    startmem = torch.cuda.memory_cached(gpu_device)
    
    x = x0
    for sub in layer:
        x = sub(x)
    x.backward(x0)
    
    endmem = torch.cuda.memory_cached(gpu_device)
    mem = endmem-startmem
    mem = mem / 1024**2
    print('Memory per iteration:', mem, 'MB')
    
    # assess how many checkpoints
    totalmem = mem * N
    print('Total memory:', totalmem, 'MB')
    
    if totalmem > memlimit:
        print('Requires memory-efficient learning!')
        M = np.ceil(totalmem/memlimit)
        print('Checkpointing every:',M)
        cpList = list(range(int(M),int(N-1),int(M)))
#         cpList = np.linspace(M,N-1,(N-2*M-1) / M).tolist()
    else:
        print('Should just use standard backward...')
        cpList = list(range(0,int(N-1)))
        M = 1
   
    return cpList,mem,totalmem,M

class unroll(nn.Module):
    def __init__(self, network, xtest, memlimit, gpu_device='cpu'):
        super(unroll, self).__init__()
        self.network = network
        self.xtest = xtest
        self.memlimit = memlimit
        self.gpu_device = gpu_device
        
        # setup hybrid checkpointing
        self.setup()
        
    def setup(self):
        for p_ in self.network[0].parameters(): p_.requires_grad_(True)

        # test memory requirements
        torch.cuda.empty_cache()
        startmem = torch.cuda.memory_cached(self.gpu_device)
        
        x = self.xtest
        for sub in self.network[0]:
            x = sub(x,device=self.gpu_device)
        x.backward(self.xtest)

        endmem = torch.cuda.memory_cached(self.gpu_device)
        mem = endmem-startmem
        mem = mem / 1024**2
#         print('Memory per iteration:', mem, 'MB')

        # assess how many checkpoints
        N = len(self.network)
        totalmem = mem * N
#         print('Total memory:', totalmem, 'MB')

        if totalmem > self.memlimit:
            print('Requires memory-efficient learning!')
            self.M = np.ceil(totalmem/self.memlimit)
            print('Checkpointing every:',int(self.M))
            self.cpList = list(range(int(self.M),int(N-1),int(self.M)))
        else:
#             print('Should just use standard backward...')
            self.cpList = [-1] #list(range(0,int(N-1)))
            self.M = 1
            
    def evaluate(self, x0, interFlag=False, testFlag = True):
        # setup storage (for debugging)
        if interFlag:
            size = [len(self.network)] + [a for a in x0.shape]
            Xall = torch.zeros(size,device=self.gpu_device)
        else:
            Xall = None

        # setup checkpointing
        if self.cpList is not []:
            size = [len(self.cpList)] + [a for a in x0.shape]
            self.Xcp = torch.zeros(size,device=self.gpu_device)
        else:
            self.Xcp = None
        cp = 0

        for p_ in self.network.parameters(): p_.requires_grad_(not testFlag)

        x = x0

        for ii in range(len(self.network)):
            if cp < len(self.cpList) and ii == self.cpList[cp]:
                self.Xcp[cp,...] = x
                cp += 1

            for layer in self.network[ii]:
#                 print(layer)
                x = layer.forward(x,device=self.gpu_device)

            if interFlag:
                Xall[ii,...] = x

        return x,self.Xcp,Xall

    def differentiate(self,xN,qN,interFlag=False):
        if interFlag:
            size = [len(self.network)] + [a for a in xN.shape]
            X = torch.zeros(size, device=self.gpu_device)
        else:
            X = None

        for p_ in self.network.parameters(): p_.requires_grad_(True) 
        # gradients should just accumalate?

        # checkpointing flag
        cp = len(self.cpList)-1

        xkm1 = xN
        qk = qN
        for ii in range(len(self.network)-1,-1,-1):
            # reverse sequence
            with torch.no_grad():
                
                if interFlag:
                    X[ii,...] = xkm1

                # checkpointing
                if cp > 0 and ii == self.cpList[cp]:
#                     print('Using Checkpoint:',ii)
                    xkm1 = self.Xcp[cp,...]
                    cp -= 1

                # calculate inverse
                else:
                    for jj in range(len(self.network[ii])-1,-1,-1):
                        layer = self.network[ii][jj]
                        xkm1 = layer.reverse(xkm1,device=self.gpu_device)

                
            # forward sequece
            xkm1 = xkm1.detach().requires_grad_(True)
            xk = xkm1
            for layer in self.network[ii]:
                xk = layer.forward(xk,device=self.gpu_device)

            # backward call
            xk.backward(qk)
            with torch.no_grad():
                qk = xkm1.grad

        return X
    
    def forward(self, x0, xT, interFlag=False, testFlag=False):
        # evaluate network
        xN,Xcp,Xforward = self.evaluate(x0, interFlag=interFlag, testFlag=testFlag)
        
        # evaluate loss
        xN = xN.detach().requires_grad_(True)
        loss = torch.sum((xN-xT)**2)
        loss.backward()
        qN = xN.grad
        
        # reverse-mode differentiation
        Xbackward = self.differentiate(xN,qN,interFlag=interFlag)
        
        return xN, loss, Xforward, Xbackward # returned for testing/debugging purposes
        