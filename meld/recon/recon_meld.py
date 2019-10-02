import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import sys

# making networks
def GD(grad):
    return nn.ModuleList([grad])

def PGD(grad,prox):
    return nn.ModuleList([grad,prox])

def genNetwork(layer,N):
    return nn.ModuleList([layer for _ in range(N)])

def makeNetwork(opList, N):
    return genNetwork(nn.ModuleList(opList), N)

class UnrolledNetwork():
    def __init__(self, network, xtest, memlimit, loss=None, setupFlag=True, device='cpu'):
        super(UnrolledNetwork, self).__init__()
        self.network = network
        self.xtest = xtest
        self.memlimit = memlimit
        self.gpu_device = device
        
        # setup hybrid checkpointing
        self.meldFlag = True # default
        self.cpList = [-1] # default
        if setupFlag: self.setup()
        
        # setup loss function
        if loss is None:
            self.lossFunc = lambda x,truth : torch.mean((x-truth)**2)
        else:
            self.lossFunc = loss
        
    def setup(self):
        for p_ in self.network.parameters(): p_.requires_grad_(True)

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
        print('Memory per iteration:', mem, 'MB')

        # assess how many checkpoints
        N = len(self.network)
        totalmem = mem * N
        print('Total memory:', totalmem, 'MB')

        if totalmem > self.memlimit:
            print('Requires memory-efficient learning!')
            self.M = np.ceil(totalmem/self.memlimit)
            print('Checkpointing every:',int(self.M))
            self.cpList = list(range(int(self.M),int(N-1),int(self.M)))
            self.meldFlag = True
        else:
            self.cpList = [-1]
            self.M = 1
            self.meldFlag = False
            
            
    def evaluate(self, x0, interFlag=False, testFlag=True):
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
                x = layer.forward(x,device=self.gpu_device)

            if interFlag:
                Xall[ii,...] = x
        return x, self.Xcp, Xall
    
    
    def loss_eval(self, x0, truth, testFlag=True):
        with torch.no_grad():
            x, _, _ = self.evaluate(x0, testFlag=True)
            loss = self.lossFunc(x, truth)
            return x, loss

    
    def differentiate(self,xN,qN,interFlag=False):
        if interFlag:
            size = [len(self.network)] + [a for a in xN.shape]
            X = torch.zeros(size, device=self.gpu_device)
        else:
            X = None

        for p_ in self.network.parameters(): p_.requires_grad_(True) 

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
                if cp >= 0 and ii == self.cpList[cp]:
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
    
    
    def forward(self, x0, truth, interFlag=False, testFlag=False):
        # memory-efficient learned design 
        if self.meldFlag:
            # evaluate network
            with torch.no_grad():
                xN,Xcp,Xforward = self.evaluate(x0, interFlag=interFlag, testFlag=testFlag)
        
            # evaluate loss
            xN = xN.detach().requires_grad_(True)
            # evaluate loss function
            loss = self.lossFunc(xN,truth)
            loss.backward()
            qN = xN.grad

            # reverse-mode differentiation
            Xbackward = self.differentiate(xN,qN,interFlag=interFlag)
            
        # standard backpropagation
        else:
            # evaluate network
            xN,Xcp,Xforward = self.evaluate(x0, interFlag=interFlag, testFlag=False)
            # evaluate loss function
            loss = self.lossFunc(xN,truth)
            # reverse-mode differentiation
            loss.backward()
            Xbackward = None
            Xforward = None
            
        return xN, loss, Xforward, Xbackward # returned for testing/debugging purposes