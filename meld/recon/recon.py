

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import sys

# making physics-based networks
def GD(grad):
    return nn.ModuleList([grad])

def PGD(grad,prox):
    return nn.ModuleList([grad,prox])

def genNetwork(layer,N):
    return nn.ModuleList([layer for _ in range(N)])

def makeNetwork(opList, N):
    return genNetwork(nn.ModuleList(opList), N)

# network evaluator (forward/backward)
class UnrolledNetwork():
    def __init__(self, model, xtest, memlimit, loss=None, setupFlag=True, ckptFlag=0, unsuperFlag=False, device='cpu',dtype=torch.float32):
        super(UnrolledNetwork, self).__init__()
        self.model = model
        self.network = self.model.network
        self.xtest = xtest
        self.memlimit = memlimit
        self.gpu_device = device
        self.dtype = dtype

        # setup hybrid checkpointing
        self.meldFlag = True # default
        self.cpList = [-1] # default
        
        # setup loss function
        self.unsuperFlag = unsuperFlag
        if loss is None:
            self.lossFunc = lambda x,truth,device : torch.mean((x-truth)**2)
        else:
            self.lossFunc = loss
            
        if setupFlag: 
            self.setup()
        
        if ckptFlag != 0:
            N = len(self.network)
            self.cpList = np.sort(list(np.linspace(int(N-ckptFlag),1,int(ckptFlag), dtype=np.int32)))
        
    def setup(self):
        for p_ in self.network.parameters(): p_.requires_grad_(True)
            
        # compute storage memory of single checkpoint
        torch.cuda.empty_cache()
        startmem = torch.cuda.memory_cached(self.gpu_device)
        self.xtest = self.xtest.to(self.gpu_device)
        endmem = torch.cuda.memory_cached(self.gpu_device)
        mem3 = (endmem - startmem) / 1024**2
        print('Memory per checkpoint: {0:d}MB'.format(int(mem3)))
        torch.cuda.empty_cache()
        

        # test memory requirements (offset + single layer)
        torch.cuda.empty_cache()
        startmem = torch.cuda.memory_cached(self.gpu_device)
        
        x = self.xtest
        for sub in self.network[0]:
            x = sub(x,device=self.gpu_device)
        if not self.unsuperFlag:
            loss = self.lossFunc(x, self.xtest, self.gpu_device)
        else:
            loss = self.lossFunc(x, self.gpu_device)
        loss.backward()

        endmem = torch.cuda.memory_cached(self.gpu_device)
        mem1 = (endmem - startmem) / 1024**2
        print('Memory per layer: {0:d}MB'.format(int(mem1)))
        
#         # test memory requirements (offset + two layer)
#         torch.cuda.empty_cache()
#         startmem = torch.cuda.memory_cached(self.gpu_device)
#         x = self.xtest
#         for layers in self.network[:2]:
#             for sub in layers:
#                 x = sub(x,device=self.gpu_device)
#         loss = self.lossFunc(x,self.xtest)
#         loss.backward()
#         endmem = torch.cuda.memory_cached(self.gpu_device)
#         mem2 = (endmem - startmem) / 1024**2
#         print('Memory per two layer: {0:d}MB'.format(int(mem2)))
#         torch.cuda.empty_cache()
        
        # assess how many checkpoints
        N = len(self.network)
        totalmem = (mem1) * N
        print('Total memory:', totalmem, 'MB')

        if totalmem > self.memlimit:
            print('Requires memory-efficient learning!')
            self.M = np.ceil(totalmem/self.memlimit)
            print('Checkpointing every:',int(self.M))
            self.cpList = list(range(1,int(N-self.M),int(self.M)))
            self.meldFlag = True
            print('Checkpoints:',self.cpList)
        else:
            self.cpList = [-1]
            self.M = 1
            self.meldFlag = False
            
            
    def evaluate(self, x0, interFlag=False, testFlag=True):
        # setup storage (for debugging)
        if interFlag:
            size = [len(self.network)] + [a for a in x0.shape]
            Xall = torch.zeros(size,device=self.gpu_device,dtype=self.dtype)
        else:
            Xall = None

        # setup checkpointing
        if self.cpList is not []:
            size = [len(self.cpList)] + [a for a in x0.shape]
            self.Xcp = torch.zeros(size,device=self.gpu_device,dtype=self.dtype)
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
            if not self.unsuperFlag:
                loss = self.lossFunc(x, truth, self.gpu_device)
            else:
                loss = self.lossFunc(x, self.gpu_device)
            return x, loss

    
    def differentiate(self,xN,qN,interFlag=False):
        if interFlag:
            size = [len(self.network)] + [a for a in xN.shape]
            X = torch.zeros(size, device=self.gpu_device,dtype=self.dtype)
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
            xk.backward(qk, create_graph=True, retain_graph=True)
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
            if not self.unsuperFlag:
                loss = self.lossFunc(xN, truth, self.gpu_device)
            else:
                loss = self.lossFunc(xN, self.gpu_device)
            
            loss.backward()
            qN = xN.grad

            # reverse-mode differentiation
            Xbackward = self.differentiate(xN,qN,interFlag=interFlag)
            
        # standard backpropagation
        else:
            # evaluate network
            xN,Xcp,Xforward = self.evaluate(x0, interFlag=interFlag, testFlag=False)
            # evaluate loss function
            if not self.unsuperFlag:
                loss = self.lossFunc(xN, truth, self.gpu_device)
            else:
                loss = self.lossFunc(xN, self.gpu_device)
            # reverse-mode differentiation
            loss.backward()
            Xbackward = None
            Xforward = None
            
        return xN, loss, Xforward, Xbackward # returned for testing/debugging purposes