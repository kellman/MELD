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
            xN,Xcp,Xforward = self.evaluate(x0, interFlag=False, testFlag=False)
            # evaluate loss function
            loss = self.lossFunc(xN,truth)
            # reverse-mode differentiation
            loss.backward()
            Xbackward = None
            Xforward = None
            
        return xN, loss, Xforward, Xbackward # returned for testing/debugging purposes
    
    
    
    
### Antiquated and simplifed ###    
    
# Evaluation of network
# class UnrolledNetwork(torch.autograd.Function):

#     @staticmethod
#     def forward(ctx, *args):
#         network, x0, cpList, interFlag, testFlag, device = args[-6], args[-5], args[-4], args[-3], args[-2], args[-1]
#         x, _ = feedforward(network, x0, interFlag=interFlag, testFlag=testFlag, device=device)
#         Xcp = None
# # #         if interFlag:
# # #             size = [len(network)] + [a for a in x0.shape]
# # #         else:
# # #             size = [0]
# # #         Xall = torch.zeros(size, device=device)

# #         # setup checkpointing
# # #         if cpList is not []:
# # #             size = [len(cpList)] + [a for a in x0.shape]
# # #             Xcp = torch.zeros(size, device=device)
# # #         else:
# # #             Xcp = None
            
# #         # setup learnable parameters
# #         for p_ in network.parameters(): 
# #             p_.requires_grad_(not testFlag)
# #             print(p_.shape, p_.requires_grad)
# #         x = torch.autograd.Variable(x0, requires_grad=True)
# #         cp = 0
# # #         with torch.no_grad():
# #         for ii in range(len(network)):
# # #             if cp < len(cpList) and ii == cpList[cp]:
# # #                 Xcp[cp,...] = x
# # #                 cp += 1
# #             for op in network[ii]:
# #                 print(ii, op)
# #                 x = op.forward(x, device=device)d
# # #             if interFlag:
# # #                 Xall[ii,...] = x
# #         print(x.grad_fn.next_function)
#         ctx.save_for_backward(x0)
#         ctx.x, ctx.network, ctx.Xcp, ctx.cpList, ctx.interFlag, ctx.device = x, network, Xcp, cpList, interFlag, device
#         print('finished with forward')
# #         x_vars = torch.autograd.Variable(x, requires_grad=True)
#         return x# , Xall
        
#     @staticmethod
#     def backward(ctx, *grad_output):
#         print('calling backwards')
#         x, network, Xcp, cpList, interFlag, device = ctx.x, ctx.network, ctx.Xcp, ctx.cpList, ctx.interFlag, ctx.device
#         xN = ctx.saved_tensor
#         qN = grad_output
        
#         if interFlag:
#             size = [len(network)] + [a for a in xN.shape]
#             X = torch.zeros(size, device=device)
#         else:
#             X = None

#         for p_ in network.parameters(): p_.requires_grad_(True) 

#         # checkpointing flag
#         cp = len(cpList)-1

#         xkm1 = xN
#         qk = qN
#         for ii in range(len(network)-1,-1,-1):
#             # reverse sequence (graph off)
#             with torch.no_grad():
#                 if interFlag:
#                     X[ii,...] = xkm1

#                 # checkpointing
#                 if cp >= 0 and ii == cpList[cp]:
#                     # print('Using Checkpoint:',ii)
#                     xkm1 = Xcp[cp,...]
#                     cp -= 1

#                 # calculate reverse
#                 else:
#                     for jj in range(len(network[ii])-1,-1,-1):
#                         op = network[ii][jj]
#                         xkm1 = op.reverse(xkm1, device=device)
                
#             # forward sequence (graph on)
#             xkm1 = xkm1.detach().requires_grad_(True)
#             xk = xkm1
#             for op in network[ii]:
#                 xk = op.forward(xk, device=device)

#             # backward call (graph off)
#             xk.backward(qk)
#             with torch.no_grad():
#                 qk = xkm1.grad
#         return X

# def recon_meld(network, x0, cpList = None, meldFlag=False, interFlag=False, testFlag=True, device='cpu'):
#     if meldFlag:
#         if cpList is None: cpList = []
# #         xN, Xall = UnrolledNetwork.apply(network, x0, cpList, interFlag, testFlag, device)
#         xN = UnrolledNetwork.apply(network, x0, cpList, interFlag, testFlag, device)
#         var_xN = torch.autograd.Variable(xN, requires_grad=True)
#         return torch.unsqueeze(var_xN,0)
# #     else:
# #         var_xN, Xall = feedforward(network, x0, interFlag=interFlag, testFlag=testFlag, device=device)
#     # returning
# #     if interFlag:
# #         return torch.unsqueeze(var_xN,0)# , Xall
# #     else:
# #         return torch.unsqueeze(var_xN,0)
     
# def feedforward(network, x0, iterFunc=None, interFlag=False, testFlag = True, device='cpu'):
#     if interFlag:
#         size = [len(network)] + [a for a in x0.shape]
#         X = torch.zeros(size)
#     else:
#         X = None
        
#     for p_ in network.parameters(): p_.requires_grad_(not testFlag)
        
#     x = x0
    
#     for ii in range(len(network)):
 
#         for op in network[ii]:
#             x = op.forward(x,device=device)
            
#         if interFlag:
#             X[ii,...] = x
            
#         if iterFunc is not None:
#             iterFunc(x)
#     return x, X
        
# def feedbackward(network,xN,qN,interFlag=False,device='cpu'):
#     if interFlag:
#         size = [len(network)] + [a for a in xN.shape]
#         X = torch.zeros(size,device=device)
#     else:
#         X = None
        
#     for p_ in network.parameters(): p_.requires_grad_(True) 
#     # gradients should just accumalate? It does :) 

#     xkm1 = xN
#     qk = qN
#     for ii in range(len(network)-1,-1,-1):
#         # reverse sequence
#         with torch.no_grad():
#             if interFlag:
#                 X[ii,...] = xkm1
#             for jj in range(len(network[ii])-1,-1,-1):
#                 layer = network[ii][jj]
#                 xkm1 = layer.reverse(xkm1,device=device)
            
#         # forward sequece
#         xkm1 = xkm1.detach().requires_grad_(True)
#         xk = xkm1
#         for layer in network[ii]:
#             xk = layer.forward(xk,device=device)
        
#         # backward call
#         xk.backward(qk)
#         with torch.no_grad():
#             qk = xkm1.grad
#     return X

# def feedforward_cp(network,x0,cpList=[],iterFunc=None,interFlag=False,testFlag = True, device='cpu'):
#     # setup storage (for debugging)
#     if interFlag:
#         size = [len(network)] + [a for a in x0.shape]
#         Xall = torch.zeros(size,device=device)
#     else:
#         Xall = None
      
#     # setup checkpointing 
#     if cpList is not []:
#         size = [len(cpList)] + [a for a in x0.shape]
#         Xcp = torch.zeros(size,device=device)
#     else:
#         Xcp = None
#     cp = 0
        
#     for p_ in network.parameters(): p_.requires_grad_(not testFlag)
        
#     x = x0
    
#     for ii in range(len(network)):
#         if cp < len(cpList) and ii == cpList[cp]:
#             print(ii,cpList[cp],cp,len(cpList))
#             Xcp[cp,...] = x
#             cp += 1

#         for layer in network[ii]:
#             x = layer.forward(x,device=device)
            
#         if interFlag:
#             Xall[ii,...] = x
            
#         if iterFunc is not None:
#             iterFunc(x)
            
#     return x,Xcp,Xall

# def feedbackward_cp(network,xN,qN,cpList=[],Xcp=None,interFlag=False,device='cpu'):
#     if interFlag:
#         size = [len(network)] + [a for a in xN.shape]
#         X = torch.zeros(size, device = device)
#     else:
#         X = None
        
#     for p_ in network.parameters(): p_.requires_grad_(True) 
#     # gradients should just accumalate?
    
#     # checkpointing flag
#     cp = len(cpList)-1
#     print(cp)
    
#     xkm1 = xN
#     qk = qN
#     for ii in range(len(network)-1,-1,-1):
#         # reverse sequence
#         with torch.no_grad():
            
#             # checkpointing
#             if cp > 0 and ii == cpList[cp]:
#                 print('Using Checkpoint:',ii)
#                 xkm1 = Xcp[cp,...]
#                 cp -= 1
                
#             # calculate inverse
#             else:
#                 for jj in range(len(network[ii])-1,-1,-1):
#                     layer = network[ii][jj]
#                     xkm1 = layer.reverse(xkm1,device=device)
                
#             if interFlag:
#                 X[ii,...] = xkm1
                
#         # forward sequece
#         xkm1 = xkm1.detach().requires_grad_(True)
#         xk = xkm1
#         for layer in network[ii]:
#             xk = layer.forward(xk,device=device)
        
#         # backward call
#         xk.backward(qk)
#         with torch.no_grad():
#             qk = xkm1.grad
            
#     return X


# def setup(layer,x0,memlimit,N,gpu_device):
    
#     for p_ in layer.parameters(): p_.requires_grad_(True)

#     # test memory
#     torch.cuda.empty_cache()
#     startmem = torch.cuda.memory_cached(gpu_device)
    
#     x = x0
#     for sub in layer:
#         x = sub(x)
#     x.backward(x0)
    
#     endmem = torch.cuda.memory_cached(gpu_device)
#     mem = endmem-startmem
#     mem = mem / 1024**2
#     print('Memory per iteration:', mem, 'MB')
    
#     # assess how many checkpoints
#     totalmem = mem * N
#     print('Total memory:', totalmem, 'MB')
    
#     if totalmem > memlimit:
#         print('Requires memory-efficient learning!')
#         M = np.ceil(totalmem/memlimit)
#         print('Checkpointing every:',M)
#         cpList = list(range(int(M),int(N-1),int(M)))
# #         cpList = np.linspace(M,N-1,(N-2*M-1) / M).tolist()
#     else:
#         print('Should just use standard backward...')
#         cpList = list(range(0,int(N-1)))
#         M = 1
   
#     return cpList,mem,totalmem,M