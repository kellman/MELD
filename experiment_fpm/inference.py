"""Inference of the model in model.py on loaded dataset.

TODO (kellman): need to finish this...

"""
import os
import argparse
import time
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from datetime import datetime

# MELD toolbox
from meld import UnrolledNetwork

# import local experiment files
import dataloader
import visualizer
import model

parser = argparse.ArgumentParser('experimental model demo')

# learning arguments
parser.add_argument('--num_batches', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=10)
parser.add_argument('--path', type=str, default='/tmp/')
parser.add_argument('--num_iter', type=int, default=1)
parser.add_argument('--step_size', type=float, default=0.001)
parser.add_argument('--loss', type=str, default='mse')
parser.add_argument('--test_freq', type=int, default=20)
parser.add_argument('--optim', type=str, default='adam')
parser.add_argument('--viz', action='store_true')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--verbose', type=bool, default=False)
parser.add_argument('--tensorboard', type=bool, default=False)
parser.add_argument('--meldFlag', type=bool, default=False)
parser.add_argument('--memlimit', type=int, default=11000)

# network specific arguments
parser.add_argument('--alpha', type=float, default=1e-1)
parser.add_argument('--num_meas', type=int, default=6)
parser.add_argument('--num_unrolls', type=int, default=6)
parser.add_argument('--num_bf', type=int, default=1)
parser.add_argument('--num_df', type=int, default=5)

# memory-efficient learning
parser.add_argument('--T', type=int, default=4)

args = parser.parse_args()

if __name__ == '__main__':
    
    if args.verbose:
        print('Torch version: %s' % str(torch.__version__))
        print('Torch CUDA version: %s' % str(torch.version.cuda))
        os.system('nvcc --version')
        print('Forcing MELD:', args.meldFlag)
        
    # Setup device
    torch.cuda.set_device(args.gpu)
    device = torch.device("cuda:"+str(args.gpu) if torch.cuda.is_available() else "cpu")
        
    # Load dataset
    path = '/home/kellman/Workspace/PYTHON/Design_FPM_pytorch/datasets_train_iccp_results/train_amp_exp_n10000.mat' 
    dataset = dataloader.dataloader(path, args.num_batches, args.batch_size, device)
    metadata = dataset.getMetadata()
    metadata['Np'] = dataset[0][0].shape[2:]
    metadata['num_bf'] = args.num_bf
    metadata['num_df'] = args.num_df
    metadata['num_unrolls'] = args.num_unrolls
    metadata['alpha'] = args.alpha
    metadata['T'] = args.T
    
    # Define network/reconstruction
    network = model.model(metadata, device=device)

    # Setup optimizer
    tvars = network.network.parameters()
    if args.optim == 'adam':
        optimizer = torch.optim.Adam(tvars, lr=args.step_size)
    elif args.optim == 'sgd':
        optimizer = torch.optim.SGD(tvars, lr=args.step_size)
    else:
        assert False, 'Not valid optimizer (sgd, adam)'
    
    # Setup loss function
    if args.loss == "mse":
        loss_func = lambda x1, x2: torch.mean((x1-x2)**2)
    else:
        assert False, 'Not valid loss function (mse, etc)'
        
    # Setup network for learning (args.meldFLAG overrides setup memory estimates)
    input_data, output_data = dataset[0]
    xtest = network.initialize(input_data[:1,...].to(device), device=device)
    trainable_network = UnrolledNetwork(network.network, 
                                        xtest, 
                                        memlimit=args.memlimit, 
                                        loss=loss_func,
                                        device=device)
    print(trainable_network.meldFlag, args.meldFlag)
    trainable_network.meldFlag = args.meldFlag