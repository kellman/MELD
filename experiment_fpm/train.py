"""Trains the model in model.py on loaded dataset.

python train.py --verbose True --num_iter 100 --batch_size 5 --test_freq 1 
--step_size 0.01 --num_unrolls 75 --alpha 0.1 --tensorboard True --meldFlag=True      

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
import time

# MELD toolbox
from meld.recon import UnrolledNetwork

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

def get_time_stamp():
    return str(datetime.now())[11:19] + '_'

def format_loss_monitor(batch, loss, time):
    return 'batch={0:3d} | loss={1:.5f} | log loss={2:2.3f} | time={3:2.3f}'.format(batch, loss, np.log10(loss), time)

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
    if args.verbose: print(trainable_network.meldFlag, args.meldFlag)
    trainable_network.meldFlag = args.meldFlag
    if args.verbose: print(trainable_network.cpList)
    
    # Setup tensorboard writer
    exp_string = 'batch_size={0:d}_stepsize={1:.3f}_loss_fn={8:}_optim={2:}_num_unrolls={3:d}_alpha={4:.3f}_num_df={5:d}_num_bf={6:d}_num_leds={7:d}_meld={9:}'.format(args.batch_size, args.step_size, args.optim, args.num_unrolls, args.alpha, args.num_df, args.num_bf, metadata['Nleds'], args.loss, trainable_network.meldFlag)
    exp_time = get_time_stamp()
    exp_dir = './runs/' + exp_time + exp_string
    if args.verbose: print(exp_dir)
    if args.tensorboard: 
        writer = SummaryWriter(exp_dir)
        
    # training loop
    for ii in range(args.num_iter):
        batch_index = np.mod(ii,args.num_batches-1)
        input_data, output_data = dataset[batch_index]
        # TODO (kellman) put data on parallel device over batch dimension
        
        
        # forward evaluation (loop over batches)
        loss_training = 0.
        network.network.zero_grad()
        for bb in range(args.batch_size):
            zgFlag = bb == 0
            start_time = time.time()
            x0 = network.initialize(input_data[bb:bb+1,...].to(device), device=device)
            xN_tmp, loss_tmp, _, _ = trainable_network.forward(x0, output_data[bb:bb+1,...].to(device))
            end_time = time.time()
            with torch.no_grad():
                loss_training += loss_tmp
        
        # gradient and projection updates
        optimizer.step()
        network.projection()
        
        # testing evaluation
        if np.mod(ii, args.test_freq) == 0:
            input_data, output_data = dataset[args.num_batches-1]
            # TODO (kellman) put data on parallel device over batch dimension
            
            
            # forward evaluation (loop over batches)
            loss_testing = 0.
            for bb in range(args.batch_size):
                x0 = network.initialize(input_data[bb:bb+1,...].to(device), device=device)
                x, loss_tmp = trainable_network.loss_eval(x0,output_data[bb:bb+1,...].to(device))
                with torch.no_grad():
                    loss_testing += loss_tmp.cpu().numpy()
            
            
            # tensorboard writer
            if args.tensorboard:
                # visualizing
                with torch.no_grad():
                    fig = visualizer.visualize(network.grad.C.data.cpu().numpy(), metadata)
                os.system('mkdir -p ' + exp_dir + '/tmp/')
                img_file_path = exp_dir + '/tmp/leds_{0:4d}.png'.format(ii)
                fig.savefig(img_file_path, transparent=True, dpi=150)
                plt.close()
                led_img = mpimg.imread(img_file_path)[...,:3]
                
                
                # writing to tensorboard
                writer.add_scalar('Loss/test', loss_testing/args.batch_size, ii)
                writer.add_scalar('Loss/train', loss_training/args.batch_size, ii)
                writer.add_image('Visual/leds', led_img, ii, dataformats='HWC')
                
                
                # saving checkpoints
                saveDict = {'model_state_dict':network.network.state_dict(),
                            'led_pattern':network.grad.C.data,
                            'loss_testing':loss_testing,
                            'loss_training':loss_training,
                            'alpha':args.alpha,
                            'num_unrolls':args.num_unrolls,
                            'num_meas':args.num_meas,
                            'num_bf':args.num_bf,
                            'num_df':args.num_df,
                           }             
                torch.save(saveDict, exp_dir + '/ckpt_{0:04d}.tar'.format(ii))
                
        # progress print statement
        print(format_loss_monitor(ii, loss_testing / args.batch_size, end_time - start_time), end="\r")
