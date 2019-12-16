import os
import pickle
import sys
import multiprocessing as mp

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

class Trainer:

    def __init__(self,job,myNav):
        """Class to train your DiffNet
        
        Parameters:
        -----------
        job : dictionary with all training parameters
        myNav : Navigator object with directory structure details 
        """
        self.job = job
        self.myNav = myNav
    
    def train(self):
        job = self.job
        do_em = job['do_em']
        n_epochs = job['n_epochs']
        lr = job['lr'] * lr_fact
        subsample = job['subsample']
        batch_size = job['batch_size']
        batch_output_freq = job['batch_output_freq']
        epoch_output_freq = job['epoch_output_freq']
        test_batch_size = job['test_batch_size']
        em_bounds = job['em_bounds']
        nntype = job['nntype']
        em_batch_size = job['em_batch_size']
        em_n_cores = job['em_n_cores']
 
        
        #close_xyz_inds_fn = os.path.join(data_dir, "close_xyz_inds.npy")
        #non_close_xyz_inds_fn = os.path.join(data_dir, "non_close_xyz_inds.npy")
        #close_xyz_inds = np.load(close_xyz_inds_fn)
        #non_close_xyz_inds = np.load(non_close_xyz_inds_fn)        

    def run(self):
        

