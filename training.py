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
        job = self.job 
        data_dir = self.myNav.whit_data_dir
        outdir = self.myNav.net_dir
        n_latent = job['n_latent']
        layer_sizes = job['layer_sizes']
        n_cores = job['n_cores']
        nntype = job['nntype']
        frac_test = job['frac_test']
        act_map = job['act_map']

        print("  loading data")
        master_fn = os.path.join(data_dir, "master.pdb")
        master = md.load(master_fn)
        out_fn = os.path.join(outdir, "master.pdb")
        master.save(out_fn)
        n_atoms = master.top.n_atoms
        n_features = 3 * n_atoms

        xtc_dir = os.path.join(data_dir, "aligned_xtcs")        

        data = self.myNav.load_traj_coords_dir(xtc_dir, "*.xtc", master.top)
        n_snapshots = len(data)
        print("    size loaded data", data.shape)

        targets = self.get_targets(act_map)
        #Here's how I'll define this method
        #indicator_dir = os.path.join(data_dir, "indicators")
        #indicators = nnutils.load_npy_dir(indicator_dir, "*.npy")
        #indicators = np.array(indicators, dtype=int)
        #targets = np.zeros((len(indicators), 1))
        #targets[:, 0] = act_map[indicators]

        wm_fn = os.path.join(data_dir, "wm.npy")
        uwm_fn = os.path.join(data_dir, "uwm.npy")
        cm_fn = os.path.join(data_dir, "cm.npy")
        wm = np.load(wm_fn)
        uwm = np.load(uwm_fn)
        cm = np.load(cm_fn).flatten()
        inds = self.get_inds()

        data -= cm

        train_inds, test_inds = self.split_test_train()
        n_train = train_inds.shape[0]
        n_test = test_inds.shape[0]
        out_fn = os.path.join(outdir, "train_inds.npy")
        np.save(out_fn, train_inds)
        out_fn = os.path.join(outdir, "test_inds.npy")
        np.save(out_fn, test_inds)
        print("    n train/test", n_train, n_test)

        old_net = nntype(layer_sizes[0:2],inds,wm,uwm)
        old_net.freeze_weights()

        for cur_layer in range(2,len(layer_sizes)):
            net = nntype(layer_sizes[0:cur_layer+1],inds)
            net.freeze_weights(old_net)
            net.cuda()
            net, targets = train(data, targets, labels, train_inds, test_inds, net, str(cur_layer), job)
            old_net = net

        #Polishing
        net.unfreeze_weights()
        net.cuda()
        net, targets = train(data, targets, labels, train_inds, test_inds, net, "polish", job, lr_fact=0.1)






