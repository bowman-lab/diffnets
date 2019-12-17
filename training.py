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
        data_dir = job['data_dir']
        n_latent = job['n_latent']
        layer_sizes = job['layer_sizes']
        n_cores = job['n_cores']
        nntype = job['nntype']
        frac_test = job['frac_test']

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
        indicator_dir = os.path.join(data_dir, "indicators")
        indicators = nnutils.load_npy_dir(indicator_dir, "*.npy")
        indicators = np.array(indicators, dtype=int)
        print("    size loaded data", data.shape, indicators.shape)

        if job["canonical"]:
            job["do_em"] = False
            targ_dir = os.path.join(data_dir, "custom_labels")
            print(targ_dir)
            targets = nnutils.load_npy_dir(targ_dir, "*.npy")
            targets = targets.flatten()
        else:
            targets = np.zeros((len(indicators), 1))
            targets[:, 0] = act_map[indicators]

        wm_fn = os.path.join(data_dir, "wm.npy")
        uwm_fn = os.path.join(data_dir, "uwm.npy")
        cm_fn = os.path.join(data_dir, "cm.npy")
        wm = np.load(wm_fn)
        if job['nntype'] = "split_ae":
            #or call a procedure here to find close inds
            close_xyz_inds_fn = os.path.join(data_dir, "close_xyz_inds.npy")
            inds1 = np.load(close_xyz_inds_fn)
            non_close_xyz_inds_fn = os.path.join(data_dir, "non_close_xyz_inds.npy")
            inds2 = np.load(non_close_xyz_inds_fn)
            #broadcast it to matrix of only atoms within 10A of mutation
            wm1 = wm[inds1[:,None],inds1]
            wm2 = wm[inds2[:,None],inds2]
        uwm = np.load(uwm_fn)
        uwm = uwm
        cm = np.load(cm_fn).flatten()
        # remove cm ahead of time
        data -= cm







