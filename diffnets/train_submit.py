import numpy as np
import mdtraj as md
from . import nnutils
import os
from .training import Trainer
from .analysis import Analysis

if __name__=='__main__':
    # Directory with processed/whitened data
    data_dir = "/outdir/whitened_data/"

    n_cores = 14
    master_fn = os.path.join(data_dir, "master.pdb")
    master = md.load(master_fn)
    n_atoms = master.top.n_atoms
    n_features = 3 * n_atoms

    n_epochs = 3
    # map from variant index to whether active (1) or inactive (0)
    # for classification labels
    # This should be in the order that variants were given to ProcessTraj
    # in data_processing_submit.py
    act_map = np.array([0, 0, 1, 1], dtype=int) #v, wt, t, s
    n_repeats = 1

    # Can choose to iterate over several "jobs" with different
    # parameters to train the DiffNet
    # See all required parameters in training_dict.txt
    jobs = []
    for rep in range(0, n_repeats):
        #lr = learning rate
        for lr in [0.0001]: #[0.001, 0.01, 0.1]:
            for n_latent in [50]:
                job = {}
                layer_sizes = [n_features]
                layer_sizes.append(int(n_features))
                layer_sizes.append(int(n_features/4))
                layer_sizes.append(n_latent)
                job['layer_sizes'] = layer_sizes
                em_bounds = [[0.0, 0.3], #v
                              [0.0, 0.3], # wt
                              [0.60, 0.90], #t
                              [0.60, 0.90]] #s
                job['do_em'] = True
                job['em_bounds'] = np.array(em_bounds)
                job['em_fn'] = 'em1'
                job['em_batch_size'] = 150
                job['em_n_cores'] = 14
                job['nntype'] = nnutils.split_sae
                job['lr'] = lr
                job['n_latent'] = n_latent
                job['rep'] = rep
                job['n_epochs'] = n_epochs
                job['data_dir'] = data_dir
                job['act_map'] = act_map
                job['batch_size'] = 32
                job['batch_output_freq'] = 500
                job['epoch_output_freq'] = 2
                job['test_batch_size'] = 1000
                job['frac_test'] = 0.1
                job['subsample'] = 10
                jobs.append(job)

    for job in jobs:
        print("starting")
        outdir = "%s_e%d_lr%1.6f_lat%d_r%d" % \
            (job['nntype'].__name__, job['n_epochs'], job['lr'],
             job['n_latent'], job['rep'])
        if job['do_em'] and hasattr(job['nntype'], 'classify'):
            outdir = outdir + "_" + job['em_fn']
        outdir = os.path.abspath(outdir)
        job['outdir'] = outdir

        if os.path.exists(outdir):
            print("skipping %s because it already exists" % outdir)
            continue
        cmd = "mkdir %s" % outdir
        os.system(cmd)

        if hasattr(job['nntype'], 'split_inds'):
            # inds1 - indices used for classification
            # inds2 - indices for rest of proteins
            # Must consider that there is an index for each x, y, and z
            # coordinate. e.g. if you want to select atom 0 and atom 1, 
            # the indices are 0,1,2 and 3,4,5
            # split_inds is one simple approach for specifying a portion
            # of the protein to classify
            job['inds1'], job['inds2'] = nnutils.split_inds(master,182,1)

        trainer = Trainer(job)
        # If your entire dataset can be held in memory, set data_in_mem=True
        # because the DiffNet will train much faster
        net = trainer.run(data_in_mem=False)
        print("network trained")
        net.cpu()
        
        a = Analysis(net,outdir,data_dir)
        a.run_core()
        print("analysis done")

