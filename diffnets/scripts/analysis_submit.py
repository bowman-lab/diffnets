import numpy as np
import pickle
import mdtraj as md
from diffnets.analysis import Analysis

#Directory with originally whitened data
data_dir = "whitened_data/"

#Directory output from training
outdir = "sae_e2_lr0.000100_lat10_r0_em1"

with open("%s/nn_best_polish.pkl" % outdir, "rb") as f:
    net = pickle.load(f)

pdb = md.load("%smaster.pdb" % data_dir)
n = pdb.n_atoms

net.cpu()
a = Analysis(net,outdir,data_dir)

#this method generates encodings (latent space) for all frames,
#produces reconstructed trajectories, produces final classification
#labels for all frames, and calculates an rmsd between the DiffNets
# reconstruction and the actual trajectories
a.run_core()

#This produces a clustering based on the latent space and then 
# finds distances that are correlated with the DiffNets classification
# score and generates a .pml that can be opened with master.pdb
# to generate a figure showing what the diffnet learned.
#Indices for feature analysis
inds = np.arange(n)
a.find_feats(inds,"rescorr-100.pml",n_states=1000,num2plot=100)

#If you've already generated clustering but want to just find features
#follow the code below
#with open("%s/cluster_2000/clusters.pkl" % outdir, "rb") as f:
#    clusters = pickle.load(f)
#a.find_feats(inds,"rescorr-100-tiny.pml",num2plot=100,clusters=clusters)

#Generate a morph of structures along the DiffNets classification score
a.morph()
#print("analysis done")
