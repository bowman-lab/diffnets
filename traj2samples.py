import numpy as np
import mdtraj as md
import torch
import os
import utils

data_dir = "/project/bowmore/mdward/projects/diffnets-code-testing/whitened_data"
xtc_dir = os.path.join(data_dir,"aligned_xtcs")
master = md.load(os.path.join(data_dir,"master.pdb"))
outdir = os.path.join(data_dir,"data")
if not os.path.exists(outdir):
        os.mkdir(outdir)

data = utils.load_traj_coords_dir(xtc_dir, "*.xtc", master.top)
print("data loaded")
cm_fn = os.path.join(data_dir, "cm.npy")
cm = np.load(cm_fn)
data -= cm
    
i = 0
print(data.shape)
for t in data:
    frame = torch.from_numpy(t).type(torch.FloatTensor)
    torch.save(frame,os.path.join(outdir,"ID-%s.pt" % i))
    i += 1


        
