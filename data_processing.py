import os
import multiprocessing as mp
import functools

import numpy as np
import mdtraj as md

#Move to its own file "nnutils.py"
class Navigator:
    """Stores all information about directory structure"""

    def __init__(self, orig_data_dir,
                       whit_data_dir,
                       net_dir,
                       var_dir_names,
                       var_pdb_fns)

        self.orig_data_dir = orig_data_dir
        self.whit_data_dir = whit_data_dir
        self.net_dir = net_dir
        self.var_dir_names = var_dir_names
        self.var_pdb_fns = var_pdb_fns

    def make_dir(dir_name):
        if not os.path.exists(dir_name)
            os.mkdir(dir_name)
    def get_fns()
        pass

class ProcessTraj(Navigator):
    """Process raw trajectory data to create organized directories
       with centered trajectories for a selection of atoms that will go into
       DiffNet"""

    def __init__(self,
                 atom_sel="name CA or name CB or name N or name C",
                 #gly_mut_ind = []):

        self.atom_sel = atom_sel
        #self.glycine_mut_inds = 
        self.master = self.make_master_pdb()
        self.n_feats = 3*self.master.top.n_atoms
    

    def make_master_pdb(self):
        pdb_fn = self.var_pdb_fns[0]
        master = md.load(os.path.join(self.orig_data_dir, "%s" % pdb_fn)
        inds = master.top.select(self.atom_sel)
        master = master.atom_slice(inds)
        master.center_coordinates()
        master_fn = os.path.join(self.whit_data_dir, "master.pdb")
        master.save(master_fn)
        return master

    def make_traj_list(self):
        traj_num = 0
        inputs = []
        i = 0
        for vd, fn in zip(var_dir_names,var_pdb_fns):
            traj_dir_fn = os.path.join(self.orig_data_dir, "%s/" % vd)
            top_fn = os.path.join(self.orig_data_dir, "%s" % fn)
            
            traj_fns = nnutils.get_fns(traj_dir_fn, "*.xtc")
            for traj_fn in traj_fns:
                inputs.append((traj_fn, top_fn, traj_num, i))
                traj_num += 1
            i += 1
        return inputs

    def _preprocess_traj(self,inputs,xtc_dir, label_dir):
        """Align to master and store traj to outdir/traj_num.xtc with zero padded
         filename"""
        traj_fn, top_fn, traj_num, var_ind = inputs
        v = self.var_dir_names[var_ind]

        if traj_num is 0:
            print("Processing", traj_num, traj_fn, top_fn)
        else:
            print("on traj", traj_num)

        traj = md.load(traj_fn, top=top_fn)

        # just keep backbone plus CB, except CB of Ser238, if present
        if traj_num is 0:
            print("Selecting inds")
        inds = traj.top.select(self.atom_sel)

        #Check for glycine mutations
        #if traj.top.residue(238-26).name == "SER":
             #print("have SER in ", v)
             #bad_atom_ind = traj.top.select('resSeq 238 and name CB')[0]
             #bad_ind = np.where(inds == bad_atom_ind)[0]
             #inds = np.delete(inds, bad_ind)
        traj = traj.atom_slice(inds)
    
        # align to master
        if traj_num is 0:
            print("Superposing")
        traj = traj.superpose(self.master, parallel=False)
        
        # save traj and its center of mass
        if traj_num is 0:
            print("Saving xtc")
        
        new_traj_fn = os.path.join(xtc_dir, str(traj_num).zfill(6) + ".xtc")
        traj.save(new_traj_fn)
        if traj_num is 0:
            print("Getting/saving CM")
        n = len(traj)
        cm = traj.xyz.reshape((n, 3*traj.top.n_atoms)).mean(axis=0)
        new_cm_fn = os.path.join(xtc_dir, "cm" + str(traj_num).zfill(6) + ".npy")
        np.save(new_cm_fn, cm)
        
        labels = var_ind * np.ones(n)
        label_fn = os.path.join(label_dir, str(traj_num).zfill(6) + ".npy")
        np.save(label_fn, labels)

    def preprocess_traj(inputs,xtc_dir,label_dir)
        n_cores = mp.cpu_count()
        pool = mp.Pool(processes=n_cores)
        f = functools.partial(_preprocess_traj,xtc_dir=xtc_dir,label_dir=label_dir)
        result = pool.map_async(f, inputs)
        result.wait()
        traj_lens = result.get()
        traj_lens = np.array(traj_lens, dtype=int)
        pool.close()

        traj_len_fn = os.path.join(self.whit_data_dir, "traj_lens.npy")
        np.save(traj_len_fn, traj_lens)
        traj_fns = nnutils.get_fns(xtc_dir, "*.xtc")
        cm_fns = nnutils.get_fns(xtc_dir, "cm*.npy")
        n_traj = len(traj_fns)
        print("  Found %d trajectories" % n_traj)
        cm = np.zeros(n_feat)
        for i, cm_fn in enumerate(cm_fns):
            d = np.load(cm_fn)
            cm += traj_lens[i] * d
        cm /= traj_lens.sum()
        cm_fn = os.path.join(self.whit_data_dir, "cm.npy")
        np.save(cm_fn, cm)

    def run(self):
        inputs = self.make_traj_list()
        xtc_dir = os.path.join(self.whit_data_dir,"aligned_xtcs")
        self.make_dir(xtc_dir)
        label_dir = os.path.join(self.whit_data_dir,"labels")
        self.make_dir(label_dir)
        self.preprocess_traj(inputs,xtc_dir,label_dir)
        
class WhitenTraj(Navigator): 
    
    def __init__(self)
