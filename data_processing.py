import os
import multiprocessing as mp
import functools
import glob

import numpy as np
import mdtraj as md
from scipy.linalg import inv, sqrtm

def make_dir(dir_name):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)

def get_fns(dir_name,pattern):
    return np.sort(glob.glob(os.path.join(dir_name, pattern)))

def load_traj_coords_dir(dir_name,pattern,top):
    fns = get_fns(dir_name, pattern)
    all_d = []
    for fn in fns:
        t = md.load(fn, top=top)
        d = t.xyz.reshape((len(t), 3*top.n_atoms))
        all_d.append(d)
    all_d = np.vstack(all_d)
    return all_d

def load_npy_dir(dir_name,pattern):
    fns = get_fns(dir_name, pattern)
    all_d = []
    for fn in fns:
        d = np.load(fn)
        all_d.append(d)
    if len(d.shape) == 1:
        all_d = np.hstack(all_d)
    else:
        all_d = np.vstack(all_d)
    return all_d

#Move to its own file "nnutils.py"
class Navigator:
    """Stores all information about directory structure"""

    def __init__(self, og_traj_dir_paths,
                       og_pdb_fn_paths,
                       whit_data_dir,
                       net_dir):
        #Data dir where trajectories are - trajectories should be
        # grouped into a specific dir for each variant
        self.og_traj_dir_paths = og_traj_dir_paths
        #path to PDB file of each variants
        self.og_pdb_fn_paths = og_pdb_fn_paths
        #new data dir for all whitened and processed data
        self.whit_data_dir = whit_data_dir
        #output of the neural net training goes in net_dir
        self.net_dir = net_dir
        #Aligned xtc output will go here
        self.xtc_dir = os.path.join(self.whit_data_dir, "aligned_xtcs")
        #indicators to indicate what variant a traj came from will go here
        self.indicator_dir = os.path.join(self.whit_data_dir, "indicators")


class ProcessTraj:
    """Process raw trajectory data to create organized directories
       with centered trajectories for a selection of atoms that will go into
       DiffNet"""

    def __init__(self,
                 myNav,
                 atom_sel="name CA or name CB or name N or name C",
                 stride=1):
                 #gly_mut_ind = []):
        self.myNav = myNav
        #Should also give option to give list of inds for each variant
        self.atom_sel = atom_sel
        self.master = self.make_master_pdb()
        self.n_feats = 3*self.master.top.n_atoms
        self.stride = stride

    def make_master_pdb(self):
        ## TODO: Add in a check that all pdbs have same number of atoms
        pdb_fn = self.myNav.og_pdb_fn_paths[0]
        master = md.load(pdb_fn)
        inds = master.top.select(self.atom_sel)
        master = master.atom_slice(inds)
        master.center_coordinates()
        master_fn = os.path.join(self.myNav.whit_data_dir, "master.pdb")
        master.save(master_fn)
        return master

    def make_traj_list(self):
        traj_num = 0
        inputs = []
        i = 0
        var_dirs = self.myNav.og_traj_dir_paths
        pdb_fns = self.myNav.og_pdb_fn_paths
        for vd, fn in zip(var_dirs,pdb_fns):
            traj_fns = get_fns(vd, "*.xtc")
            for traj_fn in traj_fns:
            #i indicates which variant the traj came from -- used for training
                inputs.append((traj_fn, fn, traj_num, i))
                traj_num += 1
            i += 1
        return inputs

    def _preprocess_traj(self,inputs):
        """Align to master and store traj to outdir/traj_num.xtc with zero 
            padded filename"""
        traj_fn, top_fn, traj_num, var_ind = inputs

        if traj_num is 0:
            print("Processing", traj_num, traj_fn, top_fn)
        else:
            print("on traj", traj_num)

        traj = md.load(traj_fn, top=top_fn, stride=self.stride)

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
        
        new_traj_fn = os.path.join(self.myNav.xtc_dir, str(traj_num).zfill(6) + ".xtc")
        traj.save(new_traj_fn)
        if traj_num is 0:
            print("Getting/saving CM")
        n = len(traj)
        cm = traj.xyz.reshape((n, 3*traj.top.n_atoms)).mean(axis=0)
        new_cm_fn = os.path.join(self.myNav.xtc_dir, "cm" + str(traj_num).zfill(6) + ".npy")
        np.save(new_cm_fn, cm)
        
        indicators = var_ind * np.ones(n)
        indicators_fn = os.path.join(self.myNav.indicator_dir, str(traj_num).zfill(6) + ".npy")
        np.save(indicators_fn, indicators)
        return n

    def preprocess_traj(self,inputs):
        n_cores = mp.cpu_count()
        pool = mp.Pool(processes=n_cores)
        f = functools.partial(self._preprocess_traj)
        result = pool.map_async(f, inputs)
        result.wait()
        traj_lens = result.get()
        traj_lens = np.array(traj_lens, dtype=int)
        pool.close()

        traj_len_fn = os.path.join(self.myNav.whit_data_dir, "traj_lens.npy")
        np.save(traj_len_fn, traj_lens)
        traj_fns = get_fns(self.myNav.xtc_dir, "*.xtc")
        cm_fns = get_fns(self.myNav.xtc_dir, "cm*.npy")
        n_traj = len(traj_fns)
        print("  Found %d trajectories" % n_traj)
        cm = np.zeros(self.n_feats)
        for i, cm_fn in enumerate(cm_fns):
            d = np.load(cm_fn)
            cm += traj_lens[i] * d
        cm /= traj_lens.sum()
        cm_fn = os.path.join(self.myNav.whit_data_dir, "cm.npy")
        np.save(cm_fn, cm)

    def run(self):
        inputs = self.make_traj_list()
        xtc_dir = os.path.join(self.myNav.whit_data_dir,"aligned_xtcs")
        make_dir(xtc_dir)
        indicator_dir = os.path.join(self.myNav.whit_data_dir,"indicators")
        make_dir(indicator_dir)
        self.preprocess_traj(inputs)
        
class WhitenTraj: 
    
    def __init__(self,myNav):
        self.myNav= myNav
        self.cm = np.load(os.path.join(self.myNav.whit_data_dir,"cm.npy"))

    def get_c00(self, coords, cm, traj_num):
        coords -= cm
        #n_coords = coords.shape[0]
        #norm_const = 1.0 / n_coords
        #c00 = np.einsum('bi,bo->io', coords, coords)
        #matmul is faster
        c00 = np.matmul(coords.transpose(),coords)
        np.save(os.path.join(self.myNav.xtc_dir, "cov"+traj_num+".npy"),c00)

    def _get_c00_xtc(self, xtc_fn, top, cm):
        traj = md.load(xtc_fn, top=top)
        traj_num = xtc_fn.split("/")[-1].split(".")[0]
        n = len(traj)
        n_atoms = traj.top.n_atoms
        coords = traj.xyz.reshape((n, 3 * n_atoms))
        self.get_c00(coords,cm,traj_num)
        return n

    def get_c00_xtc_list(self, xtc_fns, top, cm, n_cores):
        pool = mp.Pool(processes=n_cores)
        f = functools.partial(self._get_c00_xtc, top=top, cm=cm)
        result = pool.map_async(f, xtc_fns)
        result.wait()
        r = result.get()
        pool.close()        

        c00_fns = np.sort(glob.glob(os.path.join(self.myNav.xtc_dir, "cov*.npy")))
        c00 = sum(np.load(c00_fn) for c00_fn in c00_fns)
        c00 /= sum(r)
        return c00

    def get_wuw_mats(self, c00):
        uwm = sqrtm(c00).real
        wm = inv(uwm).real
        return uwm, wm
    
    def apply_unwhitening(self, whitened, uwm, cm):
        # multiply each row in whitened by c00_sqrt
        coords = np.einsum('ij,aj->ai', uwm, whitened)
        coords += cm
        return coords

    def apply_whitening(self, coords, wm, cm):
        # multiply each row in coords by inv_c00
        whitened = np.einsum('ij,aj->ai', wm, coords)
        return whitened

    def _apply_whitening_xtc_fn(self, xtc_fn, top, outdir, wm, cm):
        print("whiten", xtc_fn)
        traj = md.load(xtc_fn, top=top)

        n = len(traj)
        n_atoms = traj.top.n_atoms
        coords = traj.xyz.reshape((n, 3 * n_atoms))
        coords -= cm
        whitened = self.apply_whitening(coords, wm, cm)
        dir, fn = os.path.split(xtc_fn)
        new_fn = os.path.join(outdir, fn)
        traj = md.Trajectory(whitened.reshape((n, n_atoms, 3)), top)
        traj.save(new_fn)

    def apply_whitening_xtc_dir(self,xtc_dir, top, wm, cm, n_cores, outdir):
        xtc_fns = np.sort(glob.glob(os.path.join(xtc_dir, "*.xtc")))

        pool = mp.Pool(processes=n_cores)
        f = functools.partial(self._apply_whitening_xtc_fn, top=top, outdir=outdir, wm=wm, cm=cm)
        pool.map(f, xtc_fns)
        pool.close()

    def run(self):
        outdir = self.myNav.whit_data_dir
        whitened_dir = os.path.join(outdir,"whitened_xtcs")
        make_dir(whitened_dir)
        n_cores = mp.cpu_count()
        traj_fns = get_fns(self.myNav.xtc_dir, "*.xtc")
        master = md.load(os.path.join(outdir,"master.pdb"))
        #May run into memory issues
        c00 = self.get_c00_xtc_list(traj_fns, master.top, self.cm, n_cores)
        c00_fn = os.path.join(outdir,"c00.npy")
        np.save(c00_fn, c00)
        uwm, wm = self.get_wuw_mats(c00)
        uwm_fn = os.path.join(outdir, "uwm.npy")
        np.save(uwm_fn, uwm)
        wm_fn = os.path.join(outdir, "wm.npy")
        np.save(wm_fn, wm)
        self.apply_whitening_xtc_dir(self.myNav.xtc_dir, master.top, wm, self.cm, n_cores, whitened_dir)



