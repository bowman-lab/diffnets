import glob
import mdtraj as md
import multiprocessing as mp
import nnutils
import numpy as np
import os
import sys
import time
import whiten

#data_dir = "/home/mizimmer/documents/projects/interesting_proteins/beta_lactamase/stabilizing_mutations_data/full_data"
data_dir = "/home/shared/myosin_2019/"
#variants = ["wt_14207","em_14209","gm_14208","gem_14210"] #wt,em,gm,gem
variants = ["m182v","wt","m182t","m182s","m182n"]
variants = ["myh7-wt", "myh7-i263t"]
at_sel = "name CA or name CB or name N or name C"
outdir = os.path.join("myosin/wt_ht_2.5nm/", "wt_i263t_cabcn")
nnutils.mkdir(outdir)
n_cores = 24
TESTING = False

nnutils.mkdir(outdir)
#os.system("cp %s %s/" % (sys.argv[0], outdir))

# setup directories
xtc_dir = os.path.join(outdir, "aligned_xtcs")
nnutils.mkdir(xtc_dir)
label_dir = os.path.join(outdir, "labels")
nnutils.mkdir(label_dir)

# setup reference pdb to align all to
#pdb_fn = "/home/sukrit.singh/Projects/blact/fahSims_betaTurn/topologies/wt_14207.pdb"
pdb_fn = os.path.join(data_dir, "%s/fah_input/%s_masses.pdb" % (variants[0],variants[0]))
master = md.load(pdb_fn)
inds = master.top.select(at_sel)
master = master.atom_slice(inds)
master.center_coordinates()
#at_sel1 = np.load("myosin/close_inds.npy")
#master1 = master.atom_slice(at_sel1)
#master1.center_coordinates()
master_fn = os.path.join(outdir, "master.pdb")
master.save(master_fn)
n_feat = 3 * master.top.n_atoms

def preprocess_traj(inputs):
    """Align to master and store traj to outdir/traj_num.xtc with zero padded
    filename"""
    traj_fn, top_fn, traj_num, var_ind = inputs
    v = variants[var_ind]

    if traj_num is 0:
        print("Processing", traj_num, traj_fn, top_fn)
    else:
        print("on traj", traj_num)

    traj = md.load(traj_fn, top=top_fn)

    # just keep backbone plus CB, except CB of Ser238, if present
    if traj_num is 0:
        print("Selecting inds")
    inds = traj.top.select(at_sel)
    
    if traj.top.residue(238-26).name == "SER":
        print("have SER in ", v)
        bad_atom_ind = traj.top.select('resSeq 238 and name CB')[0]
        bad_ind = np.where(inds == bad_atom_ind)[0]
        inds = np.delete(inds, bad_ind)
    traj = traj.atom_slice(inds)
    #traj = traj.atom_slice(at_sel1)
    # align to master
    if traj_num is 0:
        print("Superposing")
    traj = traj.superpose(master, parallel=False)
    traj = traj[100::20]
    #traj = traj.atom_slice(at_sel1)
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

    return n

# for testing
#variants = ["test"]

# build list of trajecotires and other info for preprocessing
print("Getting list of traj to process")
traj_num = 0
inputs = []
for i, v in enumerate(variants):
    #traj_dir_fn = os.path.join(data_dir, "p%s/RUN*/CLONE*/" % v[-5:])
    #traj_dir_fn = os.path.join(data_dir, "Trajectories_%s" % v)
    traj_dir_fn = os.path.join(data_dir, "%s/" % v)
    top_fn = os.path.join(data_dir, "%s/fah_input/%s_masses.pdb" % (v,v))
    #top_fn = "/home/sukrit.singh/Projects/blact/fahSims_betaTurn/topologies/%s.pdb" % v

    # process in parallel, get back traj lengths, and store those
    # load all traj for variant
    #traj_dir_fn = os.path.join(dir_fn, "sims")
    print(traj_dir_fn)
    traj_fns = nnutils.get_fns(traj_dir_fn, "*.xtc")
    
    for traj_fn in traj_fns:
        inputs.append((traj_fn, top_fn, traj_num, i))
        traj_num += 1
print(inputs)

# do preprocessing
n_cores = mp.cpu_count()
print("Preprocessing traj")
pool = mp.Pool(processes=n_cores)
if TESTING:
    inputs = inputs[:2]
result = pool.map_async(preprocess_traj, inputs)
result.wait()
traj_lens = result.get()
traj_lens = np.array(traj_lens, dtype=int)
pool.close()
traj_len_fn = os.path.join(outdir, "traj_lens.npy")
np.save(traj_len_fn, traj_lens)

print("Getting center of mass")
traj_fns = nnutils.get_fns(xtc_dir, "*.xtc")
if TESTING:
    traj_fns = traj_fns[:2]
cm_fns = nnutils.get_fns(xtc_dir, "cm*.npy")
n_traj = len(traj_fns)
print("  Found %d trajectories" % n_traj)
cm = np.zeros(n_feat)
for i, cm_fn in enumerate(cm_fns):
    d = np.load(cm_fn)
    cm += traj_lens[i] * d
cm /= traj_lens.sum()
cm_fn = os.path.join(outdir, "cm.npy")
#close_xyz_inds = np.load("blac_stability/v_wt_t_s_cabcn/close_xyz_inds.npy")
#cm = cm[close_xyz_inds]
np.save(cm_fn, cm)
cm = np.load(os.path.join(outdir, "cm.npy"))
print("Whitening")
whitened_dir = os.path.join(outdir, "whitened_xtcs")
nnutils.mkdir(whitened_dir)
start = time.clock()
c00 = whiten.get_c00_xtc_list(traj_fns, master.top, cm, n_cores)
c00_fn = os.path.join(outdir, "c00.npy")
np.save(c00_fn, c00)
uwm, wm = whiten.get_wuw_mats(c00)
uwm_fn = os.path.join(outdir, "uwm.npy")
np.save(uwm_fn, uwm)
wm_fn = os.path.join(outdir, "wm.npy")
np.save(wm_fn, wm)
whiten.apply_whitening_xtc_dir(xtc_dir, master.top, wm, cm, n_cores, whitened_dir)
end = time.clock()
print("time:", end-start)
