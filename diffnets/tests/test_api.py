import os
import shutil
import mdtraj as md
import tempfile
import numpy as np

from diffnets.utils import get_fns
from diffnets.data_processing import ProcessTraj, WhitenTraj

CURR_DIR = os.getcwd()
UP_DIR = CURR_DIR[:-len(CURR_DIR.split('/')[-1])]
SCRIPTS_DIR = UP_DIR + 'scripts'

def test_default_inds():

    try:
        td = tempfile.mkdtemp(dir=CURR_DIR)
        data_dir = os.path.join(CURR_DIR,"data")
        pdb_fn1 = os.path.join(data_dir, "beta-peptide1.pdb")
        pdb = md.load(pdb_fn1)
        inds1 = pdb.top.select("name CA or name C or name CB or name N")
        pdb_fn2 = os.path.join(data_dir, "beta-peptide2.pdb")
        pdb = md.load(pdb_fn2)
        inds2 = pdb.top.select("name CA or name C or name CB or name N")

        var_dir_names = [os.path.join(data_dir,"traj1"),
                    os.path.join(data_dir,"traj2")]
        proc_traj = ProcessTraj(var_dir_names,[pdb_fn1,pdb_fn2],td)
        assert set(proc_traj.atom_sel[0]) == set(inds1)
        assert set(proc_traj.atom_sel[1]) == set(inds2) 

    finally:
        shutil.rmtree(td)

def test_whitening_correctness():

    w = WhitenTraj("./data/whitened/")
    master = md.load("./data/whitened/master.pdb")
    traj1 = md.load("./data/whitened/aligned_xtcs/000000.xtc",top=master)
    traj2 = md.load("./data/whitened/aligned_xtcs/000001.xtc",top=master)
    wm = np.load("./data/whitened/wm.npy")
    w.apply_whitening_xtc_dir(w.xtc_dir,master.top,wm,
                              w.cm,1,"./data/whitened/whitened_xtcs")
    traj_fns = get_fns("./data/whitened/whitened_xtcs/", "*.xtc")
    traj = md.load(traj_fns[0],top=master)
    coords = traj.xyz.reshape((2501, 3 * 39))
    c00_1 = np.matmul(coords.transpose(),coords)
    
    traj = md.load(traj_fns[1],top=master)
    coords = traj.xyz.reshape((2500, 3 * 39))
    c00_2 = np.matmul(coords.transpose(),coords)
    c00 = c00_1 + c00_2
    c00 /= 5001

    assert (np.abs(117 - np.sum(np.diagonal(c00))) < 1)

def test_whitening_correctness_2():
    w = WhitenTraj("./data/whitened")
    # generate dummy data
    X = np.random.rand(100, 30)
    X_s = X - X.mean(axis=0)
    cov = np.cov(X_s.transpose())
    # get whitening matrix
    uwm, wm = w.get_wuw_mats(cov)
    Y = w.apply_whitening(X_s, wm, X_s.mean(axis=0))
    whitened_cov = np.cov(Y.transpose())
    # assert that covariance of whitened data is identity
    assert np.abs(np.sum(whitened_cov) - Y.shape[1]) < .0001
    assert np.abs(np.sum(np.diagonal(whitened_cov)) - Y.shape[1]) < .0001

