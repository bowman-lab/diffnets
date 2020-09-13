import os
import shutil
import mdtraj as md
import tempfile

from diffnets.data_processing import ProcessTraj

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
