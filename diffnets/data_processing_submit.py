from .data_processing import ProcessTraj, WhitenTraj
import os

#Base path to trajectory and topology data
orig_data_dir = "/project/example"

# path to simulations for each variant
# Each of these paths should contain all trajectories for
# a given variant
var_dir_names = [os.path.join(orig_data_dir,"Trajectories_%s" % i)
                     for i in ["wt","m182v","m182t","m182s"]]


# path to pdbs of each respective variant
var_pdb_fns = [os.path.join(orig_data_dir,i) for i in
                ["prot_masses_wt.pdb","prot_masses_m182v.pdb",
                "prot_masses_m182t.pdb", "prot_masses_m182s.pdb"]]

# Output directory for processed and whitened data
outdir = "/outdir/whitened_data/"

proc_traj = ProcessTraj(var_dir_names,var_pdb_fns,outdir)
proc_traj.run()
print("Aligned trajectories")
whiten_traj = WhitenTraj(outdir)
print("starting trajectory whitening")
whiten_traj.run()

