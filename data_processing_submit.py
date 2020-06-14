from data_processing import ProcessTraj, WhitenTraj
import os
import logging

#Base path to trajectory and topology data
orig_data_dir = "/project/bowmore/mizimmer/documents/projects/interesting_proteins/beta_lactamase/stabilizing_mutations_data/full_data"

# path to simulations for each variant
var_dir_names = [os.path.join(orig_data_dir,"Trajectories_%s" % i)
                     for i in ["wt","m182v","m182t","m182s"]]

# path to pdbs of each respective variant
var_pdb_fns = [os.path.join(orig_data_dir,i) for i in
                ["prot_masses_wt.pdb","prot_masses_m182v.pdb",
                "prot_masses_m182t.pdb", "prot_masses_m182s.pdb"]]

# Output directory for processed and whitened data
outdir = "/project/bowmore/mdward/projects/diffnets-code-testing/whitened_data/"

proc_traj = ProcessTraj(var_dir_names,var_pdb_fns,outdir)
proc_traj.run()
print("Aligned trajectories")
whiten_traj = WhitenTraj(outdir)
print("starting trajectory whitening")
whiten_traj.run()

