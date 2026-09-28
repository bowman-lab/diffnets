import argparse
import functools
import os
import sys
from glob import glob
import mdtraj as md
import multiprocessing as mp
import numpy as np
import torch

def _extract_data_from_sim(inputs,pdb_path,inds_fn,whitened_dir,outdir):
    i, traj_fn = inputs
    inds = np.load(inds_fn)
    master = md.load(os.path.join(whitened_dir,"master.pdb"))
    pdb = md.load(pdb_path)
    traj = md.load(traj_fn, top=pdb)
    traj = traj.atom_slice(inds)
    traj = traj.superpose(master, parallel=False)
    data = traj.xyz.reshape((len(traj), 3*master.n_atoms))
    cm = np.load(os.path.join(whitened_dir,"cm.npy"))
    data = data - cm
    torch_traj = torch.from_numpy(data).to(torch.float32)
    torch.save(torch_traj,os.path.join(outdir,"ID-%s.pt" % i))

def convert(argv):
    parser=argparse.ArgumentParser(
        prog='convert',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='This function converts simulations (xtc files) into input data that '
                    'can be directly fed into a DiffNet.',
    )

    parser.add_argument(
        '--traj-dir', required=True,
        help='Path to directory containing trajectories of a single variant',    
    )
    parser.add_argument(
        '--pdb-path', required=True,
        help='Path to the PDB file corresponding to the trajectories',
    )
    parser.add_argument(
        '--inds-fn', required=True,
        help='Path to the numpy file containing atom indices to select',
    )
    parser.add_argument(
        '--whitened-dir', required=True,
        help='Path to the directory containing whitened reference files (master.pdb and cm.npy)',
    )
    parser.add_argument(
        '--outdir', required=True,
        help='Path to the output directory where converted data will be saved',
    )

    args = parser.parse_args(argv[1:])

    traj_fns = glob(os.path.join(args.traj_dir,"*.xtc"))
    inputs = [(i,j) for i,j in enumerate(traj_fns)]
    if not os.path.exists(args.outdir):
        os.mkdir(args.outdir)
    n_cores = int(os.environ.get('SLURM_NPROCS', 1))
    pool = mp.Pool(processes=n_cores)
    f = functools.partial(
        _extract_data_from_sim, 
        inds_fn=args.inds_fn,
        whitened_dir=args.whitened_dir,
        outdir=args.outdir,
        pdb_path=args.pdb_path
    )
    result = pool.map_async(f, inputs)
    result.wait()
    traj_lens = result.get()
    pool.close()

    return 0

if __name__ == "__main__":
    sys.exit(convert(sys.argv))