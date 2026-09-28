import sys
import argparse
import numpy as np
import mdtraj as md
from diffnets.data_processing import ProcessTraj, WhitenTraj
from diffnets.utils import get_fns

class ImproperlyConfigured(Exception):
    '''The given configuration is incomplete or otherwise not usable.'''
    pass

def process(argv):
    parser = argparse.ArgumentParser(
        prog='process',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Process simulation data for DiffNets training.')

    # Required arguments
    parser.add_argument(
        '--sim-dirs', nargs='+', required=True, 
        help='Path to an np.array containing directory names for each simulation set.')
    parser.add_argument(
        '--pdb-fns', nargs='+', required=True, 
        help='Path to an np.array containing pdb file names for each simulation set. The order should match the order of sim_dirs.')
    parser.add_argument(
        '--out-dir', required=True, 
        help='Path to the output directory where processed data will be saved.')

    # Optional arguments
    parser.add_argument(
        '--atom-sel', default=None,
        help='Path to an np.array containing a list of indices for each variant.The indices need to select equivalent atoms across variants.')
    parser.add_argument(
        '--stride', default=None,
        help='Path to an np.array containing a stride integer for each variant.')

    args = parser.parse_args(argv[1:])

    # Validate required arguments
    try:
        sim_dirs = np.load(args.sim_dirs)
    except:
        raise ImproperlyConfigured(f'Incorrect input for sim_dirs. Use --help flag for information on the correct input for sim_dirs.')

    try:
        pdb_fns = np.load(args.pdb_fns)
    except:
        raise ImproperlyConfigured(f'Incorrect input for pdb_fns. Use --help flag for information on the correct input for pdb_fns.')

    if args.atom_sel:
        try:
            atom_sel = np.load(args.atom_sel)
            #Add a check to make sure atom_sel is not same
            n_atoms = [md.load(fn).atom_slice(atom_sel[i]).n_atoms for i,fn in enumerate(pdb_fns)]
            if len(np.unique(n_atoms)) != 1:
                raise ImproperlyConfigured(
                    f'atom_sel needs to choose equivalent atoms across variants. '
                     'After performing atom_sel, pdbs have different numbers of '
                     'atoms.')
        except:
            raise ImproperlyConfigured(f'Incorrect input for atom_sel. Use --help flag for information on the correct input for atom_sel.')

    else:
        n_resis = []
        for fn in pdb_fns:
            pdb=md.load(fn)
            n_resis.append(pdb.top.n_residues)
        if len(np.unique(n_resis)) != 1:
            raise ImproperlyConfigured(
                f'The PDBs supplied have different numbers of residues. The '
                 'default atom selection does not work in this case. Please '
                 'use the --atom-sel option to choose equivalent atoms across  '
                 'different variant pdbs.')

    if args.stride:
        try:
            stride = np.load(args.stride)
        except:
            raise ImproperlyConfigured(f'Incorrect input for stride. Use --help flag for information on the correct input for stride.')

    if len(sim_dirs) != len(pdb_fns):
        raise ImproperlyConfigured(
            f'pdb_fns and sim_dirs must point to np.arrays that have '
             'the same length.')

    for sim_dir, fn in zip(sim_dirs, pdb_fns):
        traj_fns = get_fns(sim_dir, "*.xtc")
        n_traj = len(traj_fns)
        print("Found %s trajectories in %s" % (n_traj, sim_dir))
        if n_traj == 0:
            raise ImproperlyConfigured(
                "Found no trajectories in %s" % sim_dir
            )
        try:
            traj = md.load(traj_fns[0], top=fn)
        except:
            raise ImproperlyConfigured(
                f"Order of pdb_fns and sim_dirs need to "
                 "correspond to each other."
            )

    out_dir = args.out_dir

    proc_traj = ProcessTraj(sim_dirs, pdb_fns, out_dir, atom_sel=atom_sel, stride=stride)
    proc_traj.run()
    print("Aligned trajectories.")
    whiten_traj = WhitenTraj(out_dir)
    print("Starting trajectory whitening.")
    whiten_traj.run()

    return 0

if __name__ == "__main__":
    sys.exit(process(sys.argv))