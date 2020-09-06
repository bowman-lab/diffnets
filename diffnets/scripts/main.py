#Native to python
import pickle
import os
import click
#third-party libraries
import numpy as np
import mdtraj as md
#diffnets libraries
from diffnets.analysis import Analysis
from diffnets.data_processing import ProcessTraj, WhitenTraj
from diffnets.utils import get_fns

class ImproperlyConfigured(Exception):
    '''The given configuration is incomplete or otherwise not usable.'''
    pass

@click.group()
def cli():
    pass

@cli.command(name='process')
@click.argument('sim_dirs')
@click.argument('pdb_fns')
@click.argument('atom_sel')
@click.argument('outdir')
@click.option('--stride', default=1, help='Factor to subsample by.')
def preprocess_data(sim_dirs,pdb_fns,atom_sel,outdir,stride=1):
    """ sim_dirs: Path to an np.array containing directory names. The 
               array needs one directory name for each variant where each
               directory contains all trajectories for that variant.

        pdb_fns: Path to an np.array containing pdb filenames. The 
               array needs one pdb filename for each variant. The order of 
               variants should match the order of sim_dirs.

        atom_sel: Path to an np.array containing a list of indices for 
              each variant, which operates on the pdbs supplied. The indices
              need to select equivalent atoms across variants.

        outdir: Path you would like processed data to live.
 """
    try:
        var_dir_names = np.load(sim_dirs)
    except:
        click.echo(f'Incorrect input for sim_dirs. Use --help flag for '
               'information on the correct input for sim_dirs.')
        raise

    try:
        var_pdb_fns = np.load(pdb_fns)
    except:
        click.echo(f'Incorrect input for pdb_fns. Use --help flag for '
               'information on the correct input for pdb_fns.')
        raise

    try:
        atom_sel = np.load(atom_sel)
    except:
        click.echo(f'Incorrect input for atom_sel. Use --help flag for '
               'information on the correct input for atom_sel.')
        raise

    if len(var_dir_names) != len(var_pdb_fns):
        raise ImproperlyConfigured(
            f'pdb_fns and sim_dirs must point to np.arrays that have '
             'the same length')

    for vd,fn in zip(var_dir_names, var_pdb_fns):
        traj_fns = get_fns(vd, "*.xtc")
        n_traj = len(traj_fns)
        click.echo("Found %s trajectories in %s" % (n_traj,vd))
        if n_traj == 0:
            raise ImproperlyConfigured(
                "Found no trajectories in %s" % vd)
        try: 
            traj = md.load(traj_fns[0],top=fn)
        except:
            click.echo(f'Order of pdb_fns and sim_dirs need to '
                'correspond to each other.')
            raise

    n_atoms = [md.load(fn).atom_slice(atom_sel[i]).n_atoms for i,fn in enumerate(var_pdb_fns)]
    if len(np.unique(n_atoms)) != 1:
        raise ImproperlyConfigured(
                f'atom_sel needs to choose equivalent atoms across variants. '
                 'After performing atom_sel, pdbs have different numbers of '
                 'atoms.')

    proc_traj = ProcessTraj(var_dir_names,var_pdb_fns,outdir,stride=stride,
                            atom_sel=atom_sel)
    proc_traj.run()
    print("Aligned trajectories")
    whiten_traj = WhitenTraj(outdir)
    print("starting trajectory whitening")
    whiten_traj.run()
    
#Need a yml file with training parameters

#Need path to processed/whitened data dir

#Need path for output directory
@cli.command(name='train')
def train():
    pass

#Need path to processed/whitened data directory

#Need path to DiffNets training output directory

#Optional parameter to do "find_feats" on a subset of the protein

@cli.command(name='analyze')
@click.argument('data_dir')
@click.argument('net_dir')
@click.option('--inds',
              help=f'Path to a np.array that contains indices with respect '
                    'to data_dir/master.pdb. These indices will be used'
                    'to find features that distinguish variants by looking at '
                    'a subset of the protein instead of the whole protein')
@click.option('--cluster_number', default=1000,
              help=f'Number of clusters desired for clustering on latent space')
@click.option('--n_distances', default=100,
              help=f'Number of distances to plot. Takes the n distances that '
                    'are most correlated with the diffnet classification score.')
def analyze(data_dir,net_dir,inds=None,cluster_number=1000,n_distances=100):
    """ data_dir: Path to directory with processed and whitened data.

        net_dir: path to directory with output from training.
    """
    net_fn = os.path.join(net_dir,"nn_best_polish.pkl")

    try: 
        with open(net_fn, "rb") as f:
            net = pickle.load(f)
    except:
        click.echo(f'net_dir supplied either does not exist or does not '
                    'contain a trained DiffNet.')
        raise

    try:
        pdb = md.load(os.path.join(data_dir,"master.pdb"))
        n = pdb.n_atoms
    except:
        click.echo(f'data_dir supplied should contain the processed/whitened '
                    'data including master.pdb')
        raise

    net.cpu()
    a = Analysis(net,net_dir,data_dir)

    #this method generates encodings (latent space) for all frames,
    #produces reconstructed trajectories, produces final classification
    #labels for all frames, and calculates an rmsd between the DiffNets
    #reconstruction and the actual trajectories
    a.run_core()

    #This produces a clustering based on the latent space and then
    # finds distances that are correlated with the DiffNets classification
    # score and generates a .pml that can be opened with master.pdb
    # to generate a figure showing what the diffnet learned.
    #Indices for feature analysis
    if inds is None:
        inds = np.arange(n)
    a.find_feats(inds,"rescorr-100.pml",n_states=cluster_number,
                 num2plot=n_distances)

    #Generate a morph of structures along the DiffNets classification score
    a.morph()
    #print("analysis done")

if __name__=="__main__":
    cli()
