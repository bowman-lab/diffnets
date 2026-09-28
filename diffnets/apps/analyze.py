import sys
import os
import argparse
import pickle
import mdtraj as md
import numpy as np
from diffnets.analysis import Analysis

class ImproperlyConfigured(Exception):
    '''The given configuration is incomplete or otherwise not usable.'''
    pass

def analyze(argv):
    parser=argparse.ArgumentParser(
        prog='analyze',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Analyze DiffNets results.')

    # Required arguments
    parser.add_argument(
        '--data-dir', required=True,
        help='Path to directory with processed and whitened data.'
    )
    parser.add_argument(
        '--net-dir', required=True,
        help='Path to directory with output from training.'
    )

    # Optional arguments
    parser.add_argument(
        '--indices', default=None,
        help=f'Path to a np.array that contains indices with respect '
             'to data_dir/master.pdb. These indices will be used '
             'to find features that distinguish variants by looking at '
             'a subset of the protein instead of the whole protein.'
    )
    parser.add_argument(
        '--cluster-number', default=1000,
        help=f'Number of clusters desired for clustering on latent space.'
    )
    parser.add_argument(
        '--n-distances', default=100,
        help=f'Number of distances to plot. Takes the n distances that '
             'are most correlated with the diffnet classification score.'
    )

    args = parser.parse_args(argv[1:])

    net_fn = os.path.join(args.net_dir, 'nn_best_polish.pkl')

    try:
        with open(net_fn, 'rb') as f:
            net = pickle.load(f)

    except:
        raise ImproperlyConfigured(
            f'net_dir supplied either does not exist or does not '
             'contain a trained DiffNet.'
        )

    try:
        pdb = md.load(os.path.join(args.data_dir,"master.pdb"))
        n = pdb.n_atoms

    except:
        raise ImproperlyConfigured(
            f'data_dir supplied either does not exist or does not '
             'contain master.pdb'
        )

    net.cpu()
    a = Analysis(net, args.net_dir, args.data_dir)

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
    if args.indices is None:
        print("inds is none")
        inds = np.arange(n)
    else:
        try:
            inds = np.load(args.indices)
            print(inds.shape)
        except:
            raise ImproperlyConfigured(
                f'Inds needs to be a path to a np.array'
            )

    a.find_feats(inds,"rescorr-%s.pml" % args.n_distances,n_states=args.cluster_number,
                     num2plot=args.n_distances)
    
    #Generate a morph of structures along the DiffNets classification score
    a.morph()

    return 0

if __name__ == "__main__":
    sys.exit(analyze(sys.argv))