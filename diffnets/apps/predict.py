import argparse
import os
import pickle
import sys
from glob import glob
import numpy as np
import torch

class ImproperlyConfigured(Exception):
    '''The given configuration is incomplete or otherwise not usable.'''
    pass

def _chunks(arr, chunk_size):
    """Yield successive chunk_size chunks from arr."""
    for i in range(0, len(arr), chunk_size):
        yield arr[i:i + chunk_size]

def predict(argv):
    parser=argparse.ArgumentParser(
        prog='predict',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=f'Uses an already trained DiffNet to predict on a variant outside '
                     'the training. Requires the variant to be preprocessed for input. '
                     'See extract_data_from_sim for preprocessing.',
    )

    # Required arguments
    parser.add_argument(
        '--data-dir', required=True,
        help=f'Directory with pytorch float tensors for each example '
              '(i.e. frame of a simulation)',
    )
    parser.add_argument(
        '--nn-path', required=True,
        help='Path to directory with DiffNet training output',
    )
    parser.add_argument(
        '--out-dir', required=True,
        help=f'Directory to output label, latent vectors, and/or reconstructed '
              'trajectories.',
    )

    # Optional arguments
    parser.add_argument(
        '--save-labels', default=True,
        help='Whether to save the predicted labels',
    )
    parser.add_argument(
        '--save-latent', default=False,
        help='Whether to save the latent encodings',
    )
    parser.add_argument(
        '--save-recon', default=False,
        help='Whether to save the reconstructions',
    )

    args = parser.parse_args(argv[1:])

    if not args.save_labels and not args.save_latent and not args.save_recon:
        raise ImproperlyConfigured(
            f'at least one of save_labels, save_latent, or save_recon'
             'must be true.'
        )

    net = pickle.load(open("%s/nn_best_polish.pkl" % args.nn_path, 'rb'))
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    torch_trajs = glob.glob(os.path.join(args.data_dir,"*.pt"))
    traj_num = 0

    if args.save_labels:
        os.makedirs(os.path.join(args.out_dir,"labels"))

    if args.save_latent:
        os.makedirs(os.path.join(args.out_dir,"latent"))

    if args.save_recon:
        os.makedirs(os.path.join(args.out_dir,"recon"))

    for t in torch_trajs:
        ex = torch.load(t)
        encodings = []
        labels = []
        recon = []
        for batch in _chunks(ex,100):
            local_batch = batch.to(device=device, dtype=torch.float32)
            x_pred, latent, class_pred = net(local_batch)
    
            if args.save_labels:
                labels.append(class_pred) 
            if args.save_latent:
                encodings.append(latent)
            if args.save_recon:
                recon.append(x_pred)
    
        if args.save_labels:
            labels = np.concatenate([l.detach().numpy() for l in labels])
            label_dir = os.path.join(args.out_dir, "labels")
            np.save(os.path.join(label_dir,str(traj_num).zfill(6) + ".npy"),
                        labels)
    
        if args.save_latent:
            encodings = np.vstack(encodings)
            encodings_dir = os.path.join(args.out_dir, "latent")
            np.save(os.path.join(encodings_dir,str(traj_num).zfill(6) + ".npy"),
                        encodings)   
    
        if args.save_recon:
            recon = np.vstack(recon)
            recon_dir = os.path.join(args.out_dir, "recon")
            np.save(os.path.join(recon_dir,str(traj_num).zfill(6) + ".npy"),
                       recon)
        traj_num += 1

    return 0

if __name__ == '__main__':
    sys.exit(predict(sys.argv))
