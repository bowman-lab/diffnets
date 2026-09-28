import sys
import shutil
import os
import argparse
import yaml
import numpy as np
import mdtraj as md
from diffnets import nnutils
from diffnets.training import Trainer
from diffnets.utils import get_fns

class ImproperlyConfigured(Exception):
    '''The given configuration is incomplete or otherwise not usable.'''
    pass

nn_d = {
        'nnutils.split_sae': nnutils.split_sae,
        'nnutils.sae': nnutils.sae,
        'nnutils.ae': nnutils.ae,
        'nnutils.split_ae': nnutils.split_ae
}

def train(argv):
    parser = argparse.ArgumentParser(
        prog='train',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Train DiffNets model.')

    parser.add_argument(
        '--config', required=True, 
        help='Path to a yaml configuration file for training DiffNets.')

    args = parser.parse_args(argv[1:])

    config = args.config

    with open(config, 'r') as file:
        job = yaml.safe_load(file)

    required_keys = ['data_dir','n_epochs','act_map','lr','n_latent',
                     'hidden_layer_sizes','em_bounds','do_em','em_batch_size',
                     'nntype','batch_size','batch_output_freq',
                     'epoch_output_freq','test_batch_size','frac_test',
                     'subsample','outdir','data_in_mem']
    optional_keys = ["close_inds_fn","label_spreading"]

    if hasattr(job['nntype'], 'split_inds'):
        required_keys.append('close_inds_fn')

    if 'label_spreading' in job.keys():
        if job['label_spreading'] not in ['gaussian','uniform','bimodal']:
            raise ImproperlyConfigured(
                f"label_spreading must be one of 'gaussian', 'uniform', or 'bimodal'. "
            )
            
    for key in job.keys():
            try:
                required_keys.remove(key)
            except:
                if key in optional_keys:
                    continue
                else:
                    raise ImproperlyConfigured(
                    f'{key} is not a valid parameter. Check yaml file.'
                    )

    if len(required_keys) != 0:
            raise ImproperlyConfigured(
                    f'Missing the following parameters in {config} '
                     '{required_keys} ')

    data_dir  = job['data_dir']
    data_fns = get_fns(data_dir,"*.npy")
    wm_fn = os.path.join(data_dir,"wm.npy")
    if wm_fn not in data_fns:
        raise ImproperlyConfigured(
            f'Cannot find wm.npy in preprocessed data directory. Likely '
             'need to re-run data preprocessing step.'
        )           

    xtc_fns = os.path.join(data_dir,"aligned_xtcs")
    data_fns = get_fns(xtc_fns,"*.xtc")
    ind_fns = os.path.join(data_dir,"indicators")
    inds = get_fns(ind_fns,"*.npy")
    if (len(inds) != len(data_fns)) or len(inds)==0:
        raise ImproperlyConfigured(
            f'Number of files in aligned_xtcs and indicators should be '
                      'equal. Likely need to re-run data preprocessing step.'
        )
    last_indi = np.load(inds[-1])

    n_cores=int(os.environ.get('SLURM_NPROCS', 1))
    master_fn = os.path.join(job['data_dir'], 'master.pdb')
    master = md.load(master_fn)
    n_atoms = master.top.n_atoms
    n_features = n_atoms * 3
    job['layer_sizes'] = [n_features, n_features]

    if len(job['hidden_layer_sizes']) == 0:
        job['layer_sizes'].append(int(n_features/4))
    else:
         for layer in job['hidden_layer_sizes']:
             job['layer_sizes'].append(layer)

    job['layer_sizes'].append(job['n_latent'])
    job['act_map'] = np.array(job['act_map'],dtype=float)
    job['em_bounds'] = np.array(job['em_bounds'])
    job['em_n_cores'] = n_cores
    job['nntype'] = nn_d[job['nntype']]

    if len(job['act_map']) != last_indi[0]+1:
        raise ImproperlyConfigured(
            f"act_map needs to contain a value for each variant."
        )

    if n_features != job['layer_sizes'][0]:
            raise ImproperlyConfigured(
                    f'1st layer size does not match the number of xyz coordinates'
            )  
    
    if job['layer_sizes'][0]!=job['layer_sizes'][1]:
        raise ImproperlyConfigured(
                f'1st and 2nd layer size need to be equal.'
        )
      
    if job['layer_sizes'][-1]!=job['n_latent']:
        raise ImproperlyConfigured(
                f'Last layer size needs to equal number of latent variables'
        )
    
    if 'close_inds_fn' in job.keys():
        if hasattr(job['nntype'], 'split_inds'):
            inds = np.load(job['close_inds_fn'])
            close_xyz_inds = []
            for i in inds:
                close_xyz_inds.append(i*3)
                close_xyz_inds.append((i*3)+1)
                close_xyz_inds.append((i*3)+2)
            all_inds = np.arange((master.n_atoms*3))
            non_close_xyz_inds = np.setdiff1d(all_inds,close_xyz_inds)
            job['inds1'] = np.array(close_xyz_inds)
            job['inds2'] = non_close_xyz_inds
        else:
            raise ImproperlyConfigured(
                f'Indices chosen for a split autoencoder architecture '
                 '(close_inds_fn), but  a split autoencoder architecture '
                 'was not chosen (nntype)'
            )
    
    if not os.path.exists(job['outdir']):
        os.makedirs(job['outdir'])
        shutil.copyfile(config,os.path.join(job['outdir'],config))

    
    trainer = Trainer(job)
    net = trainer.run(data_in_mem=job['data_in_mem'])

    return 0

if __name__ == '__main__':
    sys.exit(train(sys.argv))
