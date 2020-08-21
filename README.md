diffnets
==============================
[//]: # (Badges)
[![Travis Build Status](https://travis-ci.com/REPLACE_WITH_OWNER_ACCOUNT/diffnets.svg?branch=master)](https://travis-ci.com/REPLACE_WITH_OWNER_ACCOUNT/diffnets)
[![codecov](https://codecov.io/gh/REPLACE_WITH_OWNER_ACCOUNT/diffnets/branch/master/graph/badge.svg)](https://codecov.io/gh/REPLACE_WITH_OWNER_ACCOUNT/diffnets/branch/master)


Supervised and self-supervised autoencoders to identify the mechanistic basis for biochemical differences between protein variants.

## Reference

If you use 'DiffNets' for published research, please cite us:

M.D. Ward, M.I. Zimmerman, S. Swamidass, G.R. Bowman. DiffNets: Self-supervised deep learning to identify the mechanistic basis for biochemical differnces between protein variants. bioRxiv. DOI: 10.1101/2020.07.01.182725, 2020.

## Installation

First, install enspara, which is documented at https://enspara.readthedocs.io/en/latest/installation.html

Then, install PyTorch 1.1.

## Building the docs / Running the tests

Documentation and testing is in its infancy stages and will be continually updated.

## Brief tutorial

The code is currently organized to be run in 2 separate chunks.

First, to process raw trajectories, fill in data_processing_submit.py to match your project directory structure. This will align all trajectories and whiten the data so it is prepared to be input for the DiffNet. This code should be run on a CPU node.

Next, fill in train_submit.py with your desired training parameters. You can see all training parameters in training_dict.txt. When submitting train_submit.py, use a CUDA compatible GPU node. train_submit.py will also automate some analysis including reconstructiong all trajectories using the autoencoder, calculating an RMSD between autoencoder reconstructed structures and their respective simulation frame, calculating classification labels for all frames, and calculating the latent vector for all frames. For other analysis, please peruse analysis.py. You may find analysis.Analysis.find_feats particularly useful.
"../../DiffNets/README.md" 27L, 1688C

### Copyright

Copyright (c) 2020, Michael D. Ward, Bowman Lab


#### Acknowledgements
 
Project based on the 
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.3.
