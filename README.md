diffnets
==============================
[//]: # (Badges)
[![Travis Build Status](https://travis-ci.com/REPLACE_WITH_OWNER_ACCOUNT/diffnets.svg?branch=master)](https://travis-ci.com/REPLACE_WITH_OWNER_ACCOUNT/diffnets)
[![codecov](https://codecov.io/gh/REPLACE_WITH_OWNER_ACCOUNT/diffnets/branch/master/graph/badge.svg)](https://codecov.io/gh/REPLACE_WITH_OWNER_ACCOUNT/diffnets/branch/master)


Supervised and self-supervised autoencoders to identify the mechanistic basis for biochemical differences between protein variants.

## Reference

If you use 'DiffNets' for published research, please cite us:

M.D. Ward, M.I. Zimmerman, S. Swamidass, G.R. Bowman. DiffNets: Self-supervised deep learning to identify the mechanistic basis for biochemical differnces between protein variants. bioRxiv. DOI: 10.1101/2020.07.01.182725, 2020.

## Dependencies

-python 3.6

-scipy, sklearn

-enspara -> which requires (MDTraj=1.8,numpy=1.14,cython, mpi4py)

-pytorch

## Recommended Installation

*Go to directory you would like diffnets (and enspara) to live in*

cd /desired/path/for/packages

*Create a conda environment that will be used for diffnets*

conda create --name diffnets python=3.6

*Enter this conda environment and install enspara*

conda activate diffnets

git clone https://github.com/bowman-lab/enspara

conda install -c conda-forge mdtraj=1.8.0

conda install numpy==1.14

conda install cython

conda install mpi4py -c conda-forge

cd enspara

pip install -e .

*Check that enspara was installed successfully*

cd /any/random/directory

python

import enspara

*Great! Now you have enspara installed, which is a dependency of diffnets but is also great on it's own for clustering and building MSMs.*

*Return to /desired/path/for/packages and download diffnets*

cd /desired/path/for/packages

git clone https://github.com/bowman-lab/diffnets

*Install pytorch*

*If you are installing on a mac or CPU only machine use this command. Note: Training on a cpu has not yet been tested and is not recommended*

conda install pytorch torchvision -c pytorch

*If you are installing on a cuda enabled GPU you will need cuda installed. Recommended to use cuda 10.1*

conda install pytorch torchvision cudatoolkit=10.1 -c pytorch

*Install diffnets*

cd diffnets

pip install -e .

*Check that diffnets was installed successfully*

cd /any/random/directory

python

import diffnets

*You now have diffnets installed. When using diffnets, remember to be in the proper conda environment! For example, run this before using diffnets*

conda activate diffnets


## Building the docs / Running the tests

Documentation and testing is in its infancy stages and will be continually updated.

## Brief tutorial

The code is currently organized to be run in 3 separate chunks. Example scripts are in the 'scripts' dir.

First, to process raw trajectories, fill in data_processing_submit.py to match your project directory structure. This will align all trajectories and whiten the data so it is prepared to be input for the DiffNet. This code should be run on a CPU node.

Next, fill in train_submit.py with your desired training parameters. You can see all training parameters in training_dict.txt. When submitting train_submit.py, use a CUDA compatible GPU node. 

Finally, fill in analysis_submit.py to automate some analysis including reconstructiong all trajectories using the autoencoder, calculating an RMSD between autoencoder reconstructed structures and their respective simulation frame, calculating classification labels for all frames, and calculating the latent vector for all frames. Additionally, this script is setup to generate a figure like Figure 6 in the paper.


### Copyright

Copyright (c) 2020, Michael D. Ward, Bowman Lab


#### Acknowledgements
 
Project based on the 
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.3.
