Featurization
=================
Featurization is the process of converting molecular dynamics simulation data into input for DiffNet training. In data_processing.py, we provide two classes **ProcessTraj** and **WhitenTraj** for this featurization. Currently, the only featurization we have setup is for training on normalized XYZ atom coordinates. Plans for alternative featurization schemes are in the works, but the functionality has not been implemented at the featuization level, or the DiffNet training architecture level. 

ProcessTraj
-----------
ProcessTraj allows ones to process simulation trajectories. Specifically, it provides functionality to operate on sets of trajectories and corresponding topology files. The main goal is to process trajectories with different numbers of atoms to an equivalent set of atoms across all trajectories, and then to normalize each atom's position. The *run* method puts all of this together. It selects a common subset of atoms (C,CA,CB,N by default), aligns all trajectories, calculates the average center of mass of each atom, and performs various other bookkeeping that is ultimately useful in the context of the next step in the pipeline, training. Using the *run* method will create an output directory that has the necessary structure and data to train a DiffNet. 

.. autosummary::
   :toctree: autosummary

   diffnets.data_processing.ProcessTraj


WhitenTraj
----------

WhitenTraj is a class that perfroms a data whitening procedure [1] that removes covariance between atoms in trajectories, which helps with DiffNet training. Specifically, it takes a directory of aligned trajectories (.xtc's) and a center of mass file that provides the average center of mass of each atom averaged over the aligned trajectories (these inputs should be created by ProcessTraj). The *run* method ultimately provides a file for a covariance matrix, a whitening matrix, and an unwhitening matrix.

[1] Wehmeyer C, Noé F. Time-lagged autoencoders: Deep learning of slow collective variables for molecular kinetics. J Chem Phys. 2018. doi:10.1063/1.5011399   

.. autosummary::
   :toctree: autosummary

   diffnets.data_processing.WhitenTraj
