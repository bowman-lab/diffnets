Tutorial
========

The code is currently organized to be run in 3 separate chunks. Tutorial below is for using the command line interface. For more custom solutions, interface with the code directly (examples are in the docs folder at 'example_api_scripts', i.e. data_processing_submit.py, train_submit.py, and analysis_submit.py)

**1. Preprocess simulation data**

The first step is to convert raw trajectories into diffnet input, which should be performed on a CPU node.::

	python /path/to/diffnets/diffnets/cli/main.py process {sim_dirs} {pdb_fns} {outdir}

where {sim_dirs} is a path to an np.array containing directory names. The array needs one directory name (string) for each variant where each directory contains all trajectories for that variant. {pdb_fns} is a path to an np.array containing pdb filenames. The array needs one pdb filename for each variant. The order of variants should match the order of {sim_dirs}. {outdir} is the path you would like processed data to live. Examples of these files can be found in docs/example_cli_files.

**2. Train the DiffNet**

The next step is to actually "train" the DiffNet.::

	python /path/to/diffnets/diffnets/cli/main.py train config.yml

where config.yml contains all the training parameters. Look at docs/train_sample.yml as an example and docs/train_sample.txt for descriptions of each parameter. Training on a GPU gives better performance than on a CPU.

**3. Analysis**

This part of the code will run a set of automated analyses. More on the way...::

	python /path/to/diffnets/diffnets/cli/main.py analyze {data_dir} {net_dir}

This analysis includes reconstructing all trajectories using the DiffNet, calculating an RMSD between DiffNet reconstructed structures and their respective simulation frame, calculating classification labels for all frames, and calculating the latent vector for all frames. Additionally, this script is setup to generate a .pml file in the {net_dir}. Loading {data_dir}/master.pdb into pymol followed by loading this .pml file will generate a figure like Figure 7 in the paper. Finally, this will provide a "morph" directory that contains a PDB containing 10 structures the represent DiffNet labels changing from 0 to 1.


