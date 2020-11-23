Installation
===============

DiffNets is currently under development, and cannot yet be installed with PyPI. Therefore, the install instructions are long-winded, but hopefully sufficiently explicit.

**These instructions assume you have the python package manager `conda` installed.**

Go to directory you would like diffnets (and enspara) to live in::

	cd /desired/path/for/packages

Create a conda environment that will be used for diffnets::

	conda create --name diffnets python=3.6

Enter this conda environment and install enspara::

	conda activate diffnets
	git clone https://github.com/bowman-lab/enspara
	conda install -c conda-forge mdtraj=1.8.0
	conda install numpy==1.14
	conda install cython
	conda install mpi4py -c conda-forge
	cd enspara
	pip install -e .

Check that enspara was installed successfully::

	cd /any/random/directory
	python
	import enspara

Great! Now you have enspara installed, which is a dependency of diffnets but is also great on it's own for clustering and building MSMs.

Return to /desired/path/for/packages and download diffnets::

	cd /desired/path/for/packages
	git clone https://github.com/bowman-lab/diffnets

Install pytorch

If you are installing on a mac or CPU only machine use this command. Note: Training is much slower on CPUs::

	conda install pytorch torchvision -c pytorch

If you are installing on a cuda enabled GPU you will need cuda installed. Recommended to use cuda 10.1::

	conda install pytorch torchvision cudatoolkit=10.1 -c pytorch

Install Click::

        conda install click

Install diffnets::

	cd diffnets
	pip install -e .

Check that diffnets was installed successfully::

	cd /any/random/directory
	python
	import diffnets

You now have diffnets installed. When using diffnets, remember to be in the proper conda environment! For example, run this before using diffnets::

	conda activate diffnets

