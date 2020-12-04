Training
=================
In the context of DiffNets, training is the process of learning how to compute a low-dimensional representation of a protein structure AND learning how to classify the structure based on the likelihood that is associated with the biochemical property of interest. The **Trainer** class handles the DiffNet training process.

Trainer
-----------
This class takes a dictionary of parameters. You can view these parameters in **docs/example_api_scripts/training_dict.txt** and find an example training script in **docs/example_api_scripts/train_submit.py**. The method *run* will carry out all training. The actual training loop is contained with the method *train*.

.. autosummary::
   :toctree: autosummary

   diffnets.training.Trainer


Different Architecture Choices
------------------------------
The DiffNets code base contains several architectures including standard autoencoders, supervised autoencoders and a version of each of these that can split the input into 2 separate encoders (e.g. to focus the classification task on an important region of the protein). You will choose an architecture as one parameter included in your training. Keep in mind that choosing a "split" architecture will require you to supply extra parameters to your Trainer object (inds1 and inds2) to denote which part of the protein goes into which encoder.

.. autosummary::
   :toctree: autosummary

   diffnets.nnutils.ae
   diffnets.nnutils.sae
   diffnets.nnutils.split_ae
   diffnets.nnutils.split_sae


