Analysis
=================
Once a DiffNet is trained, there are some automated analyses that can be run with the **Analysis** class. The *run_core* method should only be used with the sae/split_sae architecture. However, other methods are useful even for standard autoencoders. The *run_core* method computes the following for all simulation frames: the low-dimensional representation, the DiffNet classification label (i.e. a structure's association with a biochemical property), reconstructs the structure, and computes the rmsd comparing reconstructed structures to actual structures from simulation. Other methods help plotting labels assigned to each variant and creating a morph showing how structures change as the DiffNet label changes from 0 to 1.

.. autosummary::
   :toctree: autosummary

   diffnets.analysis.Analysis


