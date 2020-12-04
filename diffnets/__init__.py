"""
diffnets
Supervised and self-supervised autoencoders to identify the mechanistic basis for biochemical differences between protein variants.
"""

# Add imports here
from .training import Trainer
from .data_processing import ProcessTraj
from .analysis import Analysis

# Handle versioneer
from ._version import get_versions
versions = get_versions()
__version__ = versions['version']
__git_revision__ = versions['full-revisionid']
del get_versions, versions
