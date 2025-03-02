"""
Python implementation of Dynamical Independence for state-space systems

Based on the MATLAB SSDI-1 toolbox:
L. Barnett and A. K. Seth, "Dynamical independence: discovering emergent macroscopic 
processes in complex dynamical systems", arXiv:2106.06511 [nlin.AO], 2021.
"""

__version__ = "0.1.0"

# Import key components for easier access
from py_ssdi.models import StateSpaceModel, VARModel
from py_ssdi.metrics import dynamical_dependence, causal_emergence, dynamical_independence_gradient
from py_ssdi.utils import random_orthonormal, orthonormalize, optimize_dynamical_dependence, cluster_projections

# Import connectivity module
from py_ssdi.connectivity import (
    create_modular_connectivity,
    create_canonical_9node_model,
    create_canonical_16node_model,
    create_canonical_68node_model,
    visualize_connectivity
) 