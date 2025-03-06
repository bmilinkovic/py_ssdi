# py_ssdi: Python Implementation of Dynamical Independence

This package implements Dynamical Independence computation for linear state-space systems, as described in:

L. Barnett and A. K. Seth, *Dynamical independence: discovering emergent macroscopic processes in complex dynamical systems*, [arXiv:2106.06511 [nlin.AO]](https://arxiv.org/abs/2106.06511), 2021.

This is a Python implementation of the original MATLAB [SSDI-1 toolbox](https://github.com/lcbarnett/ssdi).

## Author

Borjan Milinkovic (borjan.milinkovic@gmail.com)

## Installation

### Using Conda (recommended)

```bash
# Clone the repository
git clone https://github.com/borjanm/py_ssdi.git
cd py_ssdi

# Create and activate the conda environment
conda env create -f environment.yml
conda activate py_ssdi

# Install the package in development mode
pip install -e .
```

The conda environment includes the following main dependencies:
- Python 3.9
- NumPy
- SciPy
- Matplotlib
- pandas
- scikit-learn
- networkx
- control (via pip)
- slycot (optional dependency for control)

### Using pip

```bash
pip install py_ssdi
```

## Usage

### Basic Dynamical Independence Analysis

```python
import numpy as np
from py_ssdi.models import StateSpaceModel, VARModel
from py_ssdi.metrics import dynamical_dependence, causal_emergence

# Create a random VAR model
n = 5  # dimension
r = 3  # model order
A, K = VARModel.create_random(n, r, 0.9)  # spectral radius 0.9

# Create a random state-space model
ss_model = StateSpaceModel.create_random(n, 3*n, 0.9)

# Compute dynamical dependence for a projection
L = np.random.randn(n, 2)  # random projection to 2D
L, _ = np.linalg.qr(L)  # orthonormalise

dd = dynamical_dependence(ss_model, L)
ce = causal_emergence(ss_model, L)

print(f"Dynamical Dependence: {dd}")
print(f"Causal Emergence: {ce}")
```

### Creating Models with Modular Connectivity Patterns

The new connectivity module allows you to create state-space models with specific modular connectivity patterns:

```python
from py_ssdi.connectivity import (
    create_modular_connectivity,
    create_canonical_9node_model,
    create_canonical_16node_model,
    visualise_connectivity
)

# Create a canonical 9-node model with 3 modules (sizes 2, 3, 4)
model_9node = create_canonical_9node_model(rho=0.9, rmii=0.2)

# Create a model with custom module sizes and connections
module_sizes = [5, 7, 8, 4]
inter_module_connections = [(0, 1), (1, 2), (2, 3), (3, 0)]  # Ring structure
custom_model = create_modular_connectivity(
    module_sizes,
    inter_module_connections,
    rho=0.8,  # spectral radius
    intra_module_density=0.7,  # connection density within modules
    rmii=0.1  # residuals multiinformation
)

# Visualise the connectivity structure
fig = visualise_connectivity(custom_model, module_sizes)
```

## Features

- State-space model creation and analysis
- VAR model creation and analysis
- Dynamical Independence metrics
- Causal Emergence calculation
- Model transformation and optimisation
- Visualisation tools for causal networks
- Modular connectivity patterns for state-space models:
  - Create models with specified module sizes and inter-module connections
  - Canonical model configurations (9-node, 16-node, 68-node)
  - Visualisation of connectivity structures

## Example Scripts

The package includes example scripts to demonstrate its functionality:

### Basic Example (py_ssdi/examples/basic_example.py)

Demonstrates the core functionality of the package:
- Creating random state-space and VAR models
- Computing dynamical dependence and causal emergence
- Optimising dynamical dependence using gradient descent
- Visualising optimisation results and causal graphs

### Connectivity Example (py_ssdi/examples/connectivity_example.py)

Showcases the new connectivity module:
- Creating state-space models with modular connectivity patterns
- Visualising the connectivity structure with module boundaries
- Optimising dynamical dependence across multiple macroscopic scales
- Comparing random vs. optimised projections for different scales

To run the examples:

```bash
# Run the basic example
python -m py_ssdi.examples.basic_example

# Run the connectivity example
python -m py_ssdi.examples.connectivity_example
```

## License

MIT License 