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
L, _ = np.linalg.qr(L)  # orthonormalize

dd = dynamical_dependence(ss_model, L)
ce = causal_emergence(ss_model, L)

print(f"Dynamical Dependence: {dd}")
print(f"Causal Emergence: {ce}")
```

## Features

- State-space model creation and analysis
- VAR model creation and analysis
- Dynamical Independence metrics
- Causal Emergence calculation
- Model transformation and optimization
- Visualization tools for causal networks
- Modular connectivity patterns for state-space models

## License

MIT License 