# py_ssdi

A Python package for Dynamical Independence analysis on Linear State-space models. This is porting of the original MATLAB code for Dynamical Independence (ssdi) developed by Dr. Lionel Barnett at the Sussex Centre for Consciousness Science (SCCS), University of Sussex. His original, MATLAB code can be found at https://github.com/lcbarnett/ssdi. 

## Installation

You can install the package using pip:

```bash
pip install py_ssdi
```

Or install from source:

```bash
git clone https://github.com/bmilinkovic/py_ssdi.git
cd py_ssdi
pip install -e .
```

## Features

- Generate random VAR(n, model_order) models using prespecified connectivity patterns
- Create Linear State-Space models with configurable hidden dimensions
- Model parameter transformation and optimisation utilities
- Visualisation tools for DI analysis

## Quick Start

```python
from py_ssdi import simulate_model

# Generate a VAR model
model_path = simulate_model(
    model_type="VAR",
    n=9,                # number of observed variables
    r_var=2,           # VAR order (number of lags)
    connectivity_name="tnet9x",
    seed=0,
    spectral_norm=0.95,
    decay=1.0
)

# Generate a State-Space model
model_path = simulate_model(
    model_type="SS",
    n_obs=5,           # observation dimension
    seed=0,
    spectral_norm=0.95
)
```

## Package Structure

- `model_generation/`: Utilities for generating VAR and State-Space models
- `optimisation/`: Dynamical Dependence optimisation tools
- `preoptimisation/`: Dynamical Dependence Pre-optimisation tools
- `visualisation/`: Visualisation and Plotting utilities for analysis visualisation
- `results/`: Directory for storing generated results. 

## Requirements

- Python >= 3.7
- NumPy >= 1.20.0
- SciPy >= 1.7.0

## License

This project is licensed under the MIT License - see the LICENSE file for details.
