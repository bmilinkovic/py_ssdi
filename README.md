# PY_SSDI (Beta version)

This is a Python package for Dynamical Independence (DI) analysis on Linear State-space (SS) models. DI is a particular framework for quantifying and detecting macroscopic observables directly from continuous-valued time-series data. 

This package is a porting of the original MATLAB code for Dynamical Independence (ssdi) developed by Dr. Lionel Barnett at the Sussex Centre for Consciousness Science (SCCS), University of Sussex. His original, MATLAB code can be found at https://github.com/lcbarnett/ssdi. 

If you have any questions or want to report any bugs please do so through the standard github channels or contact the author at: borjan.milinkovic@gmail.com

## Installation

You can install the package using pip (currently not working--but hopefully in the future):

```bash
pip install py_ssdi
```

Or install from source (recommended):

```bash
git clone https://github.com/bmilinkovic/py_ssdi.git
cd py_ssdi
pip install -e .
```

## Features

- Generates random VAR(n, model_order) models using prespecified connectivity patterns
- Create Linear State-Space models with configurable hidden dimensions
- Model parameter transformation from VAR to SS models (used in the tutorial) 
- Performs a preoptimisation procedure on a proxy Dynamical Dependence objective function
- Performs an optimisation procedure on of Dynamical Dependence utilising the spectral method (computationally more efficient)
- Implements visualisation tools for DI analysis

## Quick Start

```python
from py_ssdi import simulate_model

# Generate a VAR model
model_path = simulate_model(
    model_type="VAR",
    n=9,                # number of observed variables
    r_var=2,           # VAR order (number of lags)
    connectivity_name="tnet9x", # uses a prespecified connectivity
    seed=0,
    spectral_norm=0.95,
    decay=1.0
)

# Generate a State-Space model
model_path = simulate_model(
    model_type="SS",
    n_obs=5,           # observation dimension (number of observed variables, *not* the hidden dimension)
    seed=0,
    spectral_norm=0.95
)
```

## Package Structure

- `model_fitting/`: Fits VAR and SS models to raw time-series data (still under construction)
- `model_generation/`: Utilities for generating VAR and State-Space models
- `optimisation/`: Dynamical Dependence optimisation tools
- `preoptimisation/`: Dynamical Dependence Pre-optimisation tools
- `visualisation/`: Visualisation and Plotting utilities for analysis visualisation
- `results/`: Directory for storing generated results. 

## Requirements (the basics)

- Python >= 3.7
- NumPy >= 1.20.0
- SciPy >= 1.7.0

## License

This project is licensed under the General GNU - see the LICENSE file for details.
