"""
Basic tests for the py_ssdi package.
"""

import numpy as np
import pytest

from py_ssdi.models import StateSpaceModel, VARModel
from py_ssdi.metrics import dynamical_dependence, causal_emergence
from py_ssdi.utils import orthonormalize, random_orthonormal


def test_state_space_model_creation():
    """Test creation of a state-space model."""
    n, r = 5, 15
    model = StateSpaceModel.create_random(n, r, 0.9)
    
    assert model.n == n
    assert model.r == r
    assert model.A.shape == (r, r)
    assert model.C.shape == (n, r)
    assert model.K.shape == (r, n)
    assert model.V.shape == (n, n)
    
    # Check spectral radius
    eigs = np.linalg.eigvals(model.A)
    assert np.max(np.abs(eigs)) < 1.0


def test_var_model_creation():
    """Test creation of a VAR model."""
    n, p = 5, 3
    model = VARModel.create_random(n, p, 0.9)
    
    assert model.n == n
    assert model.p == p
    assert model.A.shape == (n, n, p)
    assert model.V.shape == (n, n)
    
    # Check stability
    companion = np.zeros((n * p, n * p))
    companion[:n, :] = model.A.reshape(n, n * p)
    companion[n:, :-n] = np.eye(n * (p - 1))
    
    eigs = np.linalg.eigvals(companion)
    assert np.max(np.abs(eigs)) < 1.0


def test_model_conversion():
    """Test conversion between state-space and VAR models."""
    # Create a VAR model
    n, p = 4, 2
    var_model = VARModel.create_random(n, p, 0.9)
    
    # Convert to state-space
    ss_model = var_model.to_state_space()
    
    assert ss_model.n == var_model.n
    assert ss_model.r == var_model.n * var_model.p
    
    # Convert back to VAR (approximately)
    var_model2 = ss_model.to_var()
    
    assert var_model2.n == var_model.n
    assert var_model2.p == ss_model.r


def test_dynamical_dependence():
    """Test calculation of dynamical dependence."""
    # Create a state-space model
    n, r = 5, 15
    model = StateSpaceModel.create_random(n, r, 0.9)
    
    # Create a random projection
    m = 2  # projection dimension
    L = random_orthonormal(n, m)
    
    # Calculate dynamical dependence
    dd = dynamical_dependence(model, L)
    
    assert isinstance(dd, float)
    assert not np.isnan(dd)
    
    # Test with VAR model
    var_model = VARModel.create_random(n, 3, 0.9)
    dd_var = dynamical_dependence(var_model, L)
    
    assert isinstance(dd_var, float)
    assert not np.isnan(dd_var)


def test_causal_emergence():
    """Test calculation of causal emergence."""
    # Create a state-space model
    n, r = 5, 15
    model = StateSpaceModel.create_random(n, r, 0.9)
    
    # Create a random projection
    m = 2  # projection dimension
    L = random_orthonormal(n, m)
    
    # Calculate causal emergence
    ce = causal_emergence(model, L)
    
    assert isinstance(ce, float)
    assert not np.isnan(ce)


def test_orthonormalization():
    """Test orthonormalization of matrices."""
    n, m = 5, 2
    L = np.random.randn(n, m)
    
    # Orthonormalize
    L_ortho = orthonormalize(L)
    
    # Check orthonormality
    assert np.allclose(L_ortho.T @ L_ortho, np.eye(m), atol=1e-10)
    
    # Check random orthonormal generation
    L_random = random_orthonormal(n, m)
    assert np.allclose(L_random.T @ L_random, np.eye(m), atol=1e-10)


if __name__ == "__main__":
    # Run tests
    test_state_space_model_creation()
    test_var_model_creation()
    test_model_conversion()
    test_dynamical_dependence()
    test_causal_emergence()
    test_orthonormalization()
    
    print("All tests passed!") 