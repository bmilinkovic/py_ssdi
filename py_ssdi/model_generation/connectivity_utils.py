# connectivity_utils.py
import numpy as np

def tnet9x():
    """
    Load the 9-node "X" topology connectivity matrix.
    Returns a (9 x 9) binary adjacency matrix.
    """
    C = np.array([
        [1, 1, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 1, 1, 0, 0, 0],
        [0, 0, 1, 1, 1, 1, 0, 0, 0],
        [1, 0, 1, 1, 1, 1, 0, 0, 0],
        [0, 0, 1, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 1, 1],
        [0, 0, 0, 1, 0, 0, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 1, 1, 1]
    ], dtype=int)
    return C


def tnet9d():
    """
    Load the 9-node "D" topology connectivity matrix.
    Returns a (9 x 9) binary adjacency matrix.
    """
    C = np.array([
        [1, 0, 1, 1, 0, 0, 0, 0, 0],
        [1, 1, 1, 0, 1, 0, 0, 0, 1],
        [0, 0, 1, 1, 1, 0, 0, 0, 0],
        [1, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 1, 0, 0],
        [0, 0, 0, 1, 0, 1, 0, 1, 1],
        [0, 0, 0, 0, 0, 0, 1, 1, 1]
    ], dtype=int)
    return C


def load_connectivity(name):
    """
    Load a predefined connectivity by name.
    name: string, e.g. 'tnet9x' or 'tnet9d'.
    Returns an (n x n) adjacency matrix. Raises ValueError if unknown.
    """
    name = name.lower()
    if name == 'tnet9x':
        return tnet9x()
    elif name == 'tnet9d':
        return tnet9d()
    else:
        raise ValueError(f"Unknown connectivity '{name}'")

