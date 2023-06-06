import numpy as np

def count(Z, threshold):
    assert len(Z.shape) == 2 # Only for 2d image
    count = np.sum(Z > threshold)
    return count

def volume(Z, dA, threshold):
    return count(Z, threshold) * dA
