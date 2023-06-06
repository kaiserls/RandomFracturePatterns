import numpy as np
import matplotlib.pyplot as plt

def crack_volume(Z, dA, threshold=0.9):
    assert len(Z.shape) == 2 # Only for 2d image
    volume = np.sum(Z > threshold) * dA
    return volume


def fractal_dimension(Z:np.ndarray, threshold, plot=False):
    # Get the type of the numpy array Z
    Z_type = Z.dtype
    # If the type is float, also the threshold should be float and between 0 and 1
    if Z_type == np.float64:
        assert 0 <= threshold <= 1 and type(threshold) == float
    elif Z_type == np.uint8:
        assert 0 <= threshold <= 255 and type(threshold) == int
    else:
        raise ValueError(
            f"Z has dtype {Z_type}, but it should be either np.float64 or np.uint8."
        )

    # Only for 2d image
    assert len(Z.shape) == 2

    # assert that the image is in [0, ]

    # From https://github.com/rougier/numpy-100 (#87)
    def boxcount(Z, k):
        S = np.add.reduceat(
            np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0),
            np.arange(0, Z.shape[1], k),
            axis=1,
        )

        # We count non-empty (0) and non-full boxes (k*k)
        return len(np.where((S > 0) & (S < k * k))[0])

    # Transform Z into a binary array
    Z = (Z < threshold)
    if plot:
        plt.imshow(Z, interpolation="none", cmap="gray")
        plt.colorbar()
        plt.show()

    # Minimal dimension of image
    p = min(Z.shape)

    # Greatest power of 2 less than or equal to p
    n = 2 ** np.floor(np.log(p) / np.log(2))

    # Extract the exponent
    n = int(np.log(n) / np.log(2))

    # Build successive box sizes (from 2**n down to 2**1)
    sizes = 2 ** np.arange(n, 1, -1)

    # Actual box counting with decreasing size
    counts = []
    for size in sizes:
        counts.append(boxcount(Z, size))

    # Fit the successive log(sizes) with log (counts)
    coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
    return -coeffs[0]
