import numpy as np
import matplotlib.pyplot as plt


def crack_volume(Z, dA, threshold=0.9):
    # Only for 2d image
    assert len(Z.shape) == 2

    # Calculate the volume of the crack.
    # The crack volume is defined as the number of pixels that are above the threshold times the area of a pixel.
    # The area of a pixel is dA.

    volume = np.sum(Z > threshold) * dA
    return volume


def fractal_dimension(Z, threshold=0.9, plot=False):
    # Only for 2d image
    assert len(Z.shape) == 2

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
    Z = Z < threshold
    if plot:
        plt.imshow(Z)
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
