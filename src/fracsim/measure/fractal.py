import numpy as np
import matplotlib.pyplot as plt

from fracsim.utils.image_type import is_01_image_with_threshold

# Source: https://gist.github.com/viveksck/1110dfca01e4ec2c608515f0d5a5b1d1
def fractal_dimension(Z:np.ndarray, threshold: float, plot=False) -> float:
    """Estimate the fractal dimension of a 2d image using the box-counting method and a polynomial fit to the log-log plot of the box-count vs box-size.

    Args:
        Z (np.ndarray): Image given asn numpy array with dtype either np.float64 in the range [0, 1]. Use 0 to denote background(empty) and 1 to denote an object(filled).
        threshold (float): Threshold over which an image pixel is considered to be filled.
        plot (bool, optional): Plot the image. Defaults to False.

    Returns:
        float: Fractal dimension of the image.
    """    

    assert is_01_image_with_threshold(Z, threshold)

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
    Z = (Z > threshold)
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
