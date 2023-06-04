import numpy as np
import matplotlib.pyplot as plt

# The proposed following measures are proposed to quantify the randomness of the crack:

# * Global quantities:
#   * Fractal dimension
#   * crack length

# * Local quantities:
#   * crack width
#   * Deviation of the crack path from the crack path of the mean parameters

# The local quantities can be transformed into global quantities by averaging over the crackpath or taking the maximum.

# They are implemented in the python files contained in ```src/measure/*.py```.


# Properties we should aim for when designing the measures:
# Independence of the grid size / number of contour points


# Local measures:
def crack_width(isolines):
    """Calculates the crack with from a list of two isolines, the bottom and the top of the crack."""
    if len(isolines) != 2:
        raise ValueError(
            "There should be exactly two isolines. There are "
            + str(len(isolines))
            + " isolines."
        )
    isolines = sorted(isolines, key=lambda x: x[0, 0])
    width = isolines[1][:, 1] - isolines[0][:, 1]
    assert np.all(width >= 0)
    return width


def crack_deviation(isolines, reference_isolines):
    """Calculates the deviation of the crack path from the reference crack path."""
    if len(isolines) != 2:
        raise ValueError("There should be exactly two isolines")
    if len(reference_isolines) != 2:
        raise ValueError("There should be exactly two isolines")
    isolines = sorted(isolines, key=lambda x: x[0, 0])
    reference_isolines = sorted(reference_isolines, key=lambda x: x[0, 0])
    deviation = np.abs(isolines[1][:, 1] - reference_isolines[1][:, 1]) + np.abs(
        isolines[0][:, 1] - reference_isolines[0][:, 1]
    )
    return deviation


# Global measures:
# TODO: The crack length is not independent of the grid size and will probably never be.
# It could be a fractal and therefore change its length with the grid size.
def crack_length(isolines):
    """Calculates the crack length from a list of two isolines, the bottom and the top of the crack."""
    if len(isolines) != 2:
        raise ValueError("There should be exactly two isolines")
    isolines = sorted(isolines, key=lambda x: x[0, 0])
    d_iso1 = np.diff(isolines[0], axis=0)
    d_iso2 = np.diff(isolines[1], axis=0)
    lengths1 = np.sqrt((d_iso1**2).sum(axis=1))
    lengths2 = np.sqrt((d_iso2**2).sum(axis=1))
    length = lengths1.sum() + lengths2.sum()
    return length
