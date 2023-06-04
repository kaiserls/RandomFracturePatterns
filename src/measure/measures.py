from src.measure.fractal import fractal_dimension, crack_volume
from src.measure.simple import crack_width, crack_deviation, crack_length
from src.tools.isolines import (
    isolines_from_app,
    interpolate_isolines,
    isolines_image_cv2,
)

import numpy as np
import pandas as pd

import src.simulation_config as cfg


def pixel_area():
    dx = (cfg.measure_max_X - cfg.measure_min_X) / cfg.measure_discretization_X
    dy = (cfg.measure_max_Y - cfg.measure_min_Y) / cfg.measure_discretization_Y
    return dx * dy


def calculate_measures(app: str, sample: int, ref_isoline: np.ndarray):
    width, deviation, length, dimension, volume = np.nan, np.nan, np.nan, np.nan, np.nan

    img = isolines_image_cv2(app, sample)
    dimension = fractal_dimension(img[:, :, 1])

    dA = pixel_area()
    volume = crack_volume(Z=img[:, :, 1], dA=dA)

    isolines = isolines_from_app(app, sample)
    length = crack_length(isolines)

    try:
        isolines = interpolate_isolines(isolines)
        width = np.mean(crack_width(isolines))
        deviation = np.mean(crack_deviation(ref_isoline, isolines))
    except Exception as e:
        print(f"Error while measuring {app} sample {sample}: {e}")
    finally:
        return width, deviation, length, dimension, volume


def create_measure_dataframe():
    entries = ["App", "width", "deviation", "length", "dimension", "volume"]
    df = pd.DataFrame(columns=entries)
    df = df.set_index("App")
    return df
