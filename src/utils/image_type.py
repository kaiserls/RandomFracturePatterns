import numpy as np

def is_01_image(img: np.ndarray):
    is_array =  type(img) == np.ndarray
    is_2d =  len(img.shape) == 2
    is_01 = np.all(np.logical_and(img >= 0, img <= 1))
    is_float64 = img.dtype == np.float64
    return is_array and is_2d and is_01 and is_float64

def is_01_image_with_threshold(img: np.ndarray, threshold: float):
    is_01 = np.all(np.logical_and(img >= 0, img <= 1))
    is_float64 = img.dtype == np.float64
    is_threshold = 0 <= threshold <= 1 and type(threshold) == float
    return is_01 and is_float64 and is_threshold

def is_0255_image(img: np.ndarray):
    is_array =  type(img) == np.ndarray
    is_2d =  len(img.shape) == 2
    is_0255 = np.all(np.logical_and(img >= 0, img <= 255))
    is_uint8 = img.dtype == np.uint8
    return is_array and is_2d and is_0255 and is_uint8

def is_0255_image_with_threshold(img: np.ndarray, threshold: int):
    is_0255 = np.all(np.logical_and(img >= 0, img <= 255))
    is_uint8 = img.dtype == np.uint8
    is_threshold = 0 <= threshold <= 255 and type(threshold) == int
    return is_0255 and is_uint8 and is_threshold

def transform_01_to_0255(data):
    assert is_01_image(data)
    data_01 = (data - data.min()) / (data.max() - data.min())
    data_0255 = (data_01 * 255).astype(np.uint8)
    assert is_0255_image(data_0255)
    return data_0255