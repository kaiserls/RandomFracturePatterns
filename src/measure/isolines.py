import skimage
import numpy as np
import pyvista as pv
import cv2

from scipy.interpolate import interp1d


from src.utils.structured_mesh import (
    scale_pixels_to_coordinates,
    reshape_data,
    reshape_points,
)


def isolines_from_vtk(mesh_file, iso_value, scaled=True):
    """Return the isolines of the OP data from the vtk file belonging to app"""
    mesh, data = mesh_data_from_vtk(mesh_file)
    assert 0 <= iso_value <= 1, "iso_value must be between 0 and 1, because OP vtk data is in range [0, 1]"
    return isolines_from_mesh_data(mesh, data, iso_value=iso_value, scaled=scaled)

def isolines_from_mesh_data(mesh, data, iso_value, scaled=True):
    # extract the isolines from the data
    assert 0 <= iso_value <= 1, "iso_value must be between 0 and 1 for this function"
    contours = skimage.measure.find_contours(data, iso_value)
    if scaled:
        contours = [scale_pixels_to_coordinates(mesh, mesh.points, c) for c in contours]
    return contours

def interpolate_isolines(isolines, target_n_points, x_min, x_max):
    """Interpolate the isolines to n points"""
    x = np.linspace(x_min, x_max, target_n_points)
    return [
        np.array([x, interp1d(c[:, 0], c[:, 1], kind="cubic")(x)]).T for c in isolines
    ]

def mesh_data_from_vtk(mesh_file):
    mesh = pv.read(mesh_file)
    data = reshape_data(mesh, mesh["OP"])
    return mesh, data

def data_to_0_255(data):
    data_01 = (data - data.min()) / (data.max() - data.min())
    return (data_01 * 255).astype(np.uint8)

def isolines_image_cv2(mesh_file, iso_value, contour_thickness):
    """Load the strucuted grid with data OP from the vtk file, generate contours with cv2,
    return an image with the contours drawn on it"""

    mesh, data = mesh_data_from_vtk(mesh_file)
    data = data_to_0_255(data)
    ret, thresh = cv2.threshold(src=data, thresh=255 * iso_value, maxval=255, type=0)

    imagesize = data.shape
    img = np.zeros((*imagesize, 3), dtype=np.uint8)

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image=img, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=contour_thickness)
    return img
