import skimage
import numpy as np
import pyvista as pv
from fracsim.utils.structured_mesh import scale_pixels_to_coordinates
from fracsim.utils.image_type import is_01_image_with_threshold


def isoline(image_01, iso_value):
    assert is_01_image_with_threshold(
        image_01, iso_value
    ), "image_01 must be a 01 image and threshold must be between 0 and 1"
    contours = skimage.measure.find_contours(image_01, iso_value)
    return contours


def scale_isolines_to_coordinates(mesh, isolines):
    return [scale_pixels_to_coordinates(mesh, c) for c in isolines]


# def isolines_from_vtk(mesh_file, iso_value):
#     """Return the isolines of the OP data from the vtk file belonging to app"""
#     mesh, data = mesh_and_2d_data_from_vtk(mesh_file)
#     assert 0 <= iso_value <= 1, "iso_value must be between 0 and 1, because OP vtk data is in range [0, 1]"
#     return isolines_from_mesh_data(mesh, data, iso_value=iso_value)

# def isolines_from_mesh_data(mesh, data, iso_value)
#     # extract the isolines from the data
#     assert 0 <= iso_value <= 1, "iso_value must be between 0 and 1 for this function"
#     contours = skimage.measure.find_contours(data, iso_value)
#     if scaled:
#         contours = [scale_pixels_to_coordinates(mesh, c) for c in contours]
#     return contours

# def interpolate_isolines(isolines, target_n_points, x_min, x_max):
#     """Interpolate the isolines to n points"""
#     x = np.linspace(x_min, x_max, target_n_points)
#     return [
#         np.array([x, interp1d(c[:, 0], c[:, 1], kind="cubic")(x)]).T for c in isolines
#     ]

# def isolines_image_cv2(mesh_file, iso_value, contour_thickness):
#     """Load the strucuted grid with data OP from the vtk file, generate contours with cv2,
#     return an image with the contours drawn on it"""

#     mesh, data = mesh_data_from_vtk(mesh_file)
#     data = data_to_0_255(data)
#     ret, thresh = cv2.threshold(src=data, thresh=255 * iso_value, maxval=255, type=0)

#     imagesize = data.shape
#     img = np.zeros((*imagesize, 3), dtype=np.uint8)

#     contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#     cv2.drawContours(image=img, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=contour_thickness)
#     return img
