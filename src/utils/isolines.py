import skimage
import numpy as np
import pyvista as pv

from scipy.interpolate import interp1d


from src.utils.mesh_helpers import (
    scale_pixels_to_coordinates,
    reshape_data,
    reshape_points,
)


def isolines_from_vtk(mesh_file, scaled=True):
    """Return the isolines of the OP data from the vtk file belonging to app"""
    mesh = pv.read(mesh_file)
    data = reshape_data(mesh, mesh["OP"])
    return isolines(mesh, data, scaled=scaled)


def interpolate_isolines(isolines, target_n_points, x_min, x_max):
    """Interpolate the isolines to n points"""
    x = np.linspace(x_min, x_max, target_n_points)
    return [
        np.array([x, interp1d(c[:, 0], c[:, 1], kind="cubic")(x)]).T for c in isolines
    ]


def isolines(mesh, data, iso_value, scaled=True):
    # extract the isolines from the data
    contours = skimage.measure.find_contours(data, iso_value)
    if scaled:
        contours = [scale_pixels_to_coordinates(mesh, mesh.points, c) for c in contours]
    return contours


def plot_isolines_pyvista_skiimage(mesh_file):
    import matplotlib.pyplot as plt

    """Load the structured grid with data from the mesh file, extract the numpy data array and plot the isolines using skimage"""
    mesh = pv.read(mesh_file)
    data_lin = mesh["OP"]
    data = reshape_data(mesh, data_lin)
    plt.imshow(data.T, cmap="gray", extent=mesh.bounds[0:4], origin="lower")
    contours = isolines(mesh, data)
    for i, c in enumerate(contours):
        plt.plot(c[:, 0], c[:, 1], marker="x", label=f"c{i}")

    min_x, max_x, min_y, max_y, min_z, max_z = mesh.bounds
    plt.xlim(min_x, max_x)
    plt.ylim(min_y, max_y)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()


def isolines_image_cv2(
    mesh_file, iso_value
):  # , imagesize = (1000,2000)):
    import cv2 as cv

    """Load the strucuted grid with data OP from the vtk file, generate contours with cv2,
    return an image with the contours drawn on it"""

    mesh = pv.read(mesh_file)
    data_lin = mesh["OP"]
    data = reshape_data(mesh, data_lin)

    # convert data to gray scale image
    data = (data - data.min()) / (data.max() - data.min())
    data = (data * 255).astype(np.uint8)
    imagesize = data.shape
    img = np.zeros((*imagesize, 3), dtype=np.uint8)

    ret, thresh = cv.threshold(data, 255 * iso_value, 255, 0)
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(img, contours, -1, (0, 255, 0), 3)
    return img


def plot_isolines_mlab(mesh_file, output_file):
    from mayavi import mlab

    # Make a figure with a black background
    fig = mlab.figure(size=(600, 1200), bgcolor=(0, 0, 0))
    # Also see methods like: fig.scene.z_plus_view(), etc
    fig.scene.camera.azimuth(215)

    source = mlab.pipeline.open(mesh_file)
    source.point_scalars_name = "OP"
    # Show the surface, colored by the scalars
    # surf = mlab.pipeline.surface(source)
    # Draw contours of the scalars on the surface
    lines = mlab.pipeline.contour_surface(source, contours=[0.9])
    mlab.savefig(filename=output_file, size=(100, 200))
    mlab.show()
