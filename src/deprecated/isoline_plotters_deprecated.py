import pyvista as pv

from src.utils.structured_mesh import (
    reshape_data,
)


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

def plot_isolines_pyvista_skiimage(mesh_file):
    import matplotlib.pyplot as plt

    """Load the structured grid with data from the mesh file, extract the numpy data array and plot the isolines using skimage"""
    mesh = pv.read(mesh_file)
    data_lin = mesh["OP"]
    data = reshape_data(mesh, data_lin)
    plt.imshow(data.T, cmap="gray", extent=mesh.bounds[0:4], origin="lower")
    contours = isolines_from_mesh_data(mesh, data)
    for i, c in enumerate(contours):
        plt.plot(c[:, 0], c[:, 1], marker="x", label=f"c{i}")

    min_x, max_x, min_y, max_y, min_z, max_z = mesh.bounds
    plt.xlim(min_x, max_x)
    plt.ylim(min_y, max_y)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()