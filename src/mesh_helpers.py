import meshio
import numpy as np
import pyvista as pv
import src.simulation_config as cfg

"""
*Note: meshio as well as newer vtk version dont understand the data written by PANDAS. The format is the tecplot format in ASCII with the file ending ```.dat``` .*
This tool provides a function ```clean_mesh``` and executable code to remove the gauss nodes which are not contained in the mesh and can not be read by most programs. The executable code takes my first example file and saves it as clean ```.vtk``` and ```.dat``` file. 
"""


def to_structured_pv(input_vtk, output_vtk, field_name: str = "OP", plot=False):
    """Use pyvista to sample the unstructured grid data with the given field name to a structured grid and save it to the given output path as vtk"""
    ugrid = pv.UnstructuredGrid(input_vtk)

    x = np.linspace(
        cfg.measure_min_X,
        cfg.measure_max_X,
        cfg.measure_discretization_X + 1,
        dtype=np.float32,
    )
    y = np.linspace(
        cfg.measure_min_Y,
        cfg.measure_max_Y,
        cfg.measure_discretization_Y + 1,
        dtype=np.float32,
    )
    z = np.linspace(0.0, 0.0, 1, dtype=np.float32)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    stgrid = pv.StructuredGrid(X, Y, Z)

    result = stgrid.sample(ugrid)
    stgrid.point_data[field_name] = result[field_name]
    stgrid.save(output_vtk)

    if plot:
        plotter = pv.Plotter()
        plotter.add_mesh(
            stgrid, color="white", opacity=0.9, scalars="OP", show_edges=True
        )
        plotter.show()


def clean_mesh(mesh: meshio.Mesh) -> meshio.Mesh:
    n_points_total = len(mesh.points)
    point_used = np.full((n_points_total,), False, dtype=bool)

    # Save which points are used by "universe array"
    cell_types = mesh.cells_dict.keys()
    for ct in cell_types:
        cells = mesh.cells_dict[ct]
        for cell in cells:
            point_used[cell] = True
    n_points_total = len(mesh.points)
    point_used = np.full((n_points_total,), False, dtype=bool)

    # Save which points are used by "universe array"
    cell_types = mesh.cells_dict.keys()
    for ct in cell_types:
        cells = mesh.cells_dict[ct]
        for cell in cells:
            point_used[cell] = True

    # Clean up points and point data
    points_cleaned = mesh.points[point_used]
    point_data_cleaned = {
        var_name: mesh.point_data[var_name][point_used]
        for var_name in mesh.point_data.keys()
    }

    # Calculate shift of point ids:
    point_id_shift = np.cumsum(
        np.logical_not(point_used)
    )  # Count the number of ignored points to get a shift for each point id

    # Clean up cell definitions by shifting ids
    cells_dict_cleaned = {ct: np.copy(mesh.cells_dict[ct]) for ct in cell_types}
    for ct in cell_types:
        cells = mesh.cells_dict[ct]
        for i, cell in enumerate(cells):
            cleaned_cell = cell - point_id_shift[cell]
            cells_dict_cleaned[ct][i] = cleaned_cell
    cells_cleaned = list(cells_dict_cleaned.items())

    # Assemble cleaned mesh
    mesh_cleaned = meshio.Mesh(
        points_cleaned,
        cells_cleaned,
        point_data=point_data_cleaned,
        cell_data=mesh.cell_data,
    )
    return mesh_cleaned


def tec_to_vtk(input_tec, output_vtk):
    """Converts all tecplot files in the app folder to vtk files and remove second order nodes."""
    try:
        mesh = meshio.read(input_tec)
        mesh_cleaned = clean_mesh(mesh)
        meshio.write(output_vtk, mesh_cleaned)
    except FileNotFoundError as e:
        print("I caught the following error for you: ", str(e))
        print("Skipping the file")


def scale_pixels_to_coordinates(mesh, points, pixel_coordinates):
    # scale the pixel coordinates to the coordinates of the mesh
    # points are the coordinates of the mesh
    # pixel_coordinates are the coordinates of the pixel
    min_x, max_x, min_y, max_y, min_z, max_z = mesh.bounds

    # scale the pixel coordinates to the coordinates of the mesh
    x = pixel_coordinates[:, 0]
    y = pixel_coordinates[:, 1]
    # z = pixel_coordinates[:,2]
    # print("points.shape", points.shape)
    dx = (max_x - min_x) / (mesh.dimensions[0] + (-1))
    dy = (max_y - min_y) / (mesh.dimensions[1] + (-1))
    # dz = (max_z - min_z) / points.shape[2]

    x = x * dx + min_x
    y = y * dy + min_y
    # z = z * dz + min_z

    return np.array([x, y]).T


def reshape_data(mesh, data):
    # reshape the data to the shape of the mesh
    nx, ny, nz = mesh.dimensions
    data = data.reshape(mesh.dimensions[0:2], order="F")
    return data


def reshape_points(mesh, points):
    # reshape the points to the shape of the mesh
    points = points.reshape(np.array(mesh.dimensions[0:2], 3), order="F")
    return points
