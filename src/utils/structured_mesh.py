import numpy as np
import pyvista as pv

def to_structured_pv(
    input_vtk,
    output_vtk,
    structured_mesh_min_x,
    structured_mesh_max_x,
    structured_mesh_min_y,
    structured_mesh_max_y,
    structured_mesh_n_x,
    structured_mesh_n_y,
    field_name: str = "OP",
    plot=False,
) -> pv.StructuredGrid:
    """Use pyvista to sample the unstructured grid data with the given field name to a structured grid and save it to the given output path as vtk"""
    ugrid = pv.UnstructuredGrid(input_vtk)

    x = np.linspace(
        structured_mesh_min_x,
        structured_mesh_max_x,
        structured_mesh_n_x,
        dtype=np.float32,
    )
    y = np.linspace(
        structured_mesh_min_y,
        structured_mesh_max_y,
        structured_mesh_n_y,
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
    
    return stgrid

# TODO: pixel indices and pixel coordinates are not the same thing!!!!
def scale_pixels_to_coordinates(mesh: pv.StructuredGrid, pixel_indices: np.ndarray) -> np.ndarray:
    """Scale some pixel indices from an image to the coordinates of the mesh representing the domain of the image.

    Args:
        mesh (pv.StructuredGrid): A structured mesh representing the domain of the image.
        pixel_coordinates (np.ndarray): Some pixel indices as int pairs (i, j), where i is the row and j is the column.???

    Returns:
        np.ndarray: The physical coordinates of the pixel as float pairs (x, y).
    """
    min_x, max_x, min_y, max_y, min_z, max_z = mesh.bounds
    dx, dy = cell_lengths_from_mesh(mesh)

    # scale the pixel coordinates to the coordinates of the mesh
    i = pixel_indices[:, 0]
    j = pixel_indices[:, 1]
    x = i * dx + min_x
    y = j * dy + min_y

    return np.array([x, y]).T

def discretization_size(x_min, x_max, n_points) -> float:
    """Calculate the discretization size of the domain."""
    return (x_max - x_min) / (n_points - 1) # The mesh has n_points - 1 segments

def cell_lengths_from_mesh(mesh: pv.StructuredGrid) -> tuple[float, float]:
    """Calculate the length of a single pixel in the mesh domain."""
    min_x, max_x, min_y, max_y, min_z, max_z = mesh.bounds
    dx = discretization_size(min_x, max_x, mesh.dimensions[0])
    dy = discretization_size(min_y, max_y, mesh.dimensions[1])
    return dx, dy

def cell_area_from_mesh(mesh: pv.StructuredGrid) -> float:
    """Calculate the area of a single pixel in the mesh domain."""
    dx, dy = cell_lengths_from_mesh(mesh)
    return dx * dy

def reshape_data(mesh: pv.StructuredGrid, data: np.ndarray) -> np.ndarray:
    """reshape the data to the shape of the mesh"""
    nx, ny, nz = mesh.dimensions
    data = data.reshape(mesh.dimensions[0:2], order="F")
    return data


def reshape_points(mesh: pv.StructuredGrid, points: np.ndarray) -> np.ndarray:
    """reshape the points to the shape of the mesh"""
    points = points.reshape(np.array(mesh.dimensions[0:2], 3), order="F")
    return points

def field_as_2d_array_from_mesh(mesh: pv.StructuredGrid, field_name: str) -> np.ndarray:
    """Get the data from the mesh as a 2D array"""
    return reshape_data(mesh, mesh[field_name])

def mesh_and_2d_data_from_vtk(mesh_file, field_name: str) -> np.ndarray:
    mesh = pv.read(mesh_file)
    data = reshape_data(mesh, mesh["OP"])
    return mesh, data