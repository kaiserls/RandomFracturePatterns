import numpy as np
import meshio

"""
*Note: meshio as well as newer vtk version dont understand the data written by PANDAS. The format is the tecplot format in ASCII with the file ending ```.dat``` .*
This tool provides a function ```clean_mesh``` and executable code to remove the gauss nodes which are not contained in the mesh and can not be read by most programs. The executable code takes my first example file and saves it as clean ```.vtk``` and ```.dat``` file. 
"""

def clean_tec_file(file_in: str, file_out: str):
    """Delete only the first empty line appearing. This is needed so that the pandas tecplot file can be read by meshio."""
    with open(file_in, "r") as f:
        lines = f.readlines()
    with open(file_out, "w") as f:
        index_of_empty_line = lines.index("\n")
        lines.pop(index_of_empty_line)
        f.writelines(lines)

def clean_mesh(mesh: meshio.Mesh) -> meshio.Mesh:
    """Remove hanging gauss nodes from meshio mesh object and return a new meshio mesh object"""
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
