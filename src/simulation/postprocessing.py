import numpy as np
from src.utils.tec import tec_to_vtk, clean_tec_file
from src.utils.structured_mesh import field_as_2d_array_from_mesh, to_structured_pv, cell_area_from_mesh, cell_lengths_from_mesh
from src.utils.image_type import transform_01_to_0255

STRUCTURED_MESH = {
    # Take the full length of the domain where the crack is located
    "structured_mesh_min_x": 0,
    "structured_mesh_max_x": 1000,
    "structured_mesh_n_x": 1000,
    # Take only the height of the domain where the crack is located (i.e. the crack is located in the middle of the domain)
    "structured_mesh_min_y": -250,
    "structured_mesh_max_y": 250,
    "structured_mesh_n_y": 500,
}

# Define the paths to the tecplot and vtk files.
# TODO: The tecplot files are not yet created. They will be created by the simulation. The path should be in the results dict.
tec_path = "PANDAS_coop/src/app/apps/2d_fracking_saturated/tec/tecplot_final_run_{}.dat"
cleaned_tec_path = "PANDAS_coop/src/app/apps/2d_fracking_saturated/tec_cleaned/tecplot_final_run_{}_cleaned.dat"
vtk_path = "PANDAS_coop/src/app/apps/2d_fracking_saturated/vtk/vtk_final_run_{}.vtk"
vtk_structured_path = (
    "PANDAS_coop/src/app/apps/2d_fracking_saturated/vtk_structured/vtk_final_run_{}.vtk"
)


def postprocess(simulation_results: list[dict]):
    postprocessing_results = []
    for simulation_result in simulation_results:
        postprocessing_result = postprocess_run(simulation_result)
        postprocessing_results.append(postprocessing_result)
    return postprocessing_results


def postprocess_run(simulation_result: dict):
    postprocessing_result = simulation_result.copy()
    postprocessing_result.update(STRUCTURED_MESH)
    run = postprocessing_result["run"]

    # Tec to vtk
    clean_tec_file(tec_path.format(run), cleaned_tec_path.format(run))
    postprocessing_result["cleaned_tec"] = cleaned_tec_path.format(run)
    tec_to_vtk(cleaned_tec_path.format(run), vtk_path.format(run))
    postprocessing_result["vtk"] = vtk_path.format(run)

    # unstructured to structured vtk and taking part of the domain according to the STRUCTURED_MESH dict
    structured_grid = to_structured_pv(
        vtk_path.format(run), vtk_structured_path.format(run), **STRUCTURED_MESH
    )
    postprocessing_result["vtk_structured"] = vtk_structured_path.format(run)

    # Structured grid cell lengths and areas
    structured_dx, structured_dy = cell_lengths_from_mesh(structured_grid)
    structured_dA = cell_area_from_mesh(structured_grid)
    postprocessing_result["structured_mesh_dx"] = structured_dx
    postprocessing_result["structured_mesh_dy"] = structured_dy
    postprocessing_result["structured_mesh_dA"] = structured_dA

    # Pure data as numpy arrays
    OP_01 = field_as_2d_array_from_mesh(structured_grid, "OP")
    OP_01_path = "results/data/OP_01.npy"
    np.save(OP_01_path, OP_01)
    postprocessing_result["OP_01"] = OP_01_path

    OP_255 = transform_01_to_0255(OP_01)
    OP_255_path = "results/data/OP_255.npy"
    np.save(OP_255_path, OP_255)
    postprocessing_result["OP_255"] = OP_255_path

    return postprocessing_result
