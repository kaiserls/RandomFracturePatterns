import pandas as pd
from src.utils.tec import tec_to_vtk, clean_tec_file
from src.utils.structured_mesh import to_structured_pv, cell_area_from_mesh, cell_lengths_from_mesh

STRUCTURED_MESH = {
    # Take the full length of the domain where the crack is located
    "structured_mesh_min_x": 0,
    "structured_mesh_max_x": 1000,
    "structured_mesh_n_discretization_x": 1000,
    # Take only the height of the domain where the crack is located (i.e. the crack is located in the middle of the domain)
    "structured_mesh_min_y": -250,
    "structured_mesh_max_y": 250,
    "structured_mesh_n_discretization_y": 500,
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
    run = postprocessing_result["run"]

    clean_tec_file(tec_path.format(run), cleaned_tec_path.format(run))
    tec_to_vtk(cleaned_tec_path.format(run), vtk_path.format(run))
    stgrid = to_structured_pv(
        vtk_path.format(run), vtk_structured_path.format(run), **STRUCTURED_MESH
    )
    structured_dx, structured_dy = cell_lengths_from_mesh(stgrid)
    structured_dA = cell_area_from_mesh(stgrid)

    postprocessing_result.update(STRUCTURED_MESH)
    postprocessing_result["cleaned_tec"] = cleaned_tec_path.format(run)
    postprocessing_result["vtk"] = vtk_path.format(run)
    postprocessing_result["vtk_structured"] = vtk_structured_path.format(run)
    postprocessing_result["structured_dx"] = structured_dx
    postprocessing_result["structured_dy"] = structured_dy
    postprocessing_result["structured_dA"] = structured_dA

    return postprocessing_result
