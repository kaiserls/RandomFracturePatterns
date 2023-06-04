import os

import pandas as pd
import numpy as np

import src.measure.fractal as fractal
import src.measure.simple as simple

from src.measure.fractal import fractal_dimension, crack_volume, pixel_area
from src.measure.simple import crack_width, crack_deviation, crack_length
from src.utils.isolines import (
    isolines_from_vtk,
    interpolate_isolines,
    isolines_image_cv2,
)

def analyze(parameters: list[dict], results: list[dict], postprocessing_results: list[dict]):
    # Create a pandas dataframe from the results and the parameters together
    parameter_df = pd.DataFrame(parameters)
    result_df = pd.DataFrame(results)
    postprocessing_result_df = pd.DataFrame(postprocessing_results)
    return analyze_from_dataframe(parameter_df, result_df, postprocessing_result_df)

def analyze_from_csv_file(parameter_file: str, result_file: str, postprocessing_result_file: str):
    # Create a pandas dataframe from the results and the parameters together
    parameter_df = pd.read_csv(parameter_file, index_col=0)
    result_df = pd.read_csv(result_file, index_col=0)
    postprocessing_result_df = pd.read_csv(postprocessing_result_file, index_col=0)
    return analyze_from_dataframe(parameter_df, result_df, postprocessing_result_df)

def analyze_from_dataframe(parameter_df: pd.DataFrame, result_df: pd.DataFrame, postprocessing_result_df: pd.DataFrame):
    analysis_results = []

    # Create a pandas dataframe from the results and the parameters together
    df = pd.concat([parameter_df, result_df, postprocessing_result_df], axis=1)

    os.mkdir("isolines", exist_ok=True)

    # Define analysis parameters
    analysis_parameters = {
        "iso_value": 0.9,
    }

    # Get the reference run. It is the one with homogeneous material properties.
    reference_run = df[df['homogeneous'] == True].iloc[0]
    reference_run_idx = reference_run.run
    print(f"Reference run is {reference_run_idx}")
    # Analyze the reference run
    reference_analysis_results = analyze_run(reference_run)
    reference_analysis_results.update(analysis_parameters)
    analysis_results.append(reference_analysis_results)

    # Analyze all other runs
    for i, row in df.iterrows():
        analysis_result = analyze_run(row, reference_data=reference_analysis_results)
        analysis_result.update(analysis_parameters)
        analysis_results.append(analysis_result)

    return analysis_results


def analyze_run(data: dict, reference_data=None):
    analysis_results = {
        "run": data["run"],
        "fractal_dimension": None,
        "volume": None,
        "length": None,
        "width": None,
        "deviation": None,
        "isolines": None,
        "interpolated_isolines": None,
    }

    img = isolines_image_cv2(mesh_file=data["vtk_structured"], iso_value=data["iso_value"])
    fractal_dimension = fractal.fractal_dimension(img[:, :, 1])
    analysis_results["fractal_dimension"] = fractal_dimension

    dA = fractal.pixel_area(min_x=data["structured_mesh_min_x"], max_x=data["structured_mesh_max_x"], n_discretization_x=data["structured_mesh_n_discretization_x"], min_y=data["structured_mesh_min_y"], max_y=data["structured_mesh_max_y"], n_discretization_y=data["structured_mesh_n_discretization_y"])
    volume = fractal.crack_volume(Z=img[:, :, 1], dA=dA)
    analysis_results["volume"] = volume

    isolines = isolines_from_vtk(mesh_file=data["vtk_structured"], iso_value=data["iso_value"])
    # Save the isolines as list of numpy arrays to a file
    isolines_path = f"isolines/isolines_{data['run']}.npy"
    np.save(isolines_path, isolines)
    analysis_results["isolines"] = isolines_path

    length = simple.crack_length(isolines)
    analysis_results["length"] = length

    try:
        interpolated_isolines = interpolate_isolines(isolines)
        interpolated_isolines_path = f"isolines/interpolated_isolines_{data['run']}.npy"
        np.save(interpolated_isolines_path, interpolated_isolines)
        analysis_results["interpolated_isolines"] = interpolated_isolines_path

        width = np.mean(simple.crack_width(isolines))
        analysis_results["width"] = width

        if reference_data is not None:
            reference_isolines = np.load(reference_data["interpolated_isolines"])
            deviation = np.mean(simple.crack_deviation(isolines=isolines, reference_isolines=reference_isolines))
            analysis_results["deviation"] = deviation
    except Exception as e:
        print(f"Error while measuring run {data['run']} with the following data: {data}")
        print(f"Error: {e}")
    
    return analysis_results