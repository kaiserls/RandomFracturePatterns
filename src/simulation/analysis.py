import os
import traceback

from PIL import Image
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


def analyze(postprocessing_results: list[dict]):
    postprocessing_result_df = pd.DataFrame(postprocessing_results)
    return analyze_from_dataframe(postprocessing_result_df)


def analyze_from_csv_file(postprocessing_result_file: str):
    postprocessing_result_df = pd.read_csv(postprocessing_result_file, index_col=0)
    return analyze_from_dataframe(postprocessing_result_df)


def analyze_from_dataframe(df: pd.DataFrame, **kwargs) -> list[dict]:
    """Analyze the simulation results. They should come from the postprocessing.

    Args:
        df (pd.DataFrame): The simulation results including derieved quantities from the postprocessing.

    Returns:
        list[dict]: The analysis results as a list of dictionaries.
    """
    # Define the analysis parameters
    analysis_parameters = {
        "iso_value": 0.9,
        "target_n_points": 201,
        "contour_thickness": 1,
    }
    analysis_parameters.update(kwargs)

    df = df.assign(**analysis_parameters)

    # Get the reference run. It is the one with homogeneous material properties.
    reference_run = df[df["homogeneous"] == True].to_dict(orient="records")
    assert len(reference_run) == 1, "There should be only one reference run."
    reference_run = reference_run[0]

    analysis_results = []
    # Analyze the reference run
    reference_analysis_results = analyze_run(reference_run)
    reference_analysis_results.update(analysis_parameters)
    analysis_results.append(reference_analysis_results)

    # Analyze all other runs
    for i, row in df.iterrows():
        if row["homogeneous"] == False:
            analysis_result = analyze_run(
                row, reference_data=reference_analysis_results
            )
            analysis_results.append(analysis_result)

    return analysis_results


def analyze_run(data: dict, reference_data=None, verbose=False):
    analysis_result = data.copy()
    # Define the possible analysis results. So even if the analysis fails, we dont get a key error.
    possible_analysis_result = {
        "fractal_dimension": None,
        "volume": None,
        "length": None,
        "width": None,
        "deviation": None,
        "isolines": None,
        "interpolated_isolines": None,
    }
    analysis_result.update(possible_analysis_result)

    img = isolines_image_cv2(
        mesh_file=data["vtk_structured"], iso_value=data["iso_value"], contour_thickness=data["contour_thickness"]
    )
    img_path = f"results/images/cv2_isolines_{data['run']}.png"
    np_img_path = f"results/images/cv2_isolines_as_np_array_{data['run']}.npy"
    analysis_result["cv2_isolines"] = img_path
    analysis_result["cv2_isolines_as_np_array"] = np_img_path
    Image.fromarray(img).save(img_path)
    np.save(np_img_path, img)

    fractal_dimension = fractal.fractal_dimension(Z=img[:, :, 1], threshold=0.9)
    analysis_result["fractal_dimension"] = fractal_dimension

    dA = fractal.pixel_area(
        min_x=data["structured_mesh_min_x"],
        max_x=data["structured_mesh_max_x"],
        n_discretization_x=data["structured_mesh_n_discretization_x"],
        min_y=data["structured_mesh_min_y"],
        max_y=data["structured_mesh_max_y"],
        n_discretization_y=data["structured_mesh_n_discretization_y"],
    )
    volume = fractal.crack_volume(Z=img[:, :, 1], dA=dA)
    analysis_result["volume"] = volume

    isolines = isolines_from_vtk(
        mesh_file=data["vtk_structured"], iso_value=data["iso_value"]
    )
    # Save the isolines as a dict of numpy arrays
    isolines_path = f"results/isolines/isolines_{data['run']}.npz"
    isolines_dict = {f"isoline_{i}": isoline for i, isoline in enumerate(isolines)}
    np.savez(isolines_path, **isolines_dict)
    analysis_result["isolines"] = isolines_path

    length = simple.crack_length(isolines)
    analysis_result["length"] = length

    max_deviation_from_middle = simple.max_deviation_from_middle(isolines)
    analysis_result["max_deviation_from_middle"] = max_deviation_from_middle

    try:
        interpolated_isolines = interpolate_isolines(
            isolines,
            target_n_points=data["target_n_points"],
            x_min=data["structured_mesh_min_x"],
            x_max=data["structured_mesh_max_x"],
        )
        interpolated_isolines_path = (
            f"results/isolines/interpolated_isolines_{data['run']}.npz"
        )
        interpolated_isolines_dict = {
            f"isoline_{i}": isoline for i, isoline in enumerate(interpolated_isolines)
        }
        np.savez(interpolated_isolines_path, **interpolated_isolines_dict)
        analysis_result["interpolated_isolines"] = interpolated_isolines_path

        width = np.mean(simple.crack_width(interpolated_isolines))
        analysis_result["width"] = width

        if reference_data is not None:
            reference_interpolated_isolines_container = np.load(
                reference_data["interpolated_isolines"]
            )
            reference_interpolated_isolines = [
                reference_interpolated_isolines_container[f"isoline_{i}"]
                for i in range(len(reference_interpolated_isolines_container.files))
            ]
            deviation = np.mean(
                simple.crack_deviation(
                    isolines=interpolated_isolines,
                    reference_isolines=reference_interpolated_isolines,
                )
            )
            analysis_result["deviation"] = deviation

    except Exception as e:
        if verbose:
            print(
                f"Error while measuring run {data['run']} with the following data:\n{data}"
            )
            print(f"Expect some analysis results to be None.")
            # print(f"Error: {e}")
            # print(''.join(traceback.TracebackException.from_exception(e).format()))
        else:
            pass

    return analysis_result
