import logging

from PIL import Image
import pandas as pd
import numpy as np
import cv2
from scipy import ndimage
import skimage.morphology as morphology

from src.measure import curvature, simple
import src.measure.fractal as fractal
import src.measure.isolines as isolines

def analyze(postprocessing_results: list[dict]):
    postprocessing_result_df = pd.DataFrame(postprocessing_results)
    return analyze_from_dataframe(postprocessing_result_df)


def analyze_from_csv_file(postprocessing_result_file: str):
    postprocessing_result_df = pd.read_csv(postprocessing_result_file, index_col=0)
    return analyze_from_dataframe(postprocessing_result_df)

def get_reference_run(df: pd.DataFrame) -> dict:
    """Get the reference run. It is the one with homogeneous material properties.

    Args:
        df (pd.DataFrame): The simulation results including derieved quantities from the postprocessing.

    Returns:
        dict: The reference run.
    """
    reference_run = df[df["homogeneous"] == True].to_dict(orient="records")
    assert len(reference_run) == 1, "There should be only one reference run."
    reference_run = reference_run[0]
    return reference_run

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
        # "contour_thickness": 1,
        "curvature_stride": 10,
    }
    analysis_parameters["threshold"] = analysis_parameters["iso_value"] * 255
    analysis_parameters.update(kwargs)
    df = df.assign(**analysis_parameters)


    analysis_results = []
    reference_run = get_reference_run(df)
    stochastic_runs = df[df["homogeneous"] == False].to_dict(orient="records")

    # Analyze the reference run
    logging.info(f"Analyzing reference run {reference_run['run']}")
    reference_analysis_results = analyze_run(reference_run)
    analysis_results.append(reference_analysis_results)

    # Analyze the stochastic runs
    for row in stochastic_runs:
        logging.info(f"Analyzing run {row['run']}")
        analysis_result = analyze_run(
            row, reference_data=reference_analysis_results
        )
        analysis_results.append(analysis_result)

    return analysis_results

def analyze_run(data: dict, reference_data=None, verbose=False):
    # analysis_result = data.copy()
    add_thresholded_image(data)
    analyze_isolines(data)
    analyze_skeleton(data)
    analyze_fractal(data)
    analyze_curvature(data)
    OP_01 = np.load(data["OP_01"])
    data["OP_01_count"] = simple.count(OP_01, threshold=data["iso_value"])
    data["OP_01_volume"] = simple.volume(OP_01, dA=data["structured_mesh_dA"], threshold=data["iso_value"])
    return data


def add_thresholded_image(data: dict):
    run = data["run"]

    # # Generate an black-white image of the crack with thresholding
    OP_01 = np.load(data["OP_01"])
    OP_01_bw = np.where(OP_01 > data["iso_value"], 1.0, 0.0)
    OP_01_bw_path = f"results/images/OP_01_bw_{run}.npy"
    np.save(OP_01_bw_path, OP_01_bw)
    data["OP_01_bw"] = OP_01_bw_path

    # Generate an black-white image of the crack with thresholding
    OP_0255 = np.load(data["OP_0255"])
    th, OP_0255_bw = cv2.threshold(OP_0255, data["threshold"], 255, cv2.THRESH_BINARY)

    # save the data as a numpy array npy and as an image
    OP_0255_bw_path = f"results/data/OP_0255_bw_{run}.npy"
    np.save(OP_0255_bw_path, OP_0255_bw)
    data["OP_0255_bw"] = OP_0255_bw_path
    OP_0255_bw_image_path = f"results/images/OP_0255_bw_{run}.png"
    Image.fromarray(OP_0255_bw).save(OP_0255_bw_image_path)
    data["OP_0255_bw_image"] = OP_0255_bw_image_path

def analyze_isolines(data: dict):
    OP_01 = np.load(data["OP_01"])
    OP_01_isolines = isolines.isoline(OP_01, iso_value=data["iso_value"])
    # Calculate the summed length of the isolines
    isoline_length = sum([simple.line_length(line) for line in OP_01_isolines])
    data["isoline_length"] = isoline_length

def analyze_skeleton(data: dict):
    OP_01 = np.load(data["OP_01_bw"])
    OP_binary = np.where(OP_01 > 0.5, 1.0, 0.0)

    skeleton = morphology.skeletonize(OP_binary)
    skeleton_path = f"results/data/skeleton_{data['run']}.npy"
    np.save(skeleton_path, skeleton)
    data["skeleton"] = skeleton_path
    skeleton_image_path = f"results/images/skeleton_{data['run']}.png"
    Image.fromarray(skeleton.astype(np.uint8) * 255).save(skeleton_image_path)
    data["skeleton_image"] = skeleton_image_path

    kernel = np.array([[2, 1, 2], [1, 0, 1], [2, 1, 2]])
    neighbour_count = ndimage.convolve(skeleton.astype(np.uint8), kernel, mode='constant', cval=0)
    straight = (neighbour_count == 2) & skeleton
    diagonal = (neighbour_count == 4) & skeleton
    half_diag = (neighbour_count == 3) & skeleton

    # Count the number of diagonal and straight pixels
    straight_length = np.sum(straight)
    diagonal_length = np.sum(diagonal) * np.sqrt(2)
    half_diag_length = np.sum(half_diag) * (1 + np.sqrt(2) / 2)

    # Add them together to get the total length
    skeleton_length = straight_length + diagonal_length + half_diag_length
    data["skeleton_length"] = skeleton_length


def analyze_fractal(data: dict):
    OP_01 = np.load(data["OP_01"])
    fractal_dimension = fractal.fractal_dimension(OP_01, threshold=data["iso_value"])
    data["fractal_dimension"] = fractal_dimension

def analyze_curvature(data: dict):
    OP_01 = np.load(data["OP_01"])
    OP_01_isolines = isolines.isoline(OP_01, iso_value=data["iso_value"])
    curvatures = []
    for isoline in OP_01_isolines:
        if len(isoline) <= data["curvature_stride"]:
            continue
        local_curvature = curvature.calc_local_curvature(contour=isoline, stride=data["curvature_stride"])
        curvature_value = np.sum(local_curvature)
        curvatures.append(curvature_value)
    total_curvature = sum(curvatures)
    data["curvature"] = total_curvature

    # Smooth the curvature 