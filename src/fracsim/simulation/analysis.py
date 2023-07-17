import logging

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
from scipy import ndimage
import scipy
import skimage.morphology as morphology

from fracsim.measure import fractal, simple, curvature, isolines


def analyze(postprocessing_results: list[dict], **kwargs) -> list[dict]:
    """Analyze the simulation results. They should come from the postprocessing.

    Args:
        postprocessing_results (list[dict]): The simulation results including derieved quantities from the postprocessing.

    Returns:
        list[dict]: The analysi s results as a list of dictionaries.
    """

    # Define the (default) analysis parameters
    analysis_parameters = {
        "iso_value": 0.98,
        "contour_thickness": 2,
        "curvature_smoothing_stride": 2,
        "gaussian_smoothing_sigma": 2.0,
    }
    analysis_parameters.update(kwargs)
    analysis_parameters["threshold"] = analysis_parameters["iso_value"] * 255

    analysis_results = []
    for postprocessing_result in postprocessing_results:
        logging.info(f"Analyzing run {postprocessing_result['run']}")
        postprocessing_result.update(analysis_parameters)
        analysis_result = analyze_run(postprocessing_result)
        analysis_results.append(analysis_result)

    return analysis_results


def analyze_run(postprocessing_result: dict) -> dict:
    """Analyze a single run.
    
    Args:
        postprocessing_result (dict): The postprocessing results for a single run.

    Returns:
        dict: The analysis results for a single run.
    """

    data = postprocessing_result
    add_thresholded_image(data)
    analyze_isolines(data)
    analyze_skeleton(data)
    analyze_fractal(data)
    analyze_curvature(data)

    OP_01 = np.load(data["OP_01"])
    data["count"] = simple.count(OP_01, threshold=data["iso_value"])
    data["volume"] = simple.volume(
        OP_01, dA=data["structured_mesh_dA"], threshold=data["iso_value"]
    )

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
    OP = np.load(data["OP_01"])
    OP_isolines = isolines.isoline(OP, iso_value=data["iso_value"])

    data["isoline_length"] = sum([simple.line_length(line) for line in OP_isolines])
    data["isoline_n_points"] = sum([len(line) for line in OP_isolines])
    data["isoline_mean_segment_length"] = data["isoline_length"] / data["isoline_n_points"]
    data["isoline_y_max"] = max(
        [
            np.max(np.abs(line[:, 0] + data["structured_mesh_min_y"]))
            for line in OP_isolines
        ]
    )

def edge_lengths(image: np.ndarray):
    """Calculate the length of the edges in a binary image."""

    kernel = np.array([[2, 1, 2], [1, 0, 1], [2, 1, 2]])
    neighbour_count = ndimage.convolve(
        image.astype(np.uint8), kernel, mode="constant", cval=0
    )
    straight = (neighbour_count == 2) & image
    diagonal = (neighbour_count == 4) & image
    half_diag = (neighbour_count == 3) & image

    # Count the number of diagonal and straight pixels
    straight_length = np.sum(straight)
    diagonal_length = np.sum(diagonal) * np.sqrt(2)
    half_diag_length = np.sum(half_diag) * (1 + np.sqrt(2) / 2)

    # Add them together to get the total length
    edge_length = straight_length + diagonal_length + half_diag_length
    return edge_length


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

    # Add them together to get the total length
    data["skeleton_length"] = edge_lengths(skeleton)

    # Add the maximum y coordinate of a pixel of the skeleton. The offset is the minimum y of the structured analysis mesh
    data["skeleton_y_max"] = np.max(
        np.abs(np.where(skeleton)[0] + data["structured_mesh_min_y"])
    )


def analyze_fractal(data: dict):
    OP_01 = np.load(data["OP_01"])
    contours = isolines.isoline(OP_01, iso_value=data["iso_value"])

    blank_image = np.zeros(OP_01.shape, dtype=np.uint8)
    # Draw the contours onto the blank image array
    for n, contour in enumerate(contours):
        contour = contour.astype(int)
        # switch x and y coordinates
        contour = np.flip(contour, axis=1)
        cv2.polylines(blank_image, [contour], isClosed=False, color=255, thickness=2)

    fractal_dimension = fractal.fractal_dimension(
        blank_image, threshold=data["iso_value"], plot=False
    )
    data["fractal_dimension"] = fractal_dimension


def analyze_curvature(data: dict):
    OP = np.load(data["OP_01"])
    OP_isolines = isolines.isoline(OP, iso_value=data["iso_value"])
    # filter out isolines that are shorted than 2*smoothing_stride + 1 points
    OP_isolines = [isoline for isoline in OP_isolines if len(isoline) > 2*data["curvature_smoothing_stride"] + 1]
    # filter out isolines that are shorter than 300 mm, we only want the two main ones
    OP_isolines = [isoline for isoline in OP_isolines if simple.line_length(isoline) > 300]
    n_isolines = len(OP_isolines)

    curvatures = [curvature.calc_local_curvature(curve=isoline, stride = data["curvature_smoothing_stride"]) for isoline in OP_isolines]
    smoothed_curvatures = [scipy.ndimage.gaussian_filter1d(curvature, data["gaussian_smoothing_sigma"]) for curvature in curvatures]
    # Set curvature to zero if it is smaller than the threshold: 0.0005
    curvatures = [np.where(np.abs(curvature) < 0.0005, 0, curvature) for curvature in curvatures]
    smoothed_curvatures = [np.where(np.abs(curvature) < 0.0005, 0, curvature) for curvature in smoothed_curvatures]


    data["total_curvature"] = sum(np.sum(np.abs(curvature)) for curvature in curvatures)
    data["total_curvature_smoothed"] = sum(np.sum(np.abs(curvature)) for curvature in smoothed_curvatures)
    
    data["mean_curvature"] = sum(np.mean(np.abs(curvature)) for curvature in curvatures) / n_isolines
    data["mean_curvature_smoothed"] = sum(np.mean(np.abs(curvature)) for curvature in smoothed_curvatures) / n_isolines

    data["max_curvature"] = max(np.max(np.abs(curvature)) for curvature in curvatures)
    data["max_curvature_smoothed"] = max(np.max(curvature) for curvature in smoothed_curvatures)

    data["sign_changes"] = sum(np.sum(np.diff(np.sign(curvature)) != 0) for curvature in curvatures)
    data["sign_changes_smoothed"] = sum(np.sum(np.diff(np.sign(curvature)) != 0) for curvature in smoothed_curvatures)
