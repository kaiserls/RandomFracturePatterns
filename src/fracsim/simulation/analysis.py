import logging

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
from scipy import ndimage
import scipy
import skimage.morphology as morphology

from fracsim.measure import curvature, simple
import fracsim.measure.fractal as fractal
import fracsim.measure.isolines as isolines


def analyze(postprocessing_results: list[dict], **kwargs) -> list[dict]:
    """Analyze the simulation results. They should come from the postprocessing.

    Args:
        postprocessing_results (list[dict]): The simulation results including derieved quantities from the postprocessing.

    Returns:
        list[dict]: The analysi s results as a list of dictionaries.
    """

    # Define the analysis parameters
    analysis_parameters = {
        "iso_value": 0.98,
        # "contour_thickness": 1,
        "curvature_smoothing_cycles": 4,
        "curvature_smoothing_kernel_size": 3,
        "curvature_smoothing_kernel_increase_size_factor": 3,
        "curvature_smoothing_kernel_type": "ones",
        "curvature_smoothing_stride": 2,
        # "curvature_smoothing_kernel_sigma": 10,
    }
    analysis_parameters.update(kwargs)
    analysis_parameters["threshold"] = analysis_parameters["iso_value"] * 255

    analysis_results = []
    # Analyze the stochastic runs
    for postprocessing_result in postprocessing_results:
        logging.info(f"Analyzing run {postprocessing_result['run']}")
        postprocessing_result.update(analysis_parameters)
        analysis_result = analyze_run(postprocessing_result)
        analysis_results.append(analysis_result)

    return analysis_results


def analyze_run(postprocessing_result: dict) -> dict:
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
    OP_01 = np.load(data["OP_01"])
    OP_01_isolines = isolines.isoline(OP_01, iso_value=data["iso_value"])
    # Calculate the summed length of the isolines
    isoline_length = sum([simple.line_length(line) for line in OP_01_isolines])
    data["isoline_length"] = isoline_length
    data["isoline_n_points"] = sum([len(line) for line in OP_01_isolines])
    data["isoline_mean_segment_length"] = isoline_length / data["isoline_n_points"]


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
    neighbour_count = ndimage.convolve(
        skeleton.astype(np.uint8), kernel, mode="constant", cval=0
    )
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

    # Add the maximum y coordinate of a pixel of the skeleton
    data["skeleton_y_max"] = np.max(np.where(skeleton)[0])
    # TODO 0 or 1 as index?


def analyze_fractal(data: dict):
    OP_01 = np.load(data["OP_01"])
    fractal_dimension = fractal.fractal_dimension(OP_01, threshold=data["iso_value"])
    data["fractal_dimension"] = fractal_dimension


def analyze_curvature(data: dict):
    OP_01 = np.load(data["OP_01"])
    OP_01_isolines = isolines.isoline(OP_01, iso_value=data["iso_value"])
    curvatures_pointwise = []
    for isoline in OP_01_isolines:
        curvature_pointwise = curvature.calc_local_curvature(isoline)
        curvatures_pointwise.append(curvature_pointwise)

    # Total curvature
    total_curvature = sum(
        np.sum(np.abs(curvature_pointwise))
        for curvature_pointwise in curvatures_pointwise
    )
    data["total_curvature"] = total_curvature
    # Mean curvature
    n_isolines = len(OP_01_isolines)
    mean_curvature = (
        sum(
            np.mean(curvature_pointwise) for curvature_pointwise in curvatures_pointwise
        )
        / n_isolines
    )
    data["mean_curvature"] = mean_curvature
    # Max curvature
    max_curvature = max(
        np.max(np.abs(curvature_pointwise))
        for curvature_pointwise in curvatures_pointwise
    )
    data["max_curvature"] = max_curvature
    # Number of sign changes in the curvature
    sign_changes = sum(
        np.sum(np.diff(np.sign(curvature_pointwise)) != 0)
        for curvature_pointwise in curvatures_pointwise
    )
    data["sign_changes"] = sign_changes

    # Curvatures smoothed by stride > 1:
    smoothed_curvatures = []
    for isoline in OP_01_isolines:
        if len(isoline) < data["curvature_smoothing_stride"]:
            print(len(isoline))
            continue
        smoothed_curvature = curvature.calc_local_curvature(
            isoline, stride=data["curvature_smoothing_stride"]
        )
        smoothed_curvatures.append(smoothed_curvature)
    max_curvature_smoothed = max(
        np.max(smoothed_curvature) for smoothed_curvature in smoothed_curvatures
    )
    data["max_curvature_smoothed"] = max_curvature_smoothed
    sign_changes_smoothed = sum(
        np.sum(np.diff(np.sign(smoothed_curvature)) != 0)
        for smoothed_curvature in smoothed_curvatures
    )
    data["sign_changes_smoothed"] = sign_changes_smoothed

    def smoothing_kernel(kernel_size, kernel_type="ones", kernel_sigma=1.0):
        if kernel_type == "ones":
            kernel = np.ones(kernel_size) / kernel_size
        elif data["curvature_smoothing_kernel_type"] == "gaussian":
            kernel = scipy.signal.gaussian(kernel_size, kernel_sigma)
        else:
            raise ValueError(
                f"Unknown kernel type {data['curvature_smoothing_kernel_type']}"
            )
        return kernel

    # plt.figure()
    kernel_size = data["curvature_smoothing_kernel_size"]
    for smoothing_cyle in range(1, data["curvature_smoothing_cycles"] + 1):
        kernel = smoothing_kernel(kernel_size)
        total_smoothed_curvature = 0
        # for i, c in enumerate(smoothed_curvatures):
        for i, c in enumerate(OP_01_isolines):
            smoothed_x = np.convolve(c[:, 0], kernel, mode="same")
            smoothed_y = np.convolve(c[:, 1], kernel, mode="same")
            smoothed_isoline = np.stack([smoothed_x, smoothed_y], axis=1)
            smoothed_curvature = curvature.calc_local_curvature_old(smoothed_isoline)
            total_smoothed_curvature += np.sum(np.abs(smoothed_curvature))
            # if i==0: # Only select the first isoline for plotting. Else it gets too crowded
            #     plt.plot(c, label=f"Smoothed curvature cycle {smoothing_cyle}")
        data[f"curvature_smoothed_{smoothing_cyle}"] = total_smoothed_curvature

        kernel_size = (
            kernel_size * data["curvature_smoothing_kernel_increase_size_factor"]
        )

    # Plot the total smoothed curvatures over the number of smoothing cycles together with the original curvature
    curvatures = [data["total_curvature"]] + [
        data[f"curvature_smoothed_{i}"]
        for i in range(1, data["curvature_smoothing_cycles"] + 1)
    ]
    np.save(f"results/data/curvatures_{data['run']}.npy", curvatures)
    data["curvatures"] = f"results/data/curvatures_{data['run']}.npy"

    # Estimate the convergence of the smoothed curvature with a polyfit and extracting the slope. use log log scale
    x = np.arange(1, data["curvature_smoothing_cycles"] + 2)
    y = curvatures
    p = np.polyfit(np.log(x), np.log(y), 1)
    data["curvature_smoothing_convergence"] = -p[0]
