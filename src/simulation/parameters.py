from src.utils.path_helpers import get_pandas_app_path
from gstools.random import MasterRNG


def create_default_parameters():
    app_path = get_pandas_app_path()
    parameters = {}

    parameters["app_path"] = app_path

    parameters["mean_mu"] = 8.077e7
    parameters["std_mu"] = 0.0e0
    parameters["lengthscale_mu"] = 5.0e1
    parameters["variogram_type_mu"] = "gaussian"

    parameters["mean_lambda"] = 5.05e8
    parameters["std_lambda"] = 1.65e8
    parameters["lengthscale_lambda"] = 5.0e1
    parameters["variogram_type_lambda"] = "gaussian"

    parameters["log_distribution"] = True
    parameters["sample"] = 0

    parameters["grid_file"] = (app_path / "gauss_nodes_stat_grob.txt").as_posix()
    parameters["random_fields_on_unstructured_grid"] = True

    parameters["x_min"] = 0.0
    parameters["x_max"] = 1000.0
    parameters["y_min"] = -1000.0
    parameters["y_max"] = 1000.0

    parameters["homogeneous"] = False
    parameters["seed_mu"] = 0
    parameters["seed_lambda"] = 0

    return parameters


def define_scenarios(default_parameters: dict):
    rng = MasterRNG(20170519)

    ref_std_lambda = default_parameters["mean_lambda"]
    stds_lambda = [ref_std_lambda * 0.25, ref_std_lambda * 0.5, ref_std_lambda]
    lengthscales_lambda = [1.0, 5.0, 15.0, 65.0, 250.0, 1000.0]
    samples = [0, 1, 2, 3]

    scenarios = []
    run = 0
    for std_lambda in stds_lambda:
        for lengthscale_lambda in lengthscales_lambda:
            for sample in samples:
                param_entry = default_parameters.copy()
                param_entry["run"] = run
                param_entry["std_lambda"] = std_lambda
                param_entry["lengthscale_lambda"] = lengthscale_lambda
                param_entry["sample"] = sample
                param_entry["seed_mu"] = rng()
                param_entry["seed_lambda"] = rng()
                scenarios.append(param_entry)
                run += 1

    homogeneous_param_entry = default_parameters.copy()
    homogeneous_param_entry["homogeneous"] = True
    homogeneous_param_entry["run"] = run
    scenarios.append(homogeneous_param_entry)

    return scenarios
