import numpy as np
import jinja2
import subprocess

from src.utils.path_helpers import get_pandas_app_path

import src.utils.random_field_generator as random_field_generator


def create_cmd(parameters: dict):
    # Use jinja2 to create the cmd file from the template
    app_path = get_pandas_app_path()
    environment = jinja2.Environment(
        loader=jinja2.FileSystemLoader(app_path),
    )

    if parameters["homogeneous"]:
        template_name = "cmd_homogen"
    elif parameters["random_fields_on_unstructured_grid"]:
        template_name = "cmd_direct"
    else:
        template_name = "cmd"

    template = environment.get_template(template_name)
    cmd = template.render(
        run=parameters["run"],
    )
    return cmd


def save_cmd(parameters, cmd):
    app_path = get_pandas_app_path()
    cmd_name = f"cmd_{parameters['run']}"
    path_to_cmd = app_path / "cmds" / cmd_name
    with open(path_to_cmd, "w") as f:
        f.write(cmd)
    return path_to_cmd.as_posix()


def random_field(parameters):
    app_path = get_pandas_app_path()
    coords = np.loadtxt(parameters["grid_file"], delimiter=" ", ndmin=2)
    random_field_generator.generate_field_data(
        min_X=parameters["x_min"],
        max_X=parameters["x_max"],
        min_Y=parameters["y_min"],
        max_Y=parameters["y_max"],
        discretization_X=120,
        discretization_Y=120,
        mean_mu=parameters["mean_mu"],
        mean_lambda=parameters["mean_lambda"],
        variance_mu=parameters["std_mu"] ** 2,
        variance_lambda=parameters["std_lambda"] ** 2,
        lengthscale_mu=parameters["lengthscale_mu"],
        lengthscale_lambda=parameters["lengthscale_lambda"],
        run=parameters["run"],
        path=app_path,
        number_of_fields=1,
        logNormal=parameters["log_distribution"],
        plot=False,
        save_txt=True,
        save_vtk=True,
        coords=coords if parameters["random_fields_on_unstructured_grid"] else None,
        seeds_mu=[parameters["seed_mu"]],
        seeds_lambda=[parameters["seed_lambda"]],
    )
    field_path = app_path / "fields" / f"field_run_{parameters['run']}_sample_{0}.txt"
    return field_path.as_posix()


def simulate(parameters, fake_run=False):
    results = parameters
    cmd = create_cmd(parameters)
    path_to_cmd = save_cmd(parameters, cmd)
    results["cmd"] = path_to_cmd

    if not parameters["homogeneous"]:
        field_path = random_field(parameters)
        results["field"] = field_path
    else:
        results["field"] = None

    if not fake_run:
        command = f"./pandas.bin --ui=plain < {path_to_cmd} >/dev/null 2>&1"
        process = subprocess.Popen(command, shell=True, cwd=parameters["app_path"])
        process.communicate()
        return_code = process.returncode
    else:
        return_code = 0

    if return_code == 0:
        print(f"Simulation {parameters['run']} finished successfully")
    else:
        print(
            f"Simulation {parameters['run']} failed with return code {return_code}. It had the parameters {parameters} and the results {results}"
        )

    results[
        "tec"
    ] = f"{parameters['app_path']}/tec/tecplot_final_run_{parameters['run']}.dat"
    results["return_code"] = return_code

    return results
