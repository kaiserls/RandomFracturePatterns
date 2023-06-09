import gstools as gs
import numpy as np
import matplotlib.pyplot as plt

from gstools.random import MasterRNG

# https://geostat-framework.readthedocs.io/projects/gstools/en/v1.4.1/examples/01_random_field/01_srf_ensemble.html#sphx-glr-examples-01-random-field-01-srf-ensemble-py


def generate_field_data(
    min_X,
    max_X,
    min_Y,
    max_Y,
    discretization_X,
    discretization_Y,
    mean_mu,
    mean_lambda,
    variance_mu,
    variance_lambda,
    lengthscale_mu,
    lengthscale_lambda,
    run,
    path,
    number_of_fields=1,
    lognormal=True,
    plot=False,
    save_txt=True,
    save_vtk=False,
    coords=None,
    seeds_mu=None,
    seeds_lambda=None,
):
    """Generate the field data"""
    if lognormal:
        mean_mu_log = np.log(mean_mu**2 / np.sqrt(variance_mu + mean_mu**2))
        mean_lambda_log = np.log(
            mean_lambda**2 / np.sqrt(variance_lambda + mean_lambda**2)
        )
        variance_mu = np.log(1 + variance_mu / mean_mu**2)
        variance_lambda = np.log(1 + variance_lambda / mean_lambda**2)
        # Now we can overwrite the mean with the lognormal parameters
        # Before we needed the old mean to calculate the lognormal variance
        mean_mu = mean_mu_log
        mean_lambda = mean_lambda_log

    if variance_mu == 0.0:
        variance_mu = 1e-10
    if variance_lambda == 0.0:
        variance_lambda = 1e-10

    mesh_type = None
    if coords is None:
        mesh_type = "structured"
    else:
        mesh_type = "unstructured"

    if mesh_type == "unstructured":
        x = coords[:, 0]
        y = coords[:, 1]
    else:
        x = np.linspace(min_X, max_X, discretization_X)
        y = np.linspace(min_Y, max_Y, discretization_Y)

        X, Y = np.meshgrid(x, y)

    def gen_seeds(n):
        rng = MasterRNG()
        return [rng() for i in range(n)]

    if seeds_mu is None:
        seeds_mu = gen_seeds(number_of_fields)
    if seeds_lambda is None:
        seeds_lambda = gen_seeds(number_of_fields)

    mu_model = gs.Gaussian(dim=2, var=variance_mu, len_scale=lengthscale_mu)
    mu_srf = gs.SRF(mu_model, mean=mean_mu)
    mu_srf.set_pos([x, y], mesh_type=mesh_type)

    lambda_model = gs.Gaussian(dim=2, var=variance_lambda, len_scale=lengthscale_lambda)
    lambda_srf = gs.SRF(lambda_model, mean=mean_lambda)
    lambda_srf.set_pos([x, y], mesh_type=mesh_type)

    if save_txt or save_vtk:
        # Create a directory for the data
        fields_path = path / "fields"
        fields_path.mkdir(exist_ok=True, parents=True)

    for i in range(number_of_fields):
        lambda_srf(seed=seeds_lambda[i], store=f"field{i}")
        mu_srf(seed=seeds_mu[i], store=f"field{i}")

        if lognormal:
            mu_srf.transform("lognormal", field=f"field{i}")  # , process=True)
            lambda_srf.transform("lognormal", field=f"field{i}")  # , process=True)

        if plot:
            # Plot the field using the gs plotting function
            lambda_srf.plot(field=f"field{i}")
            print(
                f"App: {run}, Field: {i}, Mean: {np.mean(lambda_srf[i])}, Variance: {np.var(lambda_srf[i])}, lengthscale: {lengthscale_lambda}"
            )
            x = input("Press Enter to continue...")

        if save_vtk:
            lambda_srf.vtk_export(
                str(fields_path / f"lambda_field_run_{run}_sample_{i}"),
                field_select=f"field{i}",
            )
            mu_srf.vtk_export(
                str(fields_path / f"mu_field_run_{run}_sample_{i}"),
                field_select=f"field{i}",
            )

        if save_txt:
            mu_field = getattr(mu_srf, f"field{i}")
            lambda_field = getattr(lambda_srf, f"field{i}")
            # Save the fields into one txt file in the format
            n_points = 0
            if mesh_type == "structured":
                n_points = discretization_X * discretization_Y
            else:
                n_points = x.shape[0]
            header = f"STADATASETS {n_points}"
            np.savetxt(
                fields_path / f"field_run_{run}_sample_{i}.txt",
                np.column_stack(
                    (
                        X.flatten() if mesh_type == "structured" else x,
                        Y.flatten() if mesh_type == "structured" else y,
                        mu_field.flatten(),
                        lambda_field.flatten(),
                    )
                ),
                delimiter=" ",
                header=header,
                comments="",
            )

    # plotting
    # if plot:
    #     if mesh_type == "unstructured":
    #         raise NotImplementedError("Plotting random field for unstructured meshes is not implemented")

    #     if number_of_fields >= 4:
    #         nx = 2
    #         ny = int(np.ceil(number_of_fields / nx))
    #     else:
    #         nx = number_of_fields
    #         ny = 1

    #     fig, ax = plt.subplots(nx, ny, sharex=True, sharey=True)
    #     ax = ax.flatten()
    #     for i in range(number_of_fields):
    #         ax[i].imshow(lambda_srf[i].T, origin="lower")
    #         # add colorbar to imshow
    #         cbar = fig.colorbar(ax[i].images[0], ax=ax[i])
    #         cbar.ax.set_ylabel("lambda")
    #         ax[i].set_title(f"lambda field {i}")
    #     plt.show()
