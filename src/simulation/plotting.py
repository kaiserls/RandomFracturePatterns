import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot(df: pd.DataFrame, create_subplots: bool = False):
    """Visualize the results"""
    # reference_run
    reference_run_df = df[df["homogeneous"] == True]
    assert len(reference_run_df) == 1, "There should be only one reference run."
    # other
    other_runs = df[df["homogeneous"] == False]
    # drop outliers where the length is lower than 1000. This indicates crashes, because the simulation domain is 1000 long.
    other_runs = other_runs[other_runs["isoline_length"] > 999]

    plots = ["fractal_dimension", "isoline_length", "skeleton_length", "curvature"]
    n_row_plots = len(plots)

    std_lambdas = other_runs["std_lambda"].unique()
    n_std_lambdas = len(std_lambdas)
    std_lambda_colors = plt.cm.rainbow(np.linspace(0, 1, n_std_lambdas))
    n_col_plots = n_std_lambdas if create_subplots else 1

    fig, axs = plt.subplots(
        n_row_plots, n_col_plots, figsize=(10, 20), tight_layout=True
    )
    if n_col_plots == 1:
        axs = np.array([axs]).T
    for i, plot in enumerate(plots):
        max_val, min_val = np.max(other_runs[plot]), np.min(other_runs[plot])
        hom_val = reference_run_df[plot].values[0]
        max_val = max(max_val, hom_val)*1.1
        min_val = min(min_val, hom_val)*0.9
        for j, std_lambda in enumerate(std_lambdas):
            if n_col_plots == 1:
                ax = axs[i, 0]
            else:
                ax = axs[i, j]

            lengthscales_lambda_all = other_runs[other_runs["std_lambda"] == std_lambda]["lengthscale_lambda"]
            lengthscales_lambda = lengthscales_lambda_all.unique()

            ax.scatter(
                lengthscales_lambda_all,
                other_runs[other_runs["std_lambda"] == std_lambda][plot],
                label=f"std_lambda={std_lambda:.2E}",
                color=std_lambda_colors[j],
            )

            # Draw a horizontal line at the homogeneous case
            ax.axhline(
                reference_run_df[plot].values[0], color="red", label="homogeneous"
            )

            # labels
            # if i == n_plots - 1:
            ax.set_xlabel("lengthscale_lambda")
            ax.set_ylabel(plot)
            # legend
            ax.legend(loc="upper right")

            ax.set_ylim([min_val, max_val])
            ax.set_xlim([1, 1000])
            # set log scale for x
            ax.set_xscale("log")

            # Also plot the mean of the samples
            

            ax.plot(
                lengthscales_lambda,
                [
                    other_runs.query(f"std_lambda == {std_lambda} & lengthscale_lambda == {lengthscale_lambda}")[plot].mean() for lengthscale_lambda in lengthscales_lambda
                    # other_runs[other_runs["std_lambda"] == std_lambda & other_runs["lengthscale_lambda"] == lengtscale_lambda][plot].mean(axis=0) for lengtscale_lambda in lengthscales_lambda
                ],
                label=f"mean std_lambda={std_lambda:.2E}",
                color=std_lambda_colors[j],
            )

    plt.show()
