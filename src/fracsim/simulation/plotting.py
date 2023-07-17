import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update(
    {
        "text.usetex": True,
        # "font.family": "Helvetica",
        "figure.figsize": "4.9, 3.5",
        "font.size": 11.0,
        "font.family": "serif",
        "font.serif": "Palatino",
        "axes.titlesize": "medium",
        "figure.titlesize": "medium",
        "text.latex.preamble": "\\usepackage{amsmath}\\usepackage{amssymb}\\usepackage{siunitx}",
    }
)

y_axis_labels = {
    "volume": "volume [\\unit{\\milli\\meter\\squared}]",
    "count": "count [-]",
    "fractal_dimension": "fractal dimension [-]",
    "isoline_length": "isoline length [\\unit{\\milli\meter}]",
    "skeleton_length": "skeleton length [\\unit{\\milli\meter}]",
    "skeleton_y_max": "skeleton y max [\\unit{\\milli\meter}]",
    "isoline_y_max": "isoline y max [\\unit{\\milli\meter}]",
    "sign_changes": "sign changes [-]",
    "max_curvature": "max curvature [\\unit{\\per\\milli\meter}]",
    "mean_curvature": "mean curvature [\\unit{\\per\\milli\meter}]",
    "mean_curvature_smoothed": "mean curvature smoothed [\\unit{\\per\\milli\meter}]",
    "max_curvature_smoothed": "max curvature smoothed [\\unit{\\per\\milli\meter}]",
    "sign_changes_smoothed": "sign changes smoothed [-]",
    "sign_changes_smoothed_gaussian": "sign changes smoothed gaussian [-]",
}


def plot(
    df: pd.DataFrame,
    create_subplots: bool = False,
    plots: list = None,
    show_mean=True,
    show_run_idx=False,
):
    """Visualize the results"""
    # reference_run
    reference_run_df = df[df["homogeneous"] == True]
    # assert len(reference_run_df) == 1, "There should be only one reference run."
    # other
    other_runs = df[df["homogeneous"] == False]
    # drop outliers where the length is lower than 1000. This indicates crashes, because the simulation domain is 1000 long.
    other_runs = other_runs[other_runs["isoline_length"] > 999]

    if plots is None:
        plots = [
            "volume",
            "count",
            "fractal_dimension",
            "isoline_length",
            "skeleton_length",
            "total_curvature",
            "skeleton_y_max",
            "isoline_y_max",
            "sign_changes",
            "max_curvature",
            "max_curvature_smoothed",
        ]

    n_row_plots = len(plots)

    std_lambdas = other_runs["std_lambda"].unique()
    n_std_lambdas = len(std_lambdas)
    std_lambda_colors = plt.cm.rainbow(np.linspace(0, 1, n_std_lambdas))
    n_col_plots = n_std_lambdas if create_subplots else 1

    fig, axs = plt.subplots(
        n_row_plots, n_col_plots, figsize=(10, 30), tight_layout=True
    )
    if n_col_plots == 1:
        axs = np.array([axs]).T
    for i, plot in enumerate(plots):
        max_val, min_val = np.max(other_runs[plot]), np.min(other_runs[plot])
        if len(reference_run_df) > 0:
            hom_val = reference_run_df[plot].values[0]
            max_val = max(max_val, hom_val) * 1.05
            min_val = min(min_val, hom_val) * 0.95
        for j, std_lambda in enumerate(std_lambdas):
            if n_col_plots == 1:
                ax = axs[i, 0]
            else:
                ax = axs[i, j]

            lengthscales_lambda_all = other_runs[
                other_runs["std_lambda"] == std_lambda
            ]["lengthscale_lambda"]
            lengthscales_lambda = lengthscales_lambda_all.unique()

            ax.scatter(
                lengthscales_lambda_all,
                other_runs[other_runs["std_lambda"] == std_lambda][plot],
                label="samples",  # label=f"std_lambda={std_lambda:.2E}",
                color=std_lambda_colors[j],
            )
            ax.set_title(f"$\sigma(\lambda^S)$ = {std_lambda:.2E}")
            if show_run_idx:
                for _, data_point in other_runs[
                    other_runs["std_lambda"] == std_lambda
                ].iterrows():
                    ax.annotate(
                        data_point["run"],
                        (data_point["lengthscale_lambda"], data_point[plot]),
                    )
            if len(reference_run_df) > 0:
                # Draw a horizontal line at the homogeneous case
                ax.axhline(
                    reference_run_df[plot].values[0], color="red", label="homogeneous"
                )

            # labels
            # if i == n_plots - 1:
            ax.set_xlabel("length scale of $\lambda^S$ [\\unit{\\milli\meter}]")
            # ax.set_ylabel(plot.replace("_", " "))
            ax.set_ylabel(y_axis_labels[plot])
            # legend
            ax.legend(loc="upper right")

            ax.set_ylim([min_val, max_val])
            ax.set_xlim([0.8, 1250])
            # set log scale for x
            ax.set_xscale("log")

            # Also plot the mean of the samples
            if show_mean:
                ax.plot(
                    lengthscales_lambda,
                    [
                        other_runs.query(
                            f"std_lambda == {std_lambda} & lengthscale_lambda == {lengthscale_lambda}"
                        )[plot].mean()
                        for lengthscale_lambda in lengthscales_lambda
                        # other_runs[other_runs["std_lambda"] == std_lambda & other_runs["lengthscale_lambda"] == lengtscale_lambda][plot].mean(axis=0) for lengtscale_lambda in lengthscales_lambda
                    ],
                    label="samples",  # f"mean std_lambda={std_lambda:.2E}",
                    color=std_lambda_colors[j],
                )
                ax.set_title(f"$\sigma(\lambda^S) = {std_lambda:.2E}$")

    plt.show()
