import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot(data_table: pd.DataFrame):
    """Visualize the results"""
    plots = ["fractal_dimension", "volume", "length", "width", "deviation"]
    n_plots = len(plots)

    fig, axs = plt.subplots(n_plots, 1, figsize=(10, 20))
    for i, plot in enumerate(plots):
        axs[i].set_title(plot)
        # Scatter the plot variable against the lengthscale_lambda
        axs[i].scatter(
            data_table["lengthscale_lambda"],
            data_table[plot],
            label=plot,
        )

        # homog_df = df[df.index.str.contains("homog", regex=False)]
        # # Draw a horizontal line at the homogeneous case
        # axs[i].axhline(homog_df[column].values[0], color="red", label="homogeneous")

        # labels
        axs[i].set_xlabel("lengthscale_lambda")
        axs[i].set_ylabel(plot)
        # legend
        axs[i].legend()

    plt.show()
