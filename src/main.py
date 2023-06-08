from functools import partial
import time
import pandas as pd
import psutil
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dask.distributed import Client, LocalCluster, SSHCluster
from dask import delayed, compute

import src.simulation.parameters as parameters
import src.simulation.simulation as simulation
import src.simulation.postprocessing as postprocessing
import src.simulation.analysis as analysis
import src.simulation.plotting as plotting

########################## SSHCluster #########################################
# cluster = LocalCluster()
# cluster = SSHCluster(
#     hosts=["129.69.167.191", "129.69.167.193"],
#     connect_options={"known_hosts": None, "username": "lars_k"},
#     # scheduler_options={"port": 0, "dashboard_address": ":8797"},
#     remote_python="/usr/bin/python3"
# )

# There seems to be a difference between the code above and the bash command.
# The bash command is working!
# dask-ssh 129.69.167.191 129.69.167.192 129.69.167.193 --ssh-username lars_k --remote-python /usr/bin/python3.10
# client = Client("129.69.167.191:8786")


def main():
    """The main function managing the simulation."""
    local = False
    if local:
        cluster = LocalCluster(n_workers=14, threads_per_worker=1)
        client = Client(cluster)
    else:
        cluster = SSHCluster(
            hosts=["129.69.167.191", "129.69.167.192", "129.69.167.193"],
            connect_options={"known_hosts": None, "username": "lars_k"},
            # scheduler_options={"port": 0, "dashboard_address": ":8797"},
            remote_python="/usr/bin/python3.10"
        )
        client = Client(cluster)

    print("The client is: ", client)
    print("The scheduler is: ", client.scheduler_info())
    print("The workers are: ", client.has_what())
    print("The dashboard is: ", client.dashboard_link)

    default_parameters = parameters.create_default_parameters()
    scenarios = parameters.define_scenarios(default_parameters)
    pd.DataFrame(scenarios).to_csv("results/scenarios.csv")

    scenario_simulations = [delayed(simulation.simulate)(scenario) for scenario in scenarios]
    simulation_results = compute(
        *scenario_simulations
    )  # Compute all simulations in parallel
    pd.DataFrame(simulation_results).to_csv("results/simulation_results.csv")

    postprocessing_results = postprocessing.postprocess(simulation_results)
    pd.DataFrame(postprocessing_results).to_csv("results/postprocessing_results.csv")

    analysis_results = analysis.analyze(postprocessing_results)
    analysis_results_table = pd.DataFrame(analysis_results)
    analysis_results_table.to_csv("results/analysis_results.csv")

    plotting.plot(analysis_results_table)

    client.close()


if __name__ == "__main__":
    main()
