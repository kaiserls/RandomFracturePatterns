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

def main():
    """The main function managing the simulation."""
    cluster = LocalCluster(n_workers=14, threads_per_worker=1)
    client =  Client(cluster) # Client("129.69.167.191:8786")
    
    print("The client is: ", client)
    print("The cluster is: ", cluster)
    print("The scheduler is: ", client.scheduler_info())
    print("The workers are: ", client.has_what())
    print("The dashboard is: ", client.dashboard_link)

    default_parameters = parameters.create_default_parameters()
    scenarios = parameters.define_scenarios(default_parameters)
    scenarios_table = pd.DataFrame(scenarios, columns=scenarios[0].keys())
    scenarios_table.to_csv("scenarios.csv")

    scenario_simulations = [delayed(simulation.simulate)(scenario) for scenario in scenarios]
    simulation_results = compute(*scenario_simulations) # Compute all simulations in parallel
    simulation_results_table = pd.DataFrame(simulation_results, columns=simulation_results[0].keys())
    simulation_results_table.to_csv("simulation_results.csv")

    postprocessing_results = postprocessing.postprocess(scenarios, simulation_results)
    postprocessing_results_table = pd.DataFrame(postprocessing_results, columns=postprocessing_results[0].keys())
    postprocessing_results_table.to_csv("postprocessing_results.csv")

    analysis_results = analysis.analyze(scenarios, simulation_results, postprocessing_results)
    analysis_results_table = pd.DataFrame(analysis_results, columns=analysis_results[0].keys())
    analysis_results_table.to_csv("analysis_results.csv")

    plotting.plot(analysis_results_table)

if __name__ == '__main__':
    main()