import time
import pandas as pd
import psutil
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dask.distributed import Client, LocalCluster, SSHCluster
from dask import delayed, compute

from src.simulation.parameters import add_default_parameters, define_simulations
from src.simulation.simulation import simulate
from src.simulation.postprocess import postprocess
from src.simulation.analysis import analyze

def main():
    cluster = LocalCluster(n_workers=14, threads_per_worker=1)
    client =  Client(cluster) # Client("129.69.167.191:8786")
    
    print("The client is: ", client)
    print("The cluster is: ", cluster)
    print("The scheduler is: ", client.scheduler_info())
    print("The workers are: ", client.has_what())
    print("The dashboard is: ", client.dashboard_link)

    parameters = {}
    add_default_parameters(parameters)
    scenarios = define_simulations(parameters)
    scenarios_table = pd.DataFrame(scenarios, columns=scenarios[0].keys())
    scenarios_table.to_csv("scenarios.csv")

    simulation_scenarios = [delayed(simulate)(scenario) for scenario in scenarios]
    results = compute(*simulation_scenarios) # Compute all simulations in parallel
    results_table = pd.DataFrame(results, columns=results[0].keys())
    results_table.to_csv("results.csv")

    # postprocess(parameters, results)
    # analyze(parameters, results)

if __name__ == '__main__':
    main()