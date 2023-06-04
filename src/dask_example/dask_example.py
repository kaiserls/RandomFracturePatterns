from dask.distributed import Client, SSHCluster, LocalCluster



if __name__ == "__main__":
    # cluster = LocalCluster()
    # cluster = SSHCluster(
    #     hosts=["129.69.167.191", "129.69.167.193"],
    #     connect_options={"known_hosts": None, "username": "lars_k"},
    #     # scheduler_options={"port": 0, "dashboard_address": ":8797"},
    #     remote_python="/usr/bin/python3"
    # )

    # There seems to be a difference between the code above and the bash command.
    # The bash command is working!
    # dask-ssh 129.69.167.191 129.69.167.193 --ssh-username lars_k
    client = Client("129.69.167.191:8786")

    def inc(x):
        return x + 1

    y = client.submit(inc, 10)
    print(y)
    print(y.result())
