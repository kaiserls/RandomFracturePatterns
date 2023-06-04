# Random Fractures

Repository containing the tools to perform the calculations for the thesis *Phase field fracture models with random parameter fields*. The aim of this thesis is to quantify the influence of uncertain material parameters in the porous medium on the crack propagation.

## Measures

The proposed following measures are proposed to quantify the randomness of the crack:

* Global quantities:
  * Fractal dimension
  * crack length
  * crack area/volume

* Local quantities:
  * crack width
  * Deviation of the crack path from the crack path of the mean parameters

The local quantities can be transformed into global quantities by integrating, averaging, taking the maximum,...

They are implemented in the python files contained in ```src/measure/*.py```.

## Fracture simulation

The pde models describing the crack [propagation in porous media](http:// Hydraulically induced fracturing in heterogeneous porous media using a TPM-phase-field model and geostatistics) is solved with an inhouse code [PANDAS](https://www.mib.uni-stuttgart.de/cont/). The data are output as tecplot data, containing the data for the variables at the triangle nodes, but also at the gauss nodes of the used Finite-Element shape function.

## Installation

To install pandas, get a copy of the pandas software from MIB (see above). Additionally you need to get the app folders used in this thesis. Write an email to lars.g.kaiser@gmx.de to get in contact. You propably need to install the library `libreadline-dev` with `apt` to compile pandas.

To install the python code do:

```bash
git clone xxx
cd RandomFractures
pip install -r requirements.txt
```

## Running the program

```bash
python src/main.py
```

If this does fail due due to imports failing try:

```bash
python3 -m src.main
```

## Computing on a server

1. Connect to local network over vpn.
2. Copy your key to the server
```ssh-copy-id -i ~/.ssh/keyname.pub username@ipaddress``` e. g. ``ssh-copy-id -i ~/.ssh/id_ed25519.pub lars_k@129.69.167.191```
3. Visit
```https://emma-190.hpc.simtech.uni-stuttgart.de/```
or connect with ssh
```ssh username@ipaddress``` e. g. ssh ```lars_k@129.69.167.191```
4. Monitor cluster usage with htop

Start computation with tmux or screen:

tmux:
1. Start computation and keep connection / task alive with: ```todo```
2. use ```tmux``` terminal to start session
3. Run program (in background?) ```your-long-running-command &```. (& For background run)
4. Detach with type Ctrl+B → let go keys → D
5. retach with ```tmux attach```

Advanced usage:

```tmux ls```
```tmux a -t <name>```

screen:

1. Start computation with ```screen -dm bash -c 'python3 -m src.main' -S theOptionalScreenName -L```
2. Reattach with ```screen -r theOptionalScreenName```
3. Detach from screen with ```strg+a strg+d```
4. List screens with ```screen -ls``` to check if finished
5. Kill screen if needed with ```screen -S theOptionalScreenName -X quit```

kill tasks:

```ps -o pid= -u lars_k | xargs kill -1```

scoop:

```python3 -m scoop --hostfile hosts_191193 --external-hostname 129.69.167.191 src/main.py```
