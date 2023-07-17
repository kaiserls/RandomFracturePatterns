# Random Fractures

Repository containing the tools to perform the calculations for the thesis **Quantification of fracture patterns with random parameter fields for phase field models**. The aim of this thesis is to quantify the influence of uncertain material parameters in the porous medium on the crack propagation.

## Fracture simulation

The pde models describing the crack [propagation in porous media](https://www.researchgate.net/publication/371199835_Hydraulically_induced_fracturing_in_heterogeneous_porous_media_using_a_TPM-phase-field_model_and_geostatistics) is solved with an inhouse code [PANDAS](https://www.mib.uni-stuttgart.de/cont/). The data are output as tecplot data, containing the data for the variables at the triangle nodes, but also at the gauss nodes of the used Finite-Element shape function.

## Installation

```bash
git clone https://github.com/kaiserls/RandomFracturePatterns.git
cd RandomFracturePatterns
pip install .
```

To install PANDAS, get a copy of the PANDAS software from [MIB](https://www.mib.uni-stuttgart.de/cont/), place it in the folder `RandomFracturePatterns` and compile it following the user manual. You propably need to install the library `libreadline-dev` with `apt` to compile pandas successfully.

## Running the simulations

To run the simulations from this thesis you need to get the custom pandas `app` folder used in this thesis. Write an email to [me](mailto:lars.g.kaiser@gmx.de) to get in contact.

```bash
python src/main.py
```

If this does fail, due to failing imports, try:

```bash
python3 -m src.main
```

## Computing on a server

1. Connect to local network over vpn.
2. Copy your ssh key to the server
3. Start the computation, e.g. with screen

### Screen

1. Start computation with ```screen -dm bash -c 'python3 -m src.main' -S theOptionalScreenName -L```
2. Reattach with ```screen -r theOptionalScreenName```
3. Detach from screen with ```strg+a strg+d```
4. List screens with ```screen -ls``` to check if finished
5. Kill screen if needed with ```screen -S theOptionalScreenName -X quit```

### Kill rogue tasks

```ps -o pid= -u your_username | xargs kill -1```
