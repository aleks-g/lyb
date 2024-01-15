<!--
SPDX-FileCopyrightText: 2024 Koen van Greevenbroek & Aleksander Grochowicz

SPDX-License-Identifier: CC-BY-4.0
-->


Under construction.


# Introduction

# Installation and usage

The model is built using a snakemake workflow, and [PyPSA-Eur](https://github.com/PyPSA/pypsa-eur) and [PyPSA-Eur-Sec](https://github.com/PyPSA/pypsa-eur-sec) are included as git submodules and snakemake modules.
While these two frameworks have recently been merged, the present workflow includes versions of PyPSA-Eur and PyPSA-Eur-Sec from before the merge.

1. Clone the git repository, making sure to also bring in the pypsa-eur and pypsa-eur-sec submodules with `--recurse-submodules`:

   ```sh
   git clone --recurse-submodules git@github.com:koen-vg/enabling-agency.git
   ```

2. Install a patched version of snakemake which deals properly with nested modules. First install conda or [mamba](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html), then build a conda environment containing the correct version of snakemake:

   ```sh
   mamba env create -f workflow/envs/snakemake.yaml
   ```

   Now activate the environment with `conda activate snakemake_patched`.

3. Place ERA5 cutouts produced by atlite in the `workflow/modules/pypsa-eur/cutouts` directory, following the naming scheme `europe-era5_{year}`. 

4. Execute a testing model by running the following:

   ```sh
   snakemake --configfile config/config-testing.yaml --use-conda -j all -- compute_all_intersections
   ```
   
   Before starting to run a big task, it's a good idea to check how snakemake plans to execute the various rules by running snakemake with the `-n` (`--dry-run`) argument.


# Organisation

We structure our implementation as a snakemake workflow with PyPSA-Eur and PyPSA-Eur-Sec as a snakemake submodules.
This image represents the workflow:
![Representation of the rule](images/validation_workflow.svg)

A summary of the most important rules follows:

1. The rule `compute_optimum` is analogous to `solve_network` in PyPSA-Eur. It solves the energy system with the given configuration such that the total system costs are minimised.

2. The rule `mga` is the first step used to explore the geometry of a near-optimal feasible space in cardinal directions. Allowing a total system cost increase within some chosen `eps` slack, it minimises and maximises the decision variables defined in the configuration file.

3. The rule `compute_near_opt` computes an approximation of the near-optimal space of the given network. It does so by solving the network repeatedly with different objectives in order to find extreme points of the near-optimal space.

4. The rule `compute_intersection` computes the intersection of the given near-optimal feasible spaces from step 3. Additionally, it computes the Chebyshev centre of the intersection.

5. The rule `compute_robust_networks` computes a number of robust networks located on the boundary of an intersection of near-optimal spaces.


# Computational requirements

The Snakemake workflow specification includes resource estimations, so running any workflow with the `-n` (dry-run) flag will show how much time and memory the task is expected to take. To reproduce the results presented in the paper, expect the computations to take approximately a couple of weeks on a small computing cluster.


# Configuration

A configuration file needs to have a name (corresponding to the file name of the form config-{name}.yaml) and furthermore the following:

- a "near_opt_approx" section with values for `directions`, `directions_angle_separation`, `num_parallel_solvers`, `iterations`, `conv_method`, `conv_epsilon` and `conv_iterations`;
- a "projection" section specifying variables and coefficients to aggregate to each dimension of the reduced near-optimal space.
- a "pypsa-eur" section which updates changes in the default pypsa-eur config (including the choice of countries, at which date to start a weather year, the CO_2 limit, the extendable and conventional carriers to be used, as well as options on atlite, the solver and load data).
- a "pypsa-eur-sec" section which updates changes in the defaul pypsa-eur-sec config (including choices about the gas network and which conventional generators to carry over from PyPSA-Eur).
- a "scenario" or "intersection_scenarios" section specifying the behaviour of `*_all_*` rules.
- a "robust_networks" sections specifying the options to sample robust networks (in full resolution) on the boundary of the intersection.

