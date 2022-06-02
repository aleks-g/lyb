<!--
SPDX-FileCopyrightText: 2024 Koen van Greevenbroek & Aleksander Grochowicz

SPDX-License-Identifier: CC-BY-4.0
-->


# Introduction

This repository contains code to produce input for an [interface](https://github.com/koen-vg/near-opt-interface) for exploring near-optimal solutions interactively; see Vågerö et al (2025), "Exploring near-optimal energy systems with stakeholders: a novel approach for participatory modelling" ([arXiv:2501.05280](https://arxiv.org/abs/2501.05280)).
Specifically, this repository is used to computes samples of near-optimal solutions for a given model, together with metrics evaluated on those sampled solutions.

The present repository is a stripped-down and updated version of a framework originally developed for an earlier publication on near-optimal spaces; see [https://doi.org/10.1016/j.eneco.2022.106496] for the original publication and [https://github.com/aleks-g/intersecting-near-opt-spaces] for the original repository.
While the original version was tailored to use PyPSA-Eur, the present version focusses on computing results for a single pre-generated PyPSA model.

In Vågerö et al (2025), this repository is used in conjunction with a model for Longyearbyen commissioned by the local municipality (Lokalstyret) for use in the Energiplan report. While the code base for generating said model cannot be shared under an open license, the resulting PyPSA model is provided in this repository (under `networks/energiplan/energiplan.nc`).


# Installation and usage

This repository is structured around a snakemake workflow; dependencies are mananged using conda/mamba. Make sure that you have conda/mamba installed, clone this repository, then follow these steps:

1. Install snakemake; you can use the provided environment:

   ```sh
   mamba env create -f workflow/envs/snakemake.yaml
   ```

   Now activate the environment with `conda activate snakemake`.

2. Execute the main workflow using the following command:

   ```sh
   snakemake --configfile config/config-energiplan.yaml --use-conda -j all -- results/energiplan/samples/energiplan_samples.csv
   ```


# Organisation


A summary of the most important rules follows:

1. The rule `compute_optimum` computes an initial cost-optimal solution to the given model.

2. The rule `mga` is the first step used to explore the geometry of a near-optimal feasible space in cardinal directions. Allowing a total system cost increase within some chosen `eps` slack, it minimises and maximises the decision variables defined in the configuration file.

3. The rule `compute_near_opt` computes an approximation of the near-optimal space of the given network. It does so by solving the network repeatedly with different objectives in order to find extreme points of the near-optimal space.

4. The rule `sample_near_opt_space` re-optimises previously computed solutions, this time with investment decision variables fixed. Only an operational problem remains; this it done in order to correctly compute metrics which rely on shadow prices, such as electricity prices.
