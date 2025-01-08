# SPDX-FileCopyrightText: 2024 Aleksander Grochowicz & Koen van Greevenbroek
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Sample solutions in the near-optimal feasible space.
"""

import copy
import logging
import multiprocessing
import os
import time
import warnings
from collections import OrderedDict
from multiprocessing import Pool, get_context
from pathlib import Path
from itertools import islice, product


# Suppress future warnings from pandas, as this should be fixed in
# most recent PyPSA versions.
warnings.simplefilter("ignore", FutureWarning)

import numpy as np
import pandas as pd
import pypsa
import linopy
from _helpers import configure_logging
from geometry import filter_vectors_auto, uniform_random_hypersphere_sampler, ch_centre
#from solve_network import extra_functionality as sec_extra_functionality
from utilities import get_basis_variables
from scipy.spatial import ConvexHull

def get_intersection_vertices(
    intersection: pd.DataFrame, num_vertices: int, dim_groups: list[list[str]]
) -> list[pd.Series]:
    """Return a number of spread-out vertices of the given intersection.

    The vertices will include first those which min- and maximise the
    sums of the respective groups of coordinates given in
    `dim_groups`, and then a random sample of spread-out vertices.
    """
    # Check that num_vertices is high enough
    min_num_robust = 2 ** len(dim_groups) + 2 * len(dim_groups)
    if num_vertices < min_num_robust:
        raise ValueError(
            f"Number of robust networks specified ({num_vertices}) is"
            f" lower than the minimum of {min_num_robust}."
        )

    directions = []

    # First add directions with all combinations of -1, 1 in the
    # dimension groups given in `dim_groups`.
    dim_groups_min_max = list(product([-1, 1], repeat=len(dim_groups)))
    for dim_group_coords in dim_groups_min_max:
        v = pd.Series(0, index=intersection.columns, dtype=float)
        for c, g in zip(dim_group_coords, dim_groups):
            v.loc[g] = c
        directions.append(v)

    # Then include directions where each dimension group is min/maxed
    # individually.
    for g in dim_groups:
        for sign in [-1, 1]:
            v = pd.Series(0, index=intersection.columns, dtype=float)
            v.loc[g] = sign
            directions.append(v)

    # Finally generate the rest of the directions randomly
    directions.extend(
        map(
            lambda v: pd.Series(v, index=intersection.columns),
            islice(
                filter_vectors_auto(
                    uniform_random_hypersphere_sampler(len(intersection.columns)),
                    init_angle=90,
                ),
                num_vertices - len(directions),
            ),
        )
    )

    # Now, for each direction, find the vertex of the intersection
    # furthest in that direction.
    vertices = []
    for d in directions:
        i = np.matmul(intersection.values, d.values).argmax()
        vertices.append(intersection.iloc[i])

    return vertices


def compute_metrics(n: pypsa.Network, level: float) -> pd.Series:
    """Compute the metrics as specified in the config."""
    metrics_opts = snakemake.config["sample"]["metrics"]
    metrics = pd.Series(index=metrics_opts.keys(), dtype=float)
    for metric in metrics_opts:
        if metric == "electricity_price":
            metrics[metric] = n.buses_t.marginal_price.loc[:,"Central electric bus"].mean()
        elif metric == "heat_price":
            metrics[metric] = n.buses_t.marginal_price.loc[:,"Central heat bus"].mean()
        elif metric == "emissions":
            diesel_i = n.generators.index[n.generators.carrier == "Diesel"]
            metrics[metric] = (n.generators_t.p[diesel_i] / network.generators.loc[diesel_i, "efficiency"]).sum().sum() * 0.254
        elif metric == "visual_impact":
            metrics[metric] = n.generators[n.generators.carrier == "Wind"].p_nom_opt.sum() / 4.2 # number of 4.2 MW turbines
        elif metric == "land_use":
            bioenergy_i = n.generators.index[n.generators.carrier.isin([ "Bio-fuel", "Pellets"])]
            ammonia_i = n.generators.index[n.generators.carrier == "NH3"]
            wind_area = n.generators.loc[n.generators.carrier == "Wind", "p_nom_opt"].sum() * 18000
            solar_area = n.generators.loc[n.generators.carrier == "Solar", "p_nom_opt"].sum() * 50505
            bioenergy_area = (n.generators.loc[n.generators.carrier == "NH3", "p_nom_opt"].sum() + n.generators.loc[n.generators.carrier == "Bio-fuel", "p_nom_opt"] + n.generators.loc[n.generators.carrier == "Pellets", "p_nom_opt"]).sum() * 25 # assume same value as for diesel
            diesel_area = n.generators.loc[n.generators.carrier == "Diesel", "p_nom_opt"].sum() * 25 
            bioenergy_gen_area = n.generators_t.p[bioenergy_i].sum().sum() * 12.65
            ammonia_gen_area = n.generators_t.p[ammonia_i].sum().sum() * 2.28
            battery_area = n.stores.loc[n.stores.carrier == "AC", "e_nom_opt"].sum() * 6.25 # battery
            heat_area = n.stores.loc[n.stores.carrier == "Heat", "e_nom_opt"].sum() * 1.556
            metrics[metric] = wind_area + solar_area + bioenergy_area + diesel_area + bioenergy_gen_area + ammonia_gen_area + battery_area + heat_area
        elif metric == "vulnerability":
            # Share_VRE
            share_vre = n.generators_t.p[["Solar PV", "Wind"]].sum().sum()/n.generators_t.p.sum().sum()

            # Share imports
            imported_i = n.generators.index[n.generators.carrier.isin(["Diesel", "NH3", "Bio-fuel", "Pellets"])]
            share_imports = n.generators_t.p[imported_i].sum().sum()/n.generators_t.p.sum().sum()

            # Complexity of system: count generation technologies with a capacity of at least 1 MW, grouped by carrier (max. 6 carriers)
            complexity = 1/6 * len(((n.generators.groupby("carrier").sum())[n.generators.groupby("carrier").sum()["p_nom_opt"] > 1]).index)

            # Heat storage, relative to 2 weeks of winter heat load or 1 month during the summer (4 GWh)
            heat_storage = n.stores.loc[n.stores.carrier == "Heat", "e_nom_opt"].sum()/4000


            metrics[metric] = 1 - (
               0.2 * (1 - share_vre) + 0.5*(1 - share_imports) + 0.2 * (1 - complexity) + 0.2 * (min(1,heat_storage))
            )
        elif metric == "slack":
            metrics[metric] = level
        else:
            raise ValueError(f"Unknown metric {metric}.")
    return metrics




def compute_networks(
    n: pypsa.Network,
    point: pd.Series,
    centre: pd.Series,
    basis: OrderedDict,
    out_dir: str,
    num_iter: int,
    df_metrics: pd.DataFrame,
    level: float) -> pypsa.Network:
    """Compute the network for a given point in the near-optimal feasible space."""
    m = n.copy()
    m.opts = n.opts
    m.config = n.config
    solving_options = n.config["solving"]["options"]
    solver_options = n.config["solving"]["solver"].copy()
    solver_name = solver_options.pop("name")
    tmpdir = n.config["solving"].get("tmpdir", None)
    if tmpdir is not None:
        Path(tmpdir).mkdir(parents=True, exist_ok=True)

    # Take a point that's a little towards to the centre of the intersection
    buffer = 0.01
    point = (1 - buffer) * point + buffer * centre

    def set_coordinates(n, snapshots):
        """Extra functionality to set total capacities on pypsa-eur network."""
        basis_variables = get_basis_variables(n, basis)
        # Add constraint to make the solution near-optimal.
        for key in basis_variables:
            x = basis_variables[key]
            b = point[key]
            scaling_factor = float(100 / b)
            # Define lower and upper constraints in the given basis
            # dimension, fixing the coordinates.
            name = f"fixed_investment_{key}"
            n.model.add_constraints(
                linopy.LinearExpression.from_tuples((scaling_factor * x.coeffs, x.vars)).sum(),
                "==",
                scaling_factor * b,
                name=name,
                attr="mu",
                axes=pd.Index([name]),
            )
      
    status, condition = n.optimize(
        n,
        solver_name=solver_name,
        solver_options=solver_options,
        extra_functionality=set_coordinates,
        solver_dir=tmpdir,
        # skip_objective=True,
        assign_all_duals=True,
    )
    if status == "ok":
        p_str = "_".join(map(lambda t: f"{t[0]}:{t[1]:.3f}", point.items()))
        print(f"Optimisation {num_iter} successful, exporting to {p_str}.nc")
        # Note that `num_iter` depends on when it was an input, and
        # does not necessarily reflect the order in which the outputs
        # are generated.
        m.export_to_netcdf(os.path.join(out_dir, p_str + ".nc"))

        # Compute the metrics.
        metric_val = compute_metrics(m, level)
        df_metrics.loc[num_iter] = metric_val


def determine_sample_number(slack, config):
    level_sets = config["sample"]["level_sets"]
    for level_set in level_sets:
        if level_set["min_level"] <= slack < level_set["max_level"]:
            return level_set["samples"]
    raise ValueError(f"Slack level {slack} not in any level set.")


if __name__ == "__main__":
    # Set up logging so that everything is written to the right log file.
    configure_logging(snakemake)

    # Load the network
    n = pypsa.Network(snakemake.input.network)

    n.config = snakemake.config["pypsa-longyearbyen"]
    n.opts = snakemake.config["pypsa-longyearbyen"]["opts"]

    qhull_options = n.config["near_opt_approx"]["qhull_options"]["near_opt"]

    # Load the near-optimal spaces.
    near_opt_spaces = [pd.read_csv(s, index_col=0) for s in snakemake.input.near_opt]
    slack_levels = []
    level_sets = snakemake.config["sample"]["level_sets"]
    for level_set in level_sets:
        slack_levels.extend(np.arange(100*level_set["min_level"], 100*level_set["max_level"], 100*level_set["step"]))
    spaces = dict(zip(slack_levels, near_opt_spaces))

    # Load the basis
    basis = snakemake.config["projection"]

    # Compute the centre point of the near-optimal feasible space (for geometry purposes).
    centres = {}
    for level, space in spaces.items():
        centre, _, _ = ch_centre(ConvexHull(space, qhull_options = qhull_options))
        centres[level] = centre

    
    # Make sure the output directory exists
    if not os.path.exists(snakemake.params.networks_dir):
        os.makedirs(snakemake.params.networks_dir)

    logging.info(f"Writing robust networks to {snakemake.params.networks_dir}")


    # Sample on the boundary of the near-optimal feasible space.
    vertices = {}
    metrics = {}
    for level, space in spaces.items():
        vertices[level] = get_intersection_vertices(space, determine_sample_number(level, snakemake.config), basis.keys())

        df_metrics = pd.DataFrame(columns = snakemake.config["sample"]["metrics"])

        # Operate sampled networks in parallel
        with get_context("spawn").Pool(snakemake.params.num_parallel) as p:
            p.starmap(
                compute_networks,
                [
                    (n, v, centre, basis, snakemake.params.networks_dir, num_iter, df_metrics, level)
                    for num_iter, v in enumerate(vertices)
                ],
            )
        
        metrics[level] = df_metrics
        logging.info(f"Sampled {len(vertices[level])} robust networks at slack level {level}.")
        
    # Export the vertices and metrics.
    sampled_vertices = pd.concat(vertices).to_csv(snakemake.output.sampled_vertices)
    sampled_metrics = pd.concat(metrics).to_csv(snakemake.output.sampled_metrics)
    
