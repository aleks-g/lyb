# SPDX-FileCopyrightText: 2025 Koen van Greevenbroek & Aleksander Grochowicz
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Optimising a PyPSA network with respect to its total system costs."""

import logging
import time
import warnings
from pathlib import Path

from utilities import get_basis_values
from workflow_utilities import configure_logging

# Ignore futurewarnings raised by pandas from inside pypsa, at least
# until the warning is fixed. This needs to be done _before_ pypsa and
# pandas are imported; ignore the warning this generates.
warnings.simplefilter(action="ignore", category=FutureWarning)

import pandas as pd  # noqa: E402
import pypsa  # noqa: E402

if __name__ == "__main__":
    # Set up logging so that everything is written to the right log file.
    configure_logging(snakemake)

    # Load the network and solving options.
    n = pypsa.Network(snakemake.input.network)
    solver_name = snakemake.config["solving"]["solver"].pop("name")
    tmpdir = snakemake.config["solving"].get("tmpdir", None)
    if tmpdir is not None:
        Path(tmpdir).mkdir(parents=True, exist_ok=True)
    solver_options = snakemake.config["solving"]["solver"]

    # Solve the network for the cost optimum and then get its
    # coordinates in the basis.
    logging.info("Compute initial, optimal solution.")
    t = time.time()
    # Do not need to have iterations, as there is no transmission.
    status, condition = n.optimize(
        solver_name=solver_name,
        solver_options=solver_options,
        assign_all_duals=True,
        solver_dir=tmpdir,
    )
    logging.info(f"Optimisation took {time.time() - t:.2f} seconds.")
    if "infeasible" in condition:
        labels = n.model.compute_infeasibilities()
        logging.info(f"Labels:\n{labels}")
        n.model.print_infeasibilities()
        raise RuntimeError("Solving status 'infeasible'")

    # Check if the optimisation succeeded; if not we don't output
    # anything in order to make snakemake fail. Not checking for this
    # would result in an invalid (non-optimal) network being output.
    if status == "ok":
        # Write the result to the given output files. Save the objective
        # value for further processing.
        n.export_to_netcdf(snakemake.output.optimum)
        opt_point = get_basis_values(n, snakemake.config["projection"], use_opt=True)
        pd.DataFrame(opt_point, index=[0]).to_csv(snakemake.output.optimal_point)
        with open(snakemake.output.obj, "w") as f:
            f.write(str(n.objective))
