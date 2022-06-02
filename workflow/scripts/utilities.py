# SPDX-FileCopyrightText: 2025 Koen van Greevenbroek & Aleksander Grochowicz
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Simple utility functions for working with PyPSA networks."""

from collections import OrderedDict
from pathlib import Path
from typing import Sequence

import linopy
import numpy as np
import pandas as pd
import pypsa
from pypsa.descriptors import get_extendable_i, nominal_attrs

marginal_attr = {"Generator": "p", "Link": "p", "Store": "p", "StorageUnit": "p"}


def translate_basis_variables(n: pypsa.Network, basis: dict) -> dict:
    """Follow `get_basis_variables` to get a dictionary that can be used for pypsa.optimize.optimize_mga.
    Note that `n` needs to be solved already."""

    # For each dimension, get the values of the variables in the basis.
    components = {}

    if hasattr(n, "model"):
        m = n.model
    else:
        m = n.optimize.create_model()

    for key, dim in basis.items():
        summands = []
        for spec in dim:
            # First, get the respecified variables for _all_
            # components of the given type, for example p_nom for
            # _all_ generators.
            vars = m[f"{spec['c']}-{spec['v']}"]
            # Now, we filter down to a desired subset of variables,
            # following keywords given in the specification. First,
            # narrow down to a given carrier.
            if "carrier" in spec:
                vars = vars.loc[
                    n.df(spec["c"])
                    .loc[
                        (n.df(spec["c"])["carrier"] == spec["carrier"])
                        & n.df(spec["c"])[f"{spec['v']}_extendable"]
                    ]
                    .index
                ]
            if "index" in spec:
                vars = vars.loc[[spec["index"]]]

            coeffs = pd.Series(1, index=vars.coords.to_index())
            if "weight" in spec:
                # We allow spec["weight"] to be either a string or an
                # iterable of strings. Either way, the variable
                # coefficients are multiplied by the values in the
                # component dataframe columns given by the(se) string(s).
                if isinstance(spec["weight"], str):
                    factors = [spec["weight"]]
                else:
                    factors = spec["weight"]
                for f in factors:
                    coeffs *= n.df(spec["c"]).loc[coeffs.index, f]

            summands.append(vars * coeffs)
        components[key] = summands
    return components


def get_basis_values(n: pypsa.Network, basis: OrderedDict, use_opt=True) -> OrderedDict:
    """Get the coordinates of a solved PyPSA network `n` in the given basis.

    If `use_opt` is set to True, it uses the optimal capacities *_nom_opt
    instead of *_nom capacities.
    """
    basis_caps = BasisCapacities(basis=basis, init=n, use_opt=use_opt)
    return basis_caps.project_to_coordinates()


class BasisCapacities:
    """A datastructure for storing a set of capacities of a PyPSA network."""

    def __init__(self, basis: OrderedDict, init: pypsa.Network = None, use_opt=False):
        """Initialise from a basis and optionally a PyPSA network."""
        self._basis = basis
        self._caps = {}
        if init is not None:
            # Make a copy of the initialising network, in case `init`
            # ever gets modified.
            n = init.copy()
            self._n = n
            for key, dim in basis.items():
                self._caps[key] = {}
                for spec in dim:
                    # Initialise the dataframe of capacities.

                    v = spec["v"] if not use_opt else spec["v"] + "_opt"

                    if spec["c"] not in self._caps[key]:
                        self._caps[key][spec["c"]] = pd.DataFrame(columns=["coeffs", v])

                    # Extract the capacities.
                    caps = n.df(spec["c"])[v]
                    # Filter by carrier
                    if "carrier" in spec:
                        caps = n.df(spec["c"]).loc[
                            n.df(spec["c"])["carrier"] == spec["carrier"]
                        ][v]
                    # Additionally filter by index
                    if "index" in spec:
                        caps = n.df(spec["c"]).loc[[spec["index"]]][v]

                    # Extract the coefficients.
                    coeffs = pd.Series(1, index=caps.index, name="coeffs")
                    if "weight" in spec:
                        if isinstance(spec["weight"], str):
                            factors = [spec["weight"]]
                        else:
                            factors = spec["weight"]
                        for f in factors:
                            coeffs *= n.df(spec["c"]).loc[caps.index, f]

                    # Store everything in the capacity datastructure.
                    df = pd.concat([coeffs, caps], axis="columns")
                    self._caps[key][spec["c"]] = pd.concat(
                        [self._caps[key][spec["c"]], df], axis="index"
                    )

    def project_to_coordinates(self) -> OrderedDict:
        """Project the capacities down to a space defined by the basis specification."""
        proj_values = OrderedDict()
        for key, caps in self._caps.items():
            value = 0
            for comp in caps.values():
                value += (comp.iloc[:, 0] * comp.iloc[:, 1]).sum()
            proj_values[key] = value

        return proj_values


def optimize_near_opt(
    n: pypsa.Network,
    direction: Sequence[float],
    basis: OrderedDict = None,
    max_obj: float = np.inf,
    snapshots=None,
    multi_investment_periods=False,
) -> None:
    """Solve the network `n` with custom objective function."""
    print("Start near-optimal optimisation.")
    solver_options = n.config["solving"]["solver"].copy()
    solver_name = solver_options.pop("name")
    tmpdir = n.config["solving"].get("tmpdir", None)
    if tmpdir is not None:
        Path(tmpdir).mkdir(parents=True, exist_ok=True)

    if multi_investment_periods:
        raise ValueError("Multi-investment periods not yet implemented.")

    if snapshots is None:
        snapshots = n.snapshots
    else:
        raise ValueError("Snapshots not yet implemented.")

    # create basic model
    m = n.optimize.create_model(
        snapshots=snapshots,
        multi_investment_periods=multi_investment_periods,
    )
    print("Set up model.")
    # Add budget constraint.
    m.add_constraints(get_objective(n, n.snapshots), "<=", max_obj, "Near_optimal")
    # Build alternate objective
    vars = translate_basis_variables(n, basis)
    objective = []
    for weights, dir in zip(vars.values(), direction):
        for summand in weights:
            objective.append(summand * dir)
    m.objective = linopy.merge([obj.sum() for obj in objective])
    print("Ready to optimise", m.objective)
    n.model = m
    status, condition = n.optimize.solve_model(
        solver_name=solver_name,
        solver_options=solver_options,
        solver_dir=tmpdir,
        assign_all_duals=True,
    )

    # Write near-opt coefficients into metadata.
    n.meta["max_obj"] = max_obj
    n.meta["direction"] = list(direction)
    return status, condition, n


# This function is adapted from pypsa.linopf and updated to linopy. Later updated to pypsa.optimize.
def get_objective(n, sns):
    """Return the objective function as a linear expression."""

    total = ""

    m = n.optimize.create_model()

    nom_attr = nominal_attrs.items()
    constant = 0
    for c, attr in nom_attr:
        ext_i = get_extendable_i(n, c)
        cost = n.df(c)["capital_cost"][ext_i]
        if cost.empty:
            continue

        constant += cost @ n.df(c)[attr][ext_i]

    objective = m.objective

    total = objective + float(constant)

    return total
