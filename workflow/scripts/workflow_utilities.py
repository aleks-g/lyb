# SPDX-FileCopyrightText: 2025 Koen van Greevenbroek & Aleksander Grochowicz
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Simple utilities aiding the Snakemake workflow."""


import copy
import hashlib
import json
import sys
from pathlib import Path


def hash_config(configuration: dict):
    """Compute hash of most config file contents.

    The hash helps optimisation runs initiated with the same config
    file to be reused, specifically in the `compute_near_opt`
    snakemake rule. However, we exclude some specific keywords in the
    near_opt_approx configuration from the hash computation, since
    they are only parameters but do not influence the results of the
    algorithm beyong accuracy.

    """
    config_to_be_hashed = copy.deepcopy(configuration)
    # Remove the following keywords from the near_opt_approx
    # configuration in the config file.
    for i in [
        "iterations",
        "conv_epsilon",
        "conv_iterations",
        "directions_angle_separation",
        "num_parallel_solvers",
        "qhull_options",
        "intersection_pre_cluster",
    ]:
        if i in config_to_be_hashed["near_opt_approx"]:
            del config_to_be_hashed["near_opt_approx"][i]
    if "solving" in config_to_be_hashed:
        del config_to_be_hashed["solving"]
    if "scenario" in config_to_be_hashed:
        del config_to_be_hashed["scenario"]
    if "sample" in config_to_be_hashed:
        del config_to_be_hashed["sample"]
    return hashlib.md5(
        str(json.dumps(config_to_be_hashed, sort_keys=True)).encode()
    ).hexdigest()[:8]


# This is from pypsa-eur. (__helpers.py)
def configure_logging(snakemake, skip_handlers=False):
    """
    Configure the basic behaviour for the logging module.

    Note: Must only be called once from the __main__ section of a script.

    The setup includes printing log messages to STDERR and to a log file defined
    by either (in priority order): snakemake.log.python, snakemake.log[0] or "logs/{rulename}.log".
    Additional keywords from logging.basicConfig are accepted via the snakemake configuration
    file under snakemake.config.logging.

    Parameters
    ----------
    snakemake : snakemake object
        Your snakemake object containing a snakemake.config and snakemake.log.
    skip_handlers : True | False (default)
        Do (not) skip the default handlers created for redirecting output to STDERR and file.
    """
    # First, set up the logging module.
    import logging

    kwargs = snakemake.config.get("logging", dict())
    kwargs.setdefault("level", "INFO")

    if skip_handlers is False:
        fallback_path = Path(__file__).parent.joinpath(
            "..", "logs", f"{snakemake.rule}.log"
        )
        logfile = snakemake.log.get(
            "python", snakemake.log[0] if snakemake.log else fallback_path
        )
        kwargs.update(
            {
                "handlers": [
                    # Prefer the 'python' log, otherwise take the first log for each
                    # Snakemake rule
                    logging.FileHandler(logfile),
                    logging.StreamHandler(),
                ]
            }
        )
    logging.basicConfig(**kwargs)

    # Also configure exceptions to be logged.
    # (See https://stackoverflow.com/a/16993115)
    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return

        logging.error(
            "Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback)
        )

    sys.excepthook = handle_exception
