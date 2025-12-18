"""
Prevalence-by-age target for malaria model calibration.

This module provides a calibration target that compares simulated vs observed
prevalence-by-age using mean squared error as the evaluation metric.
"""

import polars as pl
import modelops_calabaria as cb
from modelops_calabaria.core.target import LossTarget
from modelops_calabaria.core.alignment import JoinAlignment
from modelops_calabaria.core.evaluation import mean_signal_mse


@cb.calibration_target(
    model_output="prevalence_by_age",
    data={
        'observed': "data/observed_prevalence.csv"
    }
)
def prevalence_by_age_target(data_paths):
    """
    Target comparing simulated vs observed prevalence-by-age using MSE.

    This target:
    1. Loads observed prevalence-by-age data from CSV (with counts)
    2. Computes observed prevalence from counts (number_positive/number_total)
    3. Aligns with simulated data on 'age_years' column
    4. Computes mean squared error as the evaluation metric

    Args:
        data_paths: Dict with paths to data files from decorator

    Returns:
        LossTarget: Configured target for prevalence-by-age evaluation with MSE
    """
    # Load the observation data
    observed_data = pl.read_csv(data_paths['observed'])

    # Compute observed prevalence from counts
    observed_data = observed_data.with_columns(
        (pl.col("number_positive") / pl.col("number_total")).alias("prevalence")
    )

    # Create and return the target
    return LossTarget(
        name="prevalence_by_age",
        model_output="prevalence_by_age",
        data=observed_data,
        alignment=JoinAlignment(
            on_cols="age_years",
            mode="exact"  # Require exact age matches
        ),
        evaluator=mean_signal_mse(col="prevalence", weight=1.0),
    )
