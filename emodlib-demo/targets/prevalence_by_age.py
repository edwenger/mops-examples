"""
Prevalence-by-age target for malaria model calibration.

This module provides a calibration target that compares simulated vs observed
prevalence-by-age using binomial negative log-likelihood as the loss metric.
"""

import polars as pl
import modelops_calabaria as cb
from modelops_calabaria.core.target import Target
from modelops_calabaria.core.alignment import JoinAlignment
from modelops_calabaria.core.evaluation import (
    BinomialNLL,
    Evaluator,
    MeanAcrossReplicates,
    MeanReducer,
)


def replicate_mean_binomial_nll(
    p_col: str = "prevalence",
    x_col: str = "number_positive",
    n_col: str = "number_total",
) -> Evaluator:
    """
    Create a binomial NLL evaluation strategy for comparing simulated prevalence
    to observed counts.

    This uses binomial negative log-likelihood to compare simulated prevalence
    (probability) against observed counts (number positive out of number total).

    The aggregator averages the simulated prevalence across replicates, while
    keeping the observed counts as-is.

    Args:
        p_col: Name of the prevalence/probability column in simulated data
        x_col: Name of the number positive column in observed data
        n_col: Name of the number total column in observed data

    Returns:
        Evaluator configured with binomial NLL loss
    """
    from modelops_calabaria.core.evaluation import IdentityAggregator

    return Evaluator(
        aggregator=IdentityAggregator(),
        loss_fn=BinomialNLL(p_col=p_col, x_col=x_col, n_col=n_col),
        reducer=MeanReducer(),
    )


@cb.calibration_target(
    model_output="prevalence_by_age",
    data={
        'observed': "data/observed_prevalence.csv"
    }
)
def prevalence_by_age_target(data_paths):
    """
    Target comparing simulated vs observed prevalence-by-age using binomial NLL.

    This target:
    1. Loads observed prevalence-by-age data from CSV (with counts)
    2. Aligns with simulated data on 'age_years' column
    3. Computes binomial negative log-likelihood as the loss metric

    Binomial NLL is appropriate when comparing simulated prevalence (probability)
    against observed counts (number positive out of number total). Lower values
    indicate better calibration.

    Args:
        data_paths: Dict with paths to data files from decorator

    Returns:
        Target: Configured target for prevalence-by-age evaluation with binomial NLL
    """
    # Load the observation data
    observed_data = pl.read_csv(data_paths['observed'])

    # Create and return the target
    return Target(
        model_output="prevalence_by_age",
        data=observed_data,
        alignment=JoinAlignment(
            on_cols="age_years",
            mode="exact"  # Require exact age matches
        ),
        evaluation=replicate_mean_binomial_nll(
            p_col="prevalence",
            x_col="number_positive",
            n_col="number_total",
        ),
        weight=1.0
    )
