"""
Prevalence-by-age target for malaria model calibration.

This module provides a calibration target that compares simulated vs observed
prevalence-by-age using KL divergence as the loss metric.
"""

import numpy as np
import polars as pl
import modelops_calabaria as cb
from modelops_calabaria.core.target import Target
from modelops_calabaria.core.alignment import JoinAlignment
from modelops_calabaria.core.evaluation import EvaluationStrategy


def kl_divergence(p: np.ndarray, q: np.ndarray, epsilon: float = 1e-10) -> float:
    """
    Compute Kullback-Leibler divergence: KL(P || Q) = sum(P * log(P / Q)).

    This measures how much the simulated distribution Q differs from the
    observed distribution P. Lower values indicate better fit.

    Args:
        p: Observed probability distribution (reference)
        q: Simulated probability distribution (model output)
        epsilon: Small value added to avoid log(0)

    Returns:
        KL divergence (non-negative, 0 = perfect match)
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)

    # Add epsilon to avoid division by zero and log(0)
    p = np.clip(p, epsilon, 1.0)
    q = np.clip(q, epsilon, 1.0)

    # KL divergence: sum(p * log(p/q))
    return float(np.sum(p * np.log(p / q)))


class KLDivergenceEvaluation(EvaluationStrategy):
    """
    Evaluation strategy using KL divergence loss.

    This strategy computes KL divergence between observed and simulated
    prevalence values after averaging across replicates.
    """

    def __init__(self, prevalence_col: str = "prevalence"):
        """
        Initialize KL divergence evaluation.

        Args:
            prevalence_col: Name of the prevalence column to evaluate
        """
        self.prevalence_col = prevalence_col

    def __call__(self, observed: pl.DataFrame, simulated: pl.DataFrame) -> float:
        """
        Compute KL divergence loss between observed and simulated data.

        Args:
            observed: Observed data (already joined with simulated)
            simulated: Simulated data (potentially multiple replicates)

        Returns:
            KL divergence (lower is better, 0 = perfect match)
        """
        # If simulated has multiple replicates, average them first
        # The alignment should have already joined on age_years
        if simulated.height > observed.height:
            # Average across replicates (group by age_years if present)
            if 'age_years' in simulated.columns:
                simulated = simulated.group_by('age_years').agg(
                    pl.col(self.prevalence_col).mean()
                ).sort('age_years')
            else:
                # If no grouping column, just take mean
                simulated = simulated.select(
                    pl.col(self.prevalence_col).mean()
                )

        # Extract prevalence values
        # After join alignment, both dataframes should have matching rows
        observed_prev = observed[self.prevalence_col].to_numpy()
        simulated_prev = simulated[self.prevalence_col].to_numpy()

        # Ensure same length (alignment should guarantee this)
        n = min(len(observed_prev), len(simulated_prev))
        observed_prev = observed_prev[:n]
        simulated_prev = simulated_prev[:n]

        # Compute KL divergence
        return kl_divergence(observed_prev, simulated_prev)


def replicate_mean_kl_divergence(col: str = "prevalence") -> KLDivergenceEvaluation:
    """
    Create a KL divergence evaluation strategy that averages replicates first.

    This is analogous to replicate_mean_mse from the standard library.

    Args:
        col: Name of the prevalence column

    Returns:
        KLDivergenceEvaluation instance
    """
    return KLDivergenceEvaluation(prevalence_col=col)


@cb.calibration_target(
    model_output="prevalence_by_age",
    data={
        'observed': "data/observed_prevalence.csv"
    }
)
def prevalence_by_age_target(data_paths):
    """
    Target comparing simulated vs observed prevalence-by-age using KL divergence.

    This target:
    1. Loads observed prevalence-by-age data from CSV
    2. Aligns with simulated data on 'age_years' column
    3. Computes KL divergence as the loss metric

    KL divergence measures how much the simulated distribution differs from
    the observed distribution. Lower values indicate better calibration.

    Args:
        data_paths: Dict with paths to data files from decorator

    Returns:
        Target: Configured target for prevalence-by-age evaluation with KL divergence
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
        evaluation=replicate_mean_kl_divergence(col="prevalence"),
        weight=1.0
    )
