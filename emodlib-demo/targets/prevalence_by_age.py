"""
Prevalence-by-age target for malaria model calibration.

This module provides a calibration target that compares simulated vs observed
prevalence-by-age using KL divergence as the loss metric.
"""

import numpy as np
import polars as pl
import modelops_calabaria as cb
from modelops_calabaria.core.target import Target
from modelops_calabaria.core.alignment import JoinAlignment, AlignedData, LOSS_COL
from modelops_calabaria.core.evaluation.composable import Evaluator
from modelops_calabaria.core.evaluation.aggregate import MeanAcrossReplicates
from modelops_calabaria.core.evaluation.reduce import MeanReducer


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


class KLDivergenceLoss:
    """Loss function that computes KL divergence per aligned row."""

    def __init__(self, col: str):
        """
        Initialize KL divergence loss.

        Args:
            col: Name of the column to evaluate
        """
        self.col = col

    def compute(self, aligned: AlignedData) -> AlignedData:
        """
        Compute KL divergence loss for aligned data.

        This computes KL(obs || sim) for each row and adds it as a loss column.

        Args:
            aligned: Aligned observed and simulated data

        Returns:
            AlignedData with loss column added
        """
        # Get observed and simulated columns
        obs_col = aligned.get_obs_col(self.col)
        sim_col = aligned.get_sim_col(self.col)

        # Compute element-wise KL divergence contribution
        # For KL(p||q) = sum(p * log(p/q)), each element contributes p_i * log(p_i / q_i)
        epsilon = 1e-10
        obs_clipped = obs_col.clip(epsilon, 1.0)
        sim_clipped = sim_col.clip(epsilon, 1.0)

        # Compute KL divergence contribution per row
        kl_contrib = obs_clipped * (obs_clipped / sim_clipped).log()

        # Add loss column
        aligned_with_loss = aligned.data.with_columns(kl_contrib.alias(LOSS_COL))

        return AlignedData(
            data=aligned_with_loss,
            on_cols=aligned.on_cols,
            replicate_col=aligned.replicate_col,
        )


def replicate_mean_kl_divergence(col: str = "prevalence") -> Evaluator:
    """
    Create a KL divergence evaluation strategy that averages replicates first.

    This is analogous to replicate_mean_mse from the standard library.

    Args:
        col: Name of the prevalence column

    Returns:
        Evaluator configured with KL divergence loss
    """
    return Evaluator(
        aggregator=MeanAcrossReplicates([col]),
        loss_fn=KLDivergenceLoss(col=col),
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
