"""
EMOD malaria model for seasonal challenge calibration.

This model simulates multiple individuals exposed to seasonal malaria transmission
using the emodlib IntrahostComponent. It computes prevalence-by-age as the target
metric for calibration.

Thread Safety (emodlib >= 0.1.0):
---------------------------------
This model uses the instance-based MalariaConfig system for thread-safe parallel
simulations. Each trial creates its own config via create_config(), which is then
passed to IntrahostComponent.create(config). This eliminates global state and
enables safe parallel execution across Dask workers.
"""

import datetime as dt
from typing import Dict, Any, Mapping
import numpy as np
import pandas as pd
import polars as pl

from emodlib.malaria import IntrahostComponent, create_config
import modelops_calabaria as cb
from modelops_calabaria import (
    BaseModel, ParameterSpace, ParameterSet, ParameterSpec,
    ConfigurationSpace, ConfigSpec, model_output
)


def surface_area_biting_function(age_days: float) -> float:
    """
    Age-dependent biting risk function.

    Piecewise linear rising from birth to age 2, then shallower slope to age 20.
    From EMOD SusceptibilityVector.h implementation.

    Args:
        age_days: Age in days

    Returns:
        Relative biting risk (0.07 at birth, 1.0 at age 20+)
    """
    newborn_risk = 0.07
    two_year_old_risk = 0.23

    if age_days < 2 * 365:
        return newborn_risk + age_days * (two_year_old_risk - newborn_risk) / (2 * 365.)

    if age_days < 20 * 365:
        return two_year_old_risk + (age_days - 2 * 365.) * (1 - two_year_old_risk) / ((20 - 2) * 365)

    return 1.0


def month_index_from_day(t: int) -> int:
    """
    Convert day index to month index (0-11).

    Args:
        t: Day index (starting from day 0 = Jan 1, 2000)

    Returns:
        Month index (0 = January, 11 = December)
    """
    y2k = dt.datetime(2000, 1, 1)
    return (y2k + dt.timedelta(days=t)).month - 1


def monthly_eir_challenge(
    config,
    duration: int,
    monthly_eirs: tuple,
    updates_per_day: int = 2,
) -> pd.DataFrame:
    """
    Run a single individual through seasonal malaria exposure.

    This function creates a new IntrahostComponent instance and runs it through
    daily exposure based on monthly EIR values. The individual ages from 0 to
    duration days over the simulation.

    Args:
        config: MalariaConfig instance (thread-safe configuration)
        duration: Simulation duration in days
        monthly_eirs: Tuple of 12 monthly EIR values
        updates_per_day: Number of intrahost updates per day

    Returns:
        DataFrame with columns: parasite_density, gametocyte_density, fever_temperature
    """
    # Create new instance with config (thread-safe, no global state)
    ic = IntrahostComponent.create(config)
    ic.susceptibility.age = 0.0  # Start as newborn

    # Storage for daily results
    asexuals = np.zeros(duration)
    gametocytes = np.zeros(duration)
    fevers = np.zeros(duration)

    for t in range(duration):
        # Calculate age-adjusted daily EIR
        daily_eir = monthly_eirs[month_index_from_day(t)] * 12 / 365.0
        daily_eir *= surface_area_biting_function(ic.susceptibility.age)
        p_infected = 1 - np.exp(-daily_eir)

        # Stochastic challenge
        if np.random.random() < p_infected:
            ic.challenge()

        # Update intrahost dynamics
        for _ in range(updates_per_day):
            ic.update(dt=1.0/updates_per_day)

        # Record daily state
        asexuals[t] = ic.parasite_density
        gametocytes[t] = ic.gametocyte_density
        fevers[t] = ic.fever_temperature

    return pd.DataFrame({
        'parasite_density': asexuals,
        'gametocyte_density': gametocytes,
        'fever_temperature': fevers,
    })


class EmodMalariaModel(BaseModel):
    """
    EMOD malaria model for prevalence-by-age calibration.

    This model simulates N individuals from birth through 20 years of seasonal
    malaria exposure. It computes prevalence (parasite_density > 16) aggregated
    by age (year) as the primary calibration target.
    """

    # Class-level PARAMS required by modelops_calabaria BaseModel
    PARAMS = ParameterSpace((
        ParameterSpec(
            "Antigen_Switch_Rate",
            5e-10, 5e-8,
            "float",
            doc="Rate of antigenic variation (log scale)",
        ),
        ParameterSpec(
            "Falciparum_PfEMP1_Variants",
            500, 1200,
            "int",
            doc="Number of distinct PfEMP1 variants",
        ),
        ParameterSpec(
            "Max_Individual_Infections",
            3, 7,
            "int",
            doc="Maximum concurrent infections per individual",
        ),
    ))

    # Configuration space for fixed simulation settings
    CONFIG = ConfigurationSpace((
        ConfigSpec("n_people", default=10, doc="Number of individuals to simulate"),
        ConfigSpec("duration", default=20 * 365, doc="Simulation duration in days (20 years)"),
        ConfigSpec("updates_per_day", default=2, doc="Number of intrahost updates per day"),
        # Rafin Marke seasonal pattern (example from seasonal_challenge.py)
        ConfigSpec("monthly_eirs", default=(1, 1, 0.5, 1, 1, 2, 3.875, 7.75, 15.0, 3.875, 1, 1),
                   doc="Tuple of 12 monthly EIR values"),
    ))

    def __init__(self):
        """Initialize the malaria model."""
        super().__init__()

    def build_sim(self, params: ParameterSet, config: Mapping[str, Any]) -> Dict[str, Any]:
        """
        Build simulation state from parameters.

        This method packages parameters and configuration without creating
        IntrahostComponent instances (which would use global state).

        Args:
            params: Parameter set from optimization algorithm
            config: Configuration from base_config (may be patched by scenarios)

        Returns:
            Dictionary containing all simulation parameters and configuration
        """
        # Convert to native Python types to ensure C++ compatibility
        # (framework may deserialize as numpy types which pybind11 rejects)
        return {
            'antigen_switch_rate': float(params['Antigen_Switch_Rate']),
            'pfemp1_variants': int(params['Falciparum_PfEMP1_Variants']),
            'max_infections': int(params['Max_Individual_Infections']),
            'n_people': int(config['n_people']),
            'duration': int(config['duration']),
            'monthly_eirs': config['monthly_eirs'],
            'updates_per_day': int(config['updates_per_day']),
        }

    def run_sim(self, state: Dict[str, Any], seed: int = 42) -> Dict[str, Any]:
        """
        Run malaria simulation for N individuals.

        Creates a thread-safe config instance for this trial and passes it to
        all individual simulations.

        Args:
            state: Simulation state from build_sim
            seed: Random seed for this trial (used for RNG initialization)

        Returns:
            Dictionary containing raw results from all individuals
        """
        # Create thread-safe config instance for this trial
        # Ensure native Python types for pybind11 compatibility
        # (framework serialization may convert to numpy types)
        # Note: Run_Number must fit in int32 for C++ bindings
        params_dict = {
            'Run_Number': int(seed) % (2**31),
            'infection_params': {
                'Antigen_Switch_Rate': float(state['antigen_switch_rate'])
            },
            'Falciparum_PfEMP1_Variants': int(state['pfemp1_variants']),
            'Max_Individual_Infections': int(state['max_infections']),
        }

        config = create_config(params_dict)

        # Set NumPy random seed for challenge stochasticity
        # Use modulo to ensure seed fits in expected range
        np.random.seed(int(seed) % (2**32))

        # Extract values with native Python types for pybind11 compatibility
        n_people = int(state['n_people'])
        duration = int(state['duration'])
        updates_per_day = int(state['updates_per_day'])
        monthly_eirs = tuple(state['monthly_eirs'])

        # Run all individuals with the same config (thread-safe)
        individual_results = []
        for individual in range(n_people):
            df = monthly_eir_challenge(
                config=config,
                duration=duration,
                monthly_eirs=monthly_eirs,
                updates_per_day=updates_per_day,
            )
            individual_results.append(df)

        return {
            'individual_results': individual_results,
            'n_people': n_people,
            'duration': duration,
        }

    @model_output("prevalence_by_age")
    def extract_prevalence_by_age(self, raw_output: Dict[str, Any], seed: int) -> pl.DataFrame:
        """
        Extract prevalence-by-age from simulation results.

        This method:
        1. Computes prevalence (parasite_density > 16) for each individual
        2. Groups by year (age) for each individual
        3. Averages prevalence across individuals
        4. Returns as Polars DataFrame with columns: age_years, prevalence

        Args:
            raw_output: Raw simulation output from run_sim
            seed: Random seed used for the simulation

        Returns:
            DataFrame with columns: age_years (int), prevalence (float)
        """
        individual_results = raw_output['individual_results']

        # Process each individual: compute yearly prevalence
        yearly_prevalences = []
        for df in individual_results:
            # Compute prevalence (parasite density > threshold)
            df['prevalent'] = (df['parasite_density'] > 16).astype(int)

            # Group by year (assuming each year = 365 days)
            df['year'] = df.index // 365

            # Average prevalence within each year for this individual
            yearly_prev = df.groupby('year')['prevalent'].mean()
            yearly_prevalences.append(yearly_prev)

        # Combine all individuals into a single DataFrame and average
        combined = pd.DataFrame(yearly_prevalences).T  # Rows = years, Cols = individuals
        avg_prev_by_age = combined.mean(axis=1)  # Average across individuals

        # Convert to Polars DataFrame
        return pl.DataFrame({
            'age_years': avg_prev_by_age.index.tolist(),
            'prevalence': avg_prev_by_age.values.tolist(),
        })
