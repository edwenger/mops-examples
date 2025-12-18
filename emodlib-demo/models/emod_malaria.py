"""
EMOD malaria model for seasonal challenge calibration.

Simulates individuals exposed to seasonal malaria transmission using emodlib's
IntrahostComponent. Uses instance-based config (create_config) for thread-safe
parallel execution.
"""

import datetime as dt
from typing import Any, Dict, Mapping

import numpy as np
import pandas as pd
import polars as pl
from emodlib.malaria import IntrahostComponent, create_config

from modelops_calabaria import (
    BaseModel, ConfigurationSpace, ConfigSpec,
    ParameterSpace, ParameterSpec, model_output
)


def surface_area_biting_function(age_days: float) -> float:
    """Age-dependent biting risk (0.07 at birth, 1.0 at age 20+)."""
    newborn_risk = 0.07
    two_year_old_risk = 0.23

    if age_days < 2 * 365:
        return newborn_risk + age_days * (two_year_old_risk - newborn_risk) / (2 * 365.)
    if age_days < 20 * 365:
        return two_year_old_risk + (age_days - 2 * 365.) * (1 - two_year_old_risk) / ((20 - 2) * 365)
    return 1.0


def month_index_from_day(t: int) -> int:
    """Convert day index (from Jan 1, 2000) to month index (0-11)."""
    y2k = dt.datetime(2000, 1, 1)
    return (y2k + dt.timedelta(days=t)).month - 1


def monthly_eir_challenge(config, duration: int, monthly_eirs: tuple, updates_per_day: int = 2) -> pd.DataFrame:
    """Run one individual through seasonal malaria exposure, returning daily state."""
    ic = IntrahostComponent.create(config)
    ic.susceptibility.age = 0.0

    asexuals = np.zeros(duration)
    gametocytes = np.zeros(duration)
    fevers = np.zeros(duration)

    for t in range(duration):
        daily_eir = monthly_eirs[month_index_from_day(t)] * 12 / 365.0
        daily_eir *= surface_area_biting_function(ic.susceptibility.age)

        if np.random.random() < 1 - np.exp(-daily_eir):
            ic.challenge()

        for _ in range(updates_per_day):
            ic.update(dt=1.0 / updates_per_day)

        asexuals[t] = ic.parasite_density
        gametocytes[t] = ic.gametocyte_density
        fevers[t] = ic.fever_temperature

    return pd.DataFrame({
        'parasite_density': asexuals,
        'gametocyte_density': gametocytes,
        'fever_temperature': fevers,
    })


class EmodMalariaModel(BaseModel):
    """EMOD malaria model computing prevalence-by-age over 20 years of exposure."""

    PARAMS = ParameterSpace((
        ParameterSpec("Antigen_Switch_Rate", 5e-10, 5e-8, "float"),
        ParameterSpec("Falciparum_PfEMP1_Variants", 500, 1200, "int"),
        ParameterSpec("Max_Individual_Infections", 3, 7, "int"),
    ))

    CONFIG = ConfigurationSpace((
        ConfigSpec("n_people", default=10),
        ConfigSpec("duration", default=20 * 365),
        ConfigSpec("updates_per_day", default=2),
        ConfigSpec("monthly_eirs", default=(1, 1, 0.5, 1, 1, 2, 3.875, 7.75, 15.0, 3.875, 1, 1)),
    ))

    def __init__(self):
        super().__init__()

    def build_sim(self, params, config: Mapping[str, Any]) -> Dict[str, Any]:
        """Package parameters and config into simulation state."""
        # Convert to native Python types for pybind11 compatibility
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
        """Run simulation for N individuals with given parameters."""
        # Create thread-safe emodlib config (Run_Number must fit in int32)
        config = create_config({
            'Run_Number': int(seed) % (2**31),
            'infection_params': {'Antigen_Switch_Rate': float(state['antigen_switch_rate'])},
            'Falciparum_PfEMP1_Variants': int(state['pfemp1_variants']),
            'Max_Individual_Infections': int(state['max_infections']),
        })

        np.random.seed(int(seed) % (2**32))

        n_people = int(state['n_people'])
        duration = int(state['duration'])
        individual_results = [
            monthly_eir_challenge(config, duration, tuple(state['monthly_eirs']), int(state['updates_per_day']))
            for _ in range(n_people)
        ]

        return {'individual_results': individual_results, 'n_people': n_people, 'duration': duration}

    @model_output("prevalence_by_age")
    def extract_prevalence_by_age(self, raw_output: Dict[str, Any], seed: int) -> pl.DataFrame:
        """Compute yearly prevalence (parasite_density > 16) averaged across individuals."""
        yearly_prevalences = []
        for df in raw_output['individual_results']:
            df['prevalent'] = (df['parasite_density'] > 16).astype(int)
            df['year'] = df.index // 365
            yearly_prevalences.append(df.groupby('year')['prevalent'].mean())

        combined = pd.DataFrame(yearly_prevalences).T
        avg_prev_by_age = combined.mean(axis=1)

        return pl.DataFrame({
            'age_years': avg_prev_by_age.index.tolist(),
            'prevalence': avg_prev_by_age.values.tolist(),
        })
