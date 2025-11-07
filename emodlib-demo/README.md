# emodlib-demo

A ModelOps bundle project demonstrating EMOD malaria model calibration using prevalence-by-age targets.

## Overview

This example adapts the seasonal challenge scenario from emod-demo to work with ModelOps/Calabaria. It demonstrates:

- EMOD `IntrahostComponent` malaria model integration with ModelOps
- Prevalence-by-age target calibration using KL divergence
- Thread-safe parameter management for parallel execution
- Pandas/Polars data processing without xarray dependency

## Model

The `EmodMalariaModel` simulates multiple individuals exposed to seasonal malaria transmission:

- **Parameters**: `Antigen_Switch_Rate`, `Falciparum_PfEMP1_Variants`, `Max_Individual_Infections`
- **Fixed Config**: Population size, duration, monthly EIR pattern
- **Output**: Prevalence by age (year) averaged across individuals and seasons

## Target

The `prevalence_by_age` target compares simulated vs observed prevalence using KL divergence loss.

## Project Structure

```
emodlib-demo/
├── pyproject.toml              # Project configuration with test PyPI index
├── README.md                   # This file
├── models/
│   ├── __init__.py
│   └── emod_malaria.py        # EMOD malaria model implementation
├── targets/
│   ├── __init__.py
│   └── prevalence_by_age.py   # Prevalence target with KL divergence
├── data/
│   └── observed_prevalence.csv # Synthetic observed data
└── generate_observed_data.py   # Script to generate synthetic data

```

## Quick Start

```bash
# Install dependencies (requires test PyPI for emodlib)
uv sync

# Generate synthetic observed data
uv run python generate_observed_data.py

# Test model locally
uv run python -c "from models.emod_malaria import EmodMalariaModel; model = EmodMalariaModel(); print(model)"

# Add to bundle registry
mops-bundle add .

# Create manifest
mops-bundle manifest

# Push to registry
mops-bundle push
```

## Thread Safety Note

EMOD's `IntrahostComponent.set_params()` modifies global state. This model handles thread safety by:

1. Calling `set_params()` once per trial at the start of `run_sim()`
2. Creating all individual instances after parameter configuration
3. Relying on Dask's process-based parallelism for worker isolation

This ensures each trial runs with the correct parameters without race conditions.
