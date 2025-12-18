# emodlib-demo

Example calibration of an EMOD malaria model using ModelOps and Calabaria.

## Overview

This example demonstrates:
- Wrapping emodlib's `IntrahostComponent` as a ModelOps-compatible model
- Defining a prevalence-by-age calibration target with MSE loss
- Running parameter sweeps locally and on Kubernetes

## Quick Start

```bash
# Install dependencies
uv sync

# Basic model test (verifies emodlib is working)
uv run python test_model.py

# Full integration test (model + target + parallelism)
uv run python run_local_sweep.py study_small.json 2
```

## Generating New Studies

```bash
# Generate a study with Sobol sampling
uv run cb sampling sobol "models.emod_malaria:EmodMalariaModel" \
  --scenario baseline --n-samples 8 --n-replicates 2 --seed 42 --scramble \
  --targets "targets.prevalence_by_age:prevalence_by_age_target" \
  --output study.json

# Submit to Kubernetes
mops bundle push
mops jobs submit study.json
```

## Project Structure

```
models/emod_malaria.py       # EmodMalariaModel (BaseModel subclass)
targets/prevalence_by_age.py # LossTarget with MSE evaluator
data/observed_prevalence.csv # Observed prevalence by age
generate_observed_data.py    # Script to regenerate observed data
study_small.json             # Pre-generated study for local testing
```

## Documentation

- [ModelOps](https://github.com/InstituteforDiseaseModeling/modelops) - Job submission and orchestration
- [Calabaria](https://github.com/InstituteforDiseaseModeling/calabaria) - Calibration framework
- [emodlib](https://github.com/InstituteforDiseaseModeling/emodlib) - EMOD intrahost malaria model
