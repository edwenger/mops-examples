# emodlib-demo

EMOD malaria model calibration example using ModelOps and Calabaria.

This example demonstrates how to calibrate a malaria transmission model using the emodlib library with ModelOps infrastructure.

## Quick Start

```bash
# 1. Install dependencies
uv sync

# 2. Run local tests (recommended before submitting)
python test_integration.py   # Test model + target integration
python preflight_check.py    # Validate model outputs and targets

# 3. Generate study with Sobol sampling
uv run cb sampling sobol \
  "models.emod_malaria:EmodMalariaModel" \
  --scenario baseline \
  --n-samples 8 \
  --n-replicates 2 \
  --seed 42 \
  --scramble \
  --targets "targets.prevalence_by_age:prevalence_by_age_target" \
  --output study.json

# 4. Push bundle to registry
mops bundle push

# 5. Submit job
mops jobs submit study.json

# 6. Monitor execution
mops jobs list
mops jobs status <job-id>
kubectl -n modelops-dask-dev logs job/<job-id>
```

## Project Structure

```
emodlib-demo/
├── pyproject.toml               # Dependencies (emodlib>=0.0.4 from test.pypi.org)
├── models/emod_malaria.py       # EmodMalariaModel with @model_output decorator
├── targets/prevalence_by_age.py # Target with Binomial NLL loss function
├── data/observed_prevalence.csv # Observed data: age_years, number_positive, number_total
├── test_integration.py          # Local integration test (model + target)
├── preflight_check.py           # Pre-submission validation script
└── .modelops-bundle/            # Bundle registry configuration
```

## Model

**EmodMalariaModel** simulates individuals exposed to seasonal malaria transmission using the emodlib C++ library:

- **Parameters** (calibrated via Sobol sampling):
  - `Antigen_Switch_Rate`: Rate of antigenic variation (1e-9 to 5e-8)
  - `Falciparum_PfEMP1_Variants`: Number of PfEMP1 variants (500 to 1200)
  - `Max_Individual_Infections`: Maximum concurrent infections per individual (3 to 7)

- **Simulation**:
  - 10 individuals aged 0-20 years
  - 20 years duration (7300 days)
  - Seasonal EIR pattern with peak transmission May-October
  - Tracks parasite density, gametocytes, and fever

- **Model Output**: `prevalence_by_age`
  - Prevalence (fraction parasite-positive) by age year
  - Averaged across individuals and days within each year
  - Format: Polars DataFrame with columns `[age_years, prevalence]`

## Target

**prevalence_by_age_target** compares simulated prevalence against observed survey data:

- **Observed Data** (`data/observed_prevalence.csv`):
  - `age_years`: Age in years (0-19)
  - `number_positive`: Number of RDT-positive individuals
  - `number_total`: Number of individuals tested
  - Example: Age 0 has 20 positive out of 100 tested (20% prevalence)

- **Loss Function**: Binomial Negative Log-Likelihood (BinomialNLL)
  - Compares simulated prevalence (probability `p`) against observed counts (`x` successes in `n` trials)
  - Lower loss = better fit to observed data

- **Evaluation Strategy**:
  - Alignment: Join on `age_years`
  - Aggregation: Identity (no averaging needed for observed counts)
  - Reduction: Mean NLL across all age groups

## Testing Locally

Before submitting jobs to Kubernetes, validate your setup locally:

```bash
# Test that model runs and target evaluates
python test_integration.py
# Output: Runs simulation, evaluates BinomialNLL loss, shows comparison

# Validate model outputs and target registration
python preflight_check.py
# Output: Checks @model_output decorators, tests extraction, validates registry
```

## Current Status

✅ Model simulations working locally
✅ Target evaluation working with BinomialNLL loss
✅ Bundle building and pushing works
✅ Local integration tests passing
⏳ **Debugging target registration in K8s** - waiting for fixes to `mops bundle status`

See `PROGRESS.md` for detailed debugging notes.
