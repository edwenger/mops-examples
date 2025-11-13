# emodlib-demo

EMOD malaria model calibration example using ModelOps and Calabaria.

## Quick Start

```bash
# 1. Install dependencies
uv sync

# 2. Generate study with Sobol sampling
uv run python -m modelops_calabaria.cli sampling sobol \
  "models/emod_malaria.py:EmodMalariaModel" \
  --scenario baseline \
  --n-samples 64 \
  --n-replicates 2 \
  --seed 42 \
  --scramble \
  --targets "targets.prevalence_by_age:prevalence_by_age_target" \
  --output study.json

# 3. Manually fix study.json outputs field (temporary workaround)
sed -i '' 's/"outputs": null/"outputs": ["prevalence_by_age"]/' study.json

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
├── pyproject.toml              # Dependencies (emodlib>=0.0.4 from test.pypi.org)
├── models/emod_malaria.py      # EmodMalariaModel with @model_output decorator
├── targets/prevalence_by_age.py # Target with KL divergence loss
├── data/observed_prevalence.csv # Observed prevalence data
└── .modelops-bundle/           # Bundle registry configuration
```

## Model

**EmodMalariaModel** simulates individuals exposed to seasonal malaria transmission:
- **Parameters**: Antigen_Switch_Rate, Falciparum_PfEMP1_Variants, Max_Individual_Infections
- **Output**: prevalence_by_age (prevalence by year, averaged across individuals)
- **Target**: KL divergence loss comparing simulated vs observed prevalence

## Current Status

✅ Bundle building and pushing works
✅ Dependencies install on workers
✅ Simulations execute successfully
❌ **Model outputs not being extracted** - `@model_output` decorated methods not called

See `PROGRESS.md` for detailed debugging notes.
