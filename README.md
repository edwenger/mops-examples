# ModelOps Examples

Examples demonstrating ModelOps/Calabaria workflows for different modeling frameworks.

## Prerequisites

Before running any example, ensure you have:

1. **Azure CLI** installed and authenticated
   ```bash
   az login
   ```

2. **ModelOps** initialized
   ```bash
   mops init
   ```

3. **Infrastructure** running (if using cloud execution)
   ```bash
   mops infra up
   ```

## Examples

### [emodlib-demo](emodlib-demo/)
EMOD malaria model calibration using prevalence-by-age targets with KL divergence loss.

## General Workflow

Each example follows a similar pattern:

```bash
cd <example-name>

# 1. Install dependencies
uv sync

# 2. Register models and targets
mops bundle register-model models/<model>.py
mops bundle register-target targets/<target>.py

# 3. Generate study
# (using calabaria sampling commands)

# 4. Push bundle
mops bundle push

# 5. Submit job
mops jobs submit study.json
```

See individual example README files for specific commands.
