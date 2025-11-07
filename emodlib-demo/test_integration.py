"""
Integration test: Run model and evaluate against target.

This test verifies the complete workflow:
1. Instantiate model
2. Run simulation
3. Extract model output
4. Load target
5. Align and evaluate simulated vs observed data
"""

from models.emod_malaria import EmodMalariaModel
from targets.prevalence_by_age import prevalence_by_age_target
from modelops_calabaria import ParameterSet


def main():
    """Run full integration test."""
    print("=" * 60)
    print("Model + Target Integration Test")
    print("=" * 60)

    # 1. Create model and run simulation
    print("\n1. Running model simulation...")
    model = EmodMalariaModel()

    params = ParameterSet(
        space=model.space,
        values={
            'Antigen_Switch_Rate': 1e-9,
            'Falciparum_PfEMP1_Variants': 850,
            'Max_Individual_Infections': 5,
        }
    )

    state = model.build_sim(params, {})
    print(f"   Running {state['n_people']} individuals for {state['duration']} days...")
    raw_output = model.run_sim(state, seed=42)
    print(f"   ✓ Simulation complete")

    # 2. Extract model output
    print("\n2. Extracting prevalence-by-age output...")
    simulated = model.extract_prevalence_by_age(raw_output, seed=42)
    print(f"   ✓ Extracted {simulated.shape[0]} age groups")
    print(f"   Simulated prevalence (first 5 ages):")
    print(simulated.head(5))

    # 3. Load target (call the unwrapped function directly)
    print("\n3. Loading target...")
    # The decorator wraps the function, so we need to call it differently
    # For testing, we'll just instantiate the Target directly
    import polars as pl
    from modelops_calabaria.core.target import Target
    from modelops_calabaria.core.alignment import JoinAlignment
    from targets.prevalence_by_age import replicate_mean_kl_divergence

    observed_data = pl.read_csv('data/observed_prevalence.csv')
    target = Target(
        model_output="prevalence_by_age",
        data=observed_data,
        alignment=JoinAlignment(on_cols="age_years", mode="exact"),
        evaluation=replicate_mean_kl_divergence(col="prevalence"),
        weight=1.0
    )
    print(f"   ✓ Target loaded: {target.model_output}")
    print(f"   ✓ Observed data: {target.data.shape[0]} age groups")
    print(f"   Observed prevalence (first 5 ages):")
    print(target.data.head(5))

    # 4. Align data (mimicking what calabaria does)
    print("\n4. Aligning simulated and observed data...")
    # Join on age_years to align
    aligned = target.data.join(simulated, on='age_years', how='inner', suffix='_sim')
    print(f"   ✓ Aligned {aligned.shape[0]} age groups")

    # 5. Evaluate
    print("\n5. Evaluating loss...")
    observed_aligned = aligned.select(['age_years', 'prevalence'])
    simulated_aligned = aligned.select(['age_years', 'prevalence_sim']).rename({'prevalence_sim': 'prevalence'})

    loss = target.evaluation(observed_aligned, simulated_aligned)
    print(f"   ✓ KL divergence loss = {loss:.6f}")

    # 6. Show comparison
    print("\n6. Comparison summary:")
    comparison = aligned.select([
        'age_years',
        'prevalence',  # observed
        'prevalence_sim',  # simulated
    ])
    print(comparison)

    print("\n" + "=" * 60)
    print("✓ Integration test passed!")
    print("=" * 60)
    print(f"\nLoss value: {loss:.6f}")
    print("The model and target are compatible and ready for calibration.")

    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
