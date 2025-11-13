"""
Integration test: Run model and evaluate against target.

This test verifies the complete workflow:
1. Instantiate model
2. Run simulation
3. Extract model output
4. Load target
5. Align and evaluate simulated vs observed data
"""

import polars as pl
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

    # 3. Load target by calling the decorated function
    print("\n3. Loading target...")
    from targets.prevalence_by_age import prevalence_by_age_target

    target = prevalence_by_age_target()
    print(f"   ✓ Target loaded: {target.model_output}")
    print(f"   ✓ Observed data: {target.data.shape[0]} age groups")
    print(f"   Observed data (first 5 ages):")
    print(target.data.head(5))

    # 4. Evaluate target against simulation
    print("\n4. Evaluating target...")
    # The target.evaluate() method expects replicated simulation outputs
    # Format: Sequence[Mapping[output_name, DataFrame]]
    # Since we have one replicate, we pass a list with one dict
    replicated_outputs = [
        {
            'prevalence_by_age': simulated
        }
    ]

    eval_result = target.evaluate(replicated_outputs)
    print(f"   ✓ Binomial NLL loss = {eval_result.loss:.6f}")
    print(f"   ✓ Loss breakdown:")
    print(f"      - Used {eval_result.aligned_data.data.shape[0]} data points")

    # 5. Show sample comparison
    print("\n5. Sample comparison (first 5 age groups):")
    # Manually join for display
    simulated_renamed = simulated.rename({'prevalence': 'sim_prevalence'})
    joined = target.data.join(simulated_renamed, on='age_years', how='inner')

    comparison = joined.select([
        'age_years',
        pl.col('number_positive'),
        pl.col('number_total'),
        (pl.col('number_positive') / pl.col('number_total')).alias('obs_prevalence'),
        pl.col('sim_prevalence'),
    ]).head(5)
    print(comparison)

    print("\n   Target uses BinomialNLL to compare:")
    print("   - sim_prevalence (probability p from simulation)")
    print("   - number_positive (x successes from observations)")
    print("   - number_total (n trials from observations)")

    print("\n" + "=" * 60)
    print("✓ Integration test passed!")
    print("=" * 60)
    print(f"\nFinal loss: {eval_result.loss:.6f}")
    print("The model and target are fully integrated and ready for calibration.")

    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
