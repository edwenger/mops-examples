"""
Test multi-replicate aggregation to debug K8s error.

This mimics what the framework does when evaluating targets with multiple replicates.
"""

import polars as pl
from models.emod_malaria import EmodMalariaModel
from targets.prevalence_by_age import prevalence_by_age_target
from modelops_calabaria import ParameterSet


def main():
    """Test multi-replicate target evaluation."""
    print("=" * 60)
    print("Multi-Replicate Target Evaluation Test")
    print("=" * 60)

    # 1. Create model
    model = EmodMalariaModel()
    params = ParameterSet(
        space=model.space,
        values={
            'Antigen_Switch_Rate': 1e-9,
            'Falciparum_PfEMP1_Variants': 850,
            'Max_Individual_Infections': 5,
        }
    )

    # 2. Run 2 replicates with different seeds
    print("\n1. Running 2 replicates...")
    replicates = []
    for seed in [42, 43]:
        print(f"   Running replicate with seed={seed}...")
        state = model.build_sim(params, {})
        raw_output = model.run_sim(state, seed=seed)
        
        # Extract output - this is what the framework should do
        output = model.extract_prevalence_by_age(raw_output, seed=seed)
        print(f"   ✓ Extracted shape: {output.shape}")
        
        # Store as dict with output name as key
        replicates.append({
            'prevalence_by_age': output
        })

    print(f"\n2. Created {len(replicates)} replicate outputs")
    print(f"   Keys in replicate[0]: {list(replicates[0].keys())}")
    print(f"   Keys in replicate[1]: {list(replicates[1].keys())}")

    # 3. Load target
    print("\n3. Loading target...")
    target = prevalence_by_age_target()
    print(f"   ✓ Target expects output: '{target.model_output}'")

    # 4. Evaluate with multiple replicates (this is what K8s does)
    print("\n4. Evaluating target with multiple replicates...")
    try:
        eval_result = target.evaluate(replicates)
        print(f"   ✓ Evaluation succeeded!")
        print(f"   ✓ Loss = {eval_result.loss:.6f}")
        print(f"   ✓ Used {eval_result.aligned_data.data.shape[0]} data points")
        
        # Show that replicates were aggregated
        print("\n5. Checking replicate aggregation...")
        print("   First replicate prevalence (age 0-4):")
        print(replicates[0]['prevalence_by_age'].head(5))
        print("\n   Second replicate prevalence (age 0-4):")
        print(replicates[1]['prevalence_by_age'].head(5))
        
    except Exception as e:
        print(f"   ✗ Evaluation FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1

    print("\n" + "=" * 60)
    print("✓ Multi-replicate test passed!")
    print("=" * 60)
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
