"""
Test script for EMOD malaria model.

This script verifies that the model can be instantiated, run, and produce
valid output without requiring full ModelOps infrastructure.
"""

from models.emod_malaria import EmodMalariaModel


def test_model_instantiation():
    """Test that the model can be instantiated."""
    print("Testing model instantiation...")
    model = EmodMalariaModel()
    print(f"✓ Model created: {model.__class__.__name__}")
    print(f"✓ Parameter space: {len(model.space.specs)} parameters")
    for spec in model.space.specs:
        print(f"  - {spec.name}")
    return model


def test_model_execution():
    """Test that the model can run a simulation."""
    print("\nTesting model execution...")
    model = EmodMalariaModel()

    # Create a parameter set (use midpoint values manually for testing)
    from modelops_calabaria import ParameterSet
    params = ParameterSet(
        space=model.space,
        values={
            'Antigen_Switch_Rate': 1e-9,
            'Falciparum_PfEMP1_Variants': 850,
            'Max_Individual_Infections': 5,
        }
    )
    print(f"✓ Created test parameters")

    # Build simulation state
    state = model.build_sim(params, {})
    print(f"✓ Built simulation state")
    print(f"  - n_people: {state['n_people']}")
    print(f"  - duration: {state['duration']} days")

    # Run simulation
    print(f"✓ Running simulation (this may take 30-60 seconds)...")
    raw_output = model.run_sim(state, seed=42)
    print(f"✓ Simulation completed")
    print(f"  - Simulated {len(raw_output['individual_results'])} individuals")

    # Extract prevalence-by-age
    print(f"✓ Extracting prevalence-by-age output...")
    df = model.extract_prevalence_by_age(raw_output, seed=42)
    print(f"✓ Extracted output: {df.shape[0]} age groups")
    print(f"\nPrevalence by age (first 10 years):")
    print(df.head(10))

    return df


def main():
    """Run all tests."""
    print("=" * 60)
    print("EMOD Malaria Model Test Suite")
    print("=" * 60)

    try:
        model = test_model_instantiation()
        df = test_model_execution()

        print("\n" + "=" * 60)
        print("✓ All tests passed!")
        print("=" * 60)

    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
