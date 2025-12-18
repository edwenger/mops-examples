#!/usr/bin/env python
"""
Local sweep runner to test the model with multiple parameter sets in parallel.

This simulates what the ModelOps framework does when running jobs on K8s/Dask,
but runs locally using multiprocessing.
"""
import json
import multiprocessing as mp
import time
import sys
from pathlib import Path

from models.emod_malaria import EmodMalariaModel
from targets.prevalence_by_age import prevalence_by_age_target
from modelops_calabaria import ParameterSet


def run_single_trial(args):
    """Run a single trial with given parameters and seed."""
    param_idx, params_dict, seed = args

    try:
        start_time = time.time()
        print(f"[{time.time():.2f}] Trial {param_idx} (seed={seed}) starting...")

        # Create model instance
        model = EmodMalariaModel()

        # Create parameter set
        params = ParameterSet(
            space=model.space,
            values=params_dict
        )

        # Build and run simulation (pass base_config as the config parameter)
        state = model.build_sim(params, model.base_config)
        raw_output = model.run_sim(state, seed=seed)

        # Extract output
        output = model.extract_prevalence_by_age(raw_output, seed=seed)

        # Evaluate against target
        target = prevalence_by_age_target()
        eval_result = target.evaluate([{'prevalence_by_age': output}])

        elapsed = time.time() - start_time
        print(f"[{time.time():.2f}] Trial {param_idx} (seed={seed}) SUCCESS: loss={eval_result.loss:.6f}, elapsed={elapsed:.1f}s")

        return {
            'param_idx': param_idx,
            'seed': seed,
            'status': 'SUCCESS',
            'loss': eval_result.loss,
            'elapsed': elapsed,
            'params': params_dict,
        }

    except Exception as e:
        elapsed = time.time() - start_time
        print(f"[{time.time():.2f}] Trial {param_idx} (seed={seed}) FAILED after {elapsed:.1f}s: {e}")
        import traceback
        traceback.print_exc()
        return {
            'param_idx': param_idx,
            'seed': seed,
            'status': 'FAILED',
            'error': str(e),
            'elapsed': elapsed,
            'params': params_dict,
        }


def main():
    """Run local sweep from study.json file."""
    # Parse arguments
    study_file = sys.argv[1] if len(sys.argv) > 1 else 'study_small.json'
    n_workers = int(sys.argv[2]) if len(sys.argv) > 2 else 3

    print(f"{'='*70}")
    print(f"LOCAL SWEEP RUNNER")
    print(f"{'='*70}")
    print(f"Study file: {study_file}")
    print(f"Workers: {n_workers}")

    # Load study
    with open(study_file) as f:
        study = json.load(f)

    n_replicates = study.get('n_replicates', 1)
    parameter_sets = study['parameter_sets']

    print(f"Parameter sets: {len(parameter_sets)}")
    print(f"Replicates per set: {n_replicates}")
    print(f"Total trials: {len(parameter_sets) * n_replicates}")
    print(f"{'='*70}\n")

    # Build list of trials (param_idx, params, seed)
    trials = []
    for idx, ps in enumerate(parameter_sets):
        for rep in range(n_replicates):
            seed = 42 + idx * 1000 + rep
            trials.append((idx, ps['params'], seed))

    # Run in parallel
    start_time = time.time()
    with mp.Pool(processes=n_workers) as pool:
        results = pool.map(run_single_trial, trials)
    total_elapsed = time.time() - start_time

    # Summarize results
    print(f"\n{'='*70}")
    print(f"RESULTS (total time: {total_elapsed:.1f}s)")
    print(f"{'='*70}")

    successes = [r for r in results if r['status'] == 'SUCCESS']
    failures = [r for r in results if r['status'] == 'FAILED']

    print(f"\nSuccesses: {len(successes)}/{len(results)}")
    if successes:
        losses = [r['loss'] for r in successes]
        print(f"Loss range: {min(losses):.6f} - {max(losses):.6f}")

        # Show top 3 best losses (lower is better)
        best = sorted(successes, key=lambda r: r['loss'])[:3]
        print("\nTop 3 best parameter sets (lowest loss):")
        for r in best:
            print(f"  Loss={r['loss']:.6f}: {r['params']}")

    if failures:
        print(f"\nFailures: {len(failures)}")
        for r in failures:
            print(f"  Trial {r['param_idx']}: {r['error']}")
        return 1

    print(f"\n{'='*70}")
    print(f"ALL TRIALS COMPLETED SUCCESSFULLY")
    print(f"{'='*70}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
