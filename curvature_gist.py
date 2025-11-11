#!/usr/bin/env python3
"""
Self-contained Python snippet for κ(n) curvature computation with Z-transformation 
and prime/composite classification, including bootstrap CI on accuracy for sequence diagnostics.

This gist implements cognitive-number-theory's κ(n) curvature and Z(n) diagnostic transform
for validation within the range [10^14, 10^18) per project policy.

Requirements:
    - numpy
    - sympy

Usage:
    python curvature_gist.py [--min-n MIN_N] [--sample-count N] [--v-param V] [--bootstrap-samples N]
    
Example:
    python curvature_gist.py --min-n 100000000000000 --sample-count 500 --v-param 1.0 --bootstrap-samples 1000
"""

import numpy as np
import sympy as sp
import random

# Validation range constants (non-negotiable project policy)
RANGE_MIN = 10**14
RANGE_MAX = 10**18

# Validation range constants (non-negotiable project policy)
RANGE_MIN = 10**14
RANGE_MAX = 10**18


def enforce_validation_range(min_n: int, max_n: int) -> None:
    """
    Enforce that validation range is within [10^14, 10^18) per project policy.
    
    Args:
        min_n: Minimum value in range
        max_n: Maximum value in range
        
    Raises:
        ValueError: If range is outside allowed bounds
    """
    if not (RANGE_MIN <= min_n < max_n <= RANGE_MAX):
        raise ValueError(
            f"Validation range must be within [{RANGE_MIN}, {RANGE_MAX}] "
            "per project policy. "
            f"Got: [{min_n}, {max_n}]"
        )


def kappa(n, base=np.e**2):
    """κ(n) curvature: d(n) * ln(n) / e²."""
    d_n = len(list(sp.divisors(n)))
    return d_n * np.log(n) / base if n > 1 else 0


def delta_n(n, v=1.0):
    """Distortion: v * κ(n)."""
    return v * kappa(n)


def z_transform(n, v=1.0):
    """Z(n): n / exp(delta_n)."""
    return n / np.exp(delta_n(n, v))


def is_prime(n):
    """Simple primality check."""
    return sp.isprime(n)


def classify_by_kappa(n, threshold=1.0):
    """Classify as prime if κ(n) < threshold."""
    return kappa(n) < threshold


def bootstrap_ci(accuracies, n_resamples=1000):
    """95% bootstrap CI on mean accuracy."""
    resamples = np.random.choice(accuracies, (n_resamples, len(accuracies)), replace=True)
    stats = np.mean(resamples, axis=1)
    return np.percentile(stats, [2.5, 97.5])


# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='κ(n) curvature computation with Z-transformation and prime/composite classification. '
                    'Validation range: [10^14, 10^18) per project policy.'
    )
    parser.add_argument('--min-n', type=int, default=RANGE_MIN,
                       help=f'Minimum value of n (default: {RANGE_MIN})')
    parser.add_argument('--sample-count', type=int, default=500,
                       help='Number of samples to draw from validation range (default: 500)')
    parser.add_argument('--sample-band', type=int, default=10**6,
                       help='Size of band to sample from [min-n, min-n + sample-band) (default: 10^6)')
    parser.add_argument('--v-param', type=float, default=1.0,
                       help='v-parameter for Z-transformation (default: 1.0)')
    parser.add_argument('--threshold', type=float, default=1.5,
                       help='Classification threshold (default: 1.5)')
    parser.add_argument('--bootstrap-samples', type=int, default=1000,
                       help='Number of bootstrap resamples (default: 1000)')
    
    args = parser.parse_args()
    
    # Enforce validation range
    max_n = args.min_n + args.sample_band
    enforce_validation_range(args.min_n, max_n)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # Example usage for single number (within validation range)
    n_demo = RANGE_MIN + 1000000
    print(f"κ({n_demo}): {kappa(n_demo):.4f}")
    print(f"Z({n_demo}): {z_transform(n_demo, v=args.v_param):.4f}")
    print()
    
    # Sample numbers from validation range
    print(f"Sampling {args.sample_count} numbers from [{args.min_n}, {max_n})...")
    seq = [random.randrange(args.min_n, max_n) for _ in range(args.sample_count)]
    
    # Batch analysis
    print(f"Running batch analysis on {len(seq)} samples...")
    kappas = [kappa(i) for i in seq]
    primes = [is_prime(i) for i in seq]
    classifications = [classify_by_kappa(i, threshold=args.threshold) for i in seq]
    accuracy = np.mean(np.array(classifications) == np.array(primes))
    print(f"Accuracy: {accuracy:.4f}")
    print()
    
    # Bootstrap on batches (50 replicates)
    print(f"Running bootstrap analysis (50 replicates of {min(500, len(seq))} samples each)...")
    accs = []
    bootstrap_sample_size = min(500, len(seq))
    for _ in range(50):
        sample_indices = np.random.choice(len(seq), bootstrap_sample_size, replace=True)
        sample = [seq[i] for i in sample_indices]
        cls = [classify_by_kappa(i, args.threshold) for i in sample]
        true = [is_prime(i) for i in sample]
        accs.append(np.mean(np.array(cls) == np.array(true)))
    ci = bootstrap_ci(accs, n_resamples=args.bootstrap_samples)
    print(f"95% CI on mean accuracy: {ci}")
    print()
    
    # Save artifacts
    print("Saving artifacts...")
    np.savetxt('kappas.csv', kappas, delimiter=',', header='kappa', comments='')
    print("Saved: kappas.csv")
    print()
    
    # Run plan
    print("# Run plan")
    print(f"# Validation range: [{RANGE_MIN}, {RANGE_MAX}] per project policy")
    print(f"# Sampled {args.sample_count} numbers from [{args.min_n}, {max_n})")
    print(f"# Threshold: {args.threshold}")
    print("# Metric: Mean accuracy; 95% bootstrap CI (1000 resamples)")
    print("# Cmd: python curvature_gist.py")
    print("# Artifacts: kappas.csv (np.savetxt('kappas.csv', kappas))")
    print()
    print(f"Results: Accuracy={accuracy:.4f}, 95% CI={ci}")


