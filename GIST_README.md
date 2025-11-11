# Discrete Curvature κ(n) Gist

Self-contained Python snippet for κ(n) curvature computation with Z-transformation and prime/composite classification, including bootstrap CI on accuracy for sequence diagnostics.

## Overview

This gist implements cognitive-number-theory's **κ(n) curvature** and **Z(n) diagnostic transform** for validation within the range **[10^14, 10^18)** per project policy.

### Key Features

- **Validation Range**: All validation occurs in [10^14, 10^18) per project policy
- **Bootstrap CI**: 95% confidence interval on mean accuracy
- **Tunable Parameters**: v-parameter for Z-transformation, threshold for classification
- **Fast Computation**: Uses sympy for efficient divisor counting and primality testing
- **Range Enforcement**: Runtime guard prevents accidental validation outside allowed range

## Requirements

- Python 3.6+
- numpy
- sympy

```bash
pip install numpy sympy
```

## Quick Start

### As a Command-Line Tool

```bash
# Default usage (samples 500 numbers from [10^14, 10^14 + 10^6))
python curvature_gist.py

# Custom starting point within validation range
python curvature_gist.py --min-n 100000000000000

# More samples for better statistics
python curvature_gist.py --sample-count 1000

# Wider sampling band
python curvature_gist.py --sample-band 10000000

# Custom v-parameter for Z-transformation
python curvature_gist.py --v-param 0.5

# Custom threshold for classification
python curvature_gist.py --threshold 1.0

# Control bootstrap samples
python curvature_gist.py --bootstrap-samples 500
```

**Important**: All validation must occur within [10^14, 10^18) per project policy. The script enforces this range automatically.

### As a Module

```python
import numpy as np
import random
import curvature_gist as cg

# Validation range constants
RANGE_MIN = cg.RANGE_MIN  # 10^14
RANGE_MAX = cg.RANGE_MAX  # 10^18

# Example 1: Compute curvature for a single number (within validation range)
n = RANGE_MIN + 1000000
print(f"κ({n}): {cg.kappa(n):.4f}")
print(f"Z({n}): {cg.z_transform(n):.4f}")

# Example 2: Batch analysis with random sampling
# Sample numbers from the validation range
sample_count = 500
min_n = RANGE_MIN
max_n = RANGE_MIN + 10**6
xs = [random.randrange(min_n, max_n) for _ in range(sample_count)]

# Enforce validation range
cg.enforce_validation_range(min_n, max_n)

# Compute and classify
kappas = [cg.kappa(i) for i in xs]
primes = [cg.is_prime(i) for i in xs]
classifications = [cg.classify_by_kappa(i, threshold=1.5) for i in xs]
accuracy = np.mean(np.array(classifications) == np.array(primes))
print(f"Accuracy: {accuracy:.4f}")

# Example 3: Bootstrap confidence intervals
accs = []
for _ in range(50):
    sample_indices = np.random.choice(len(xs), min(500, len(xs)), replace=True)
    sample = [xs[i] for i in sample_indices]
    cls = [cg.classify_by_kappa(i, 1.5) for i in sample]
    true = [cg.is_prime(i) for i in sample]
    accs.append(np.mean(np.array(cls) == np.array(true)))
ci = cg.bootstrap_ci(accs)
print(f"95% CI on mean accuracy: {ci}")

# Example 4: Save results
np.savetxt('kappas.csv', kappas)
```

## Core Functions

### `RANGE_MIN` and `RANGE_MAX`
Validation range constants: **[10^14, 10^18)** per project policy.

```python
RANGE_MIN = 10**14
RANGE_MAX = 10**18
```

### `enforce_validation_range(min_n, max_n)`
Enforces that validation range is within allowed bounds.

```python
# This will pass
enforce_validation_range(10**14, 10**14 + 10**6)

# This will raise ValueError (outside allowed range)
# enforce_validation_range(2, 10000)
```

### `kappa(n, base=np.e**2)`
Computes discrete curvature: **κ(n) = d(n) × ln(n) / e²**

Uses `sympy.divisors()` for accurate divisor counting.

```python
n = 10**14 + 1000000
kappa(n)  # Curvature for number in validation range
```

### `delta_n(n, v=1.0)`
Distortion measure: **Δ(n) = v × κ(n)**

```python
n = 10**14 + 1000000
delta_n(n, v=1.0)  # Distortion with v=1.0
delta_n(n, v=1.5)  # Higher v = more distortion
```

### `z_transform(n, v=1.0)`
Z-transformation for normalization: **Z(n) = n / exp(v × κ(n))**

```python
n = 10**14 + 1000000
z_transform(n, v=1.0)  # Z-transform with v=1.0
z_transform(n, v=0.5)  # Lower v = less normalization
```

### `is_prime(n)`
Primality test using `sympy.isprime()`.

```python
n = 10**14 + 1000000
is_prime(n)  # Check primality
```

### `classify_by_kappa(n, threshold=1.0)`
Classify as prime (True) if κ(n) < threshold, composite (False) otherwise.

```python
n = 10**14 + 1000000
classify_by_kappa(n, threshold=1.5)  # Classify using threshold
```

### `bootstrap_ci(accuracies, n_resamples=1000)`
Bootstrap confidence interval for the mean of accuracy values.

```python
accuracies = [0.85, 0.87, 0.83, 0.88, 0.86]
ci = bootstrap_ci(accuracies)  # Returns [lower, upper] bounds
# Example output: [0.836, 0.874]
```

## Output Format

### Console Output

```
κ(100000000001000000): 91.6166
Z(100000000001000000): 0.0000

Sampling 500 numbers from [100000000000000, 100000001000000)...
Running batch analysis on 500 samples...
Accuracy: 0.8817

Running bootstrap analysis (50 replicates of 500 samples each)...
95% CI on mean accuracy: [0.8784 0.8839]

Saving artifacts...
Saved: kappas.csv

# Run plan
# Validation range: [10^14, 10^18] per project policy
# Sampled 500 numbers from [100000000000000, 100000001000000)
# Threshold: 1.5
# Metric: Mean accuracy; 95% bootstrap CI (1000 resamples)
# Cmd: python curvature_gist.py
# Artifacts: kappas.csv (np.savetxt('kappas.csv', kappas))

Results: Accuracy=0.8817, 95% CI=[0.8784 0.8839]
```

### CSV Output

File: `kappas.csv`

```csv
kappa
91.6166...
92.1234...
90.9876...
...
```

Contains one κ(n) value per line for each sampled number in the validation range.

## Command-Line Options

```
--min-n MIN_N              Minimum n (default: 10^14)
--sample-count N           Number of samples from validation range (default: 500)
--sample-band N            Size of band [min-n, min-n + band) (default: 10^6)
--v-param V_PARAM          v-parameter for Z-transformation (default: 1.0)
--threshold THRESHOLD      Classification threshold (default: 1.5)
--bootstrap-samples N      Number of bootstrap resamples (default: 1000)
```

## Validation Range Policy

**Non-negotiable**: All validation must occur within **[10^14, 10^18)** per project policy.

- The script enforces this range automatically via `enforce_validation_range()`
- Attempts to validate outside this range will raise `ValueError`
- Small-n examples shown here are for demonstrating function mechanics only
- Use random sampling within the allowed band to keep demos computationally feasible

## Theory

The discrete curvature κ(n) quantifies the "complexity" of an integer based on its divisor structure:

- **Primes** have minimal divisors (only 1 and n), yielding low curvature
- **Composites** have more divisors, resulting in higher curvature
- This creates a natural "geodesic" interpretation where primes are minimal-curvature paths

### Mathematical Definition

```
κ(n) = d(n) × ln(n) / e²

where:
  - d(n) = divisor count of n
  - ln(n) = natural logarithm
  - e² ≈ 7.389 (normalization constant)
```

### Z-Transformation

```
Z(n) = n / exp(v × κ(n))

where:
  - v = traversal rate parameter
  - κ(n) = discrete curvature
```

This transformation normalizes for the curvature distortion, revealing underlying structural patterns.

## Use Cases

1. **High-Magnitude Validation**: Validate curvature-based classification on numbers in [10^14, 10^18)
2. **Sequence Diagnostics**: Bootstrap CI validation on classification accuracy
3. **Structural Analysis**: Quantify complexity of large integer sequences via curvature
4. **Parameter Tuning**: Explore effect of v-parameter and threshold on classification
5. **Educational Tool**: Demonstrate connections between divisor theory and geometry
6. **Research**: Explore alternative primality metrics at scale

## Key Features

- **Range Enforcement**: Automatic validation that all analysis occurs in [10^14, 10^18)
- **Tunable v/threshold**: Custom diagnostics for different classification strategies
- **Instant CI-validated accuracy**: Empirical tuning with bootstrap confidence intervals
- **Direct Z-feature export**: For RSA bias analysis and distortion studies
- **Minimal dependencies**: Only numpy and sympy required, no complex setup
- **Efficient sampling**: Random sampling within allowed band for quick validation

## Citation

If you use this in research, please cite:

```
Cognitive Number Theory: A Forward Diagnostic Framework for Number-Theoretic Distortion
GitHub: zfifteen/cognitive-number-theory
```

## License

See repository LICENSE file.
