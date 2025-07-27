# Cognitive Model: A Forward Diagnostic Framework for Number-Theoretic Distortion

## Overview

This repository presents a theoretical and computational framework for analyzing discrete integer sequences through a geometry-inspired "curvature" model. By drawing a pedagogical analogy to relativistic distortions, we define a **forward diagnostic map** that highlights structural irregularities—especially those arising from divisor density. This model is intended for **structural analysis**, not for blind inversion of unknown values.

## Key Concepts

1. **Curvature Function**

   $$
   \kappa(n) = \frac{d(n) \cdot \ln(n)}{e^2}
   $$

   * **d(n)**: Divisor count of $n$ (i.e., $\sigma_0(n)$).
   * **ln(n)**: Natural logarithm of $n$.
   * **Normalization**: Constant $e^2$ determined empirically.
   * **Interpretation**: Higher divisor counts and larger values yield greater local "curvature".

2. **Distortion Mapping (Forward Model)**

   $$
   \Delta_n = v \cdot \kappa(n)
   $$

   * **v**: A user-defined "traversal rate" parameter (e.g., cognition or iteration speed).
   * **$\Delta_n$**: Modeled distortion at $n$.
   * **Purpose**: Encodes how rapid progression through integers skews apparent structure.

3. **Perceived Value**

   $$
   n_{\text{perceived}} = n \times \exp\bigl(\Delta_n\bigr)
   $$

   * Applies exponential scaling to the true integer based on $\Delta_n$.
   * Emphasizes how distortion amplifies structural irregularities in composites.

4. **Z-Transformation (Context-Dependent Normalization)**

   $$
   Z(n) \;=\; \frac{n}{\exp\bigl(v \cdot \kappa(n)\bigr)}
   $$

   * **Forward diagnostic use only**: Assumes knowledge of $n$ and $v$ to normalize distortion.
   * **Outcome**: Reveals underlying structural stability, particularly for primes where $\kappa(n)$ is minimal.

## Empirical Validation

* **Prime vs. Composite Curvature (n = 2–49)**

  * Prime average curvature: \~0.739
  * Composite average curvature: \~2.252
  * Ratio: Composites ≈3.05× higher curvature

* **Classification Test**

  * Simple threshold on $\kappa(n)$ yields \~83% accuracy distinguishing primes from composites.

These results demonstrate that primes appear as "minimal-curvature geodesics" within the discrete sequence, providing a quantitative diagnostic measure of number complexity.

## Implementation

* **Language**: Python 3
* **Modules**:

  * `divisor_density.py`: Efficient divisor-counting and curvature computation.
  * `distortion_model.py`: Functions for $\Delta_n$, perceived values, and Z-transformation.
  * `analysis.py`: Scripts to generate statistics, plots, and CSV exports.

### Example Usage

```bash
python analysis.py --max-n 10000 --rate 1.0 --output results.csv
```

Generates curvature statistics and writes them to `results.csv`.

## Limitations & Scope

1. **Forward Diagnostic Only**

   * The Z-transformation **requires** known $n$ and rate $v$. It **does not** serve as a standalone inverse to recover unknown integers from perceived values.
2. **Context-Dependent Parameters**

   * Parameters like $v$ (traversal rate) must be set or estimated; values are not inferred solely from data.
3. **Metaphorical Analogy**

   * References to relativity and geodesics are pedagogical. The core mathematics stands independently of physical interpretations.

## Future Directions

* **Parameter Estimation**: Explore data-driven methods to approximate traversal rates from observed distortions.
* **Enhanced Classification**: Integrate curvature features into machine-learning classifiers for primality testing.
* **Theoretical Extensions**: Investigate connections between divisor-based curvature and deeper analytic number theory.

