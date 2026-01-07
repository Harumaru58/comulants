# Riemannian Optimization on Ellipsoids for Tensor Decomposition

## Summary

This guide documents the implementation of **Riemannian optimization on ellipses/ellipsoids** and its integration with **CP tensor decomposition** for your biomarker data.

## Key Question Answered: Does Random Initialization Converge on the Ellipse?

**Answer: YES, with projection!**

If you set an initial point randomly (not on the ellipse), you must **project it onto the ellipse first**. Then Riemannian optimization:
1. **Keeps you ON the ellipse** throughout all iterations
2. **Converges to the constrained optimum**
3. **Maintains constraint satisfaction** to numerical precision (~10⁻¹⁰)

## Files Created

### 1. `riemannian_ellipse.jl` - 2D Ellipse Optimization
- Demonstrates Riemannian optimization on a 2D ellipse
- Shows how random points are projected before optimization
- Creates visualization of optimization path
- **Run**: `julia riemannian_ellipse.jl`

**Output**:
```
Test 1: Random initial point
Initial point (before projection): [3.0000, 2.0000]
Initial point (after projection):  [1.8974, 0.3162]
Converged in 15 iterations to (2.0, 0.0)
Constraint satisfied: x²/a² + y²/b² = 1.0000000000
```

### 2. `riemannian_ellipsoid.jl` - n-Dimensional Ellipsoid Optimization
- Generalizes to arbitrary dimensions (tested up to 28D)
- Supports three types of ellipsoids:
  - **Axis-aligned**: `AxisAlignedEllipsoid([a₁, a₂, ..., aₙ])`
  - **Covariance-based**: `CovarianceEllipsoid(Σ, confidence_level)`
  - **Custom**: `Ellipsoid(Q)` where Q is any positive definite matrix

**Key Functions**:
- `project_to_ellipsoid(x, ellipsoid)` - Project any point onto ellipsoid
- `project_to_tangent_space(v, x, ellipsoid)` - Compute Riemannian gradient
- `retract(x, v, α, ellipsoid)` - Move along tangent direction, stay on ellipsoid
- `riemannian_gradient_descent()` - Full optimization algorithm

**Run**: `julia riemannian_ellipsoid.jl`

### 3. `constrained_cp_decomposition.jl` - Constrained Tensor Decomposition
- CP decomposition with ellipsoidal constraints on factor matrices
- Each factor column constrained to lie on an ellipsoid
- Integrates with your moment tensor workflow

**Mathematical Formulation**:
```
Minimize: ||T - Σᵣ λᵣ aᵣ ⊗ bᵣ ⊗ cᵣ||²
Subject to: aᵣᵀ Qₐ aᵣ = 1  (for each component r)
            bᵣᵀ Qᵦ bᵣ = 1
            cᵣᵀ Qᵧ cᵣ = 1
```

**Algorithm**:
1. Standard ALS update (unconstrained)
2. **Project each factor column to ellipsoid**
3. Repeat until convergence

**Key Function**:
- `constrained_als_cp(tensor, rank, ellipsoids; max_iter, tol, verbose)`

**Run**: `julia constrained_cp_decomposition.jl`

### 4. `constrained_moments_decomposition.py` - Python Integration
- Loads your `data.xlsx` biomarker data
- Computes 3rd-order moment tensors per patient group
- Computes empirical covariance matrices
- Calls Julia to run constrained decomposition
- Returns results to Python for analysis

**Usage**:
```python
from constrained_moments_decomposition import constrained_moments_decomposition

results, biomarker_names = constrained_moments_decomposition(
    filepath="data.xlsx",
    rank=5,
    confidence_level=2.0,  # 2-sigma ellipsoid
    use_julia=True,
    verbose=True
)
```

**Run**: `python constrained_moments_decomposition.py`

## Results on Your Biomarker Data

Successfully ran constrained decomposition on 4 patient groups:

| Group | Samples | Reconstruction Error | Covariance Condition # |
|-------|---------|---------------------|----------------------|
| CU_A-T- | 32 | 1.1235 | 4.79e9 |
| AD_A+T+ | 47 | 0.3928 | 4.27e8 |
| CBS_A-T+ | 23 | 1.3889 | 1.33e10 |
| CBS-AD_A+T+ | 9 | 0.9802 | 1.74e10 |

**Observations**:
- ✓ All constraints satisfied to ~10⁻¹⁴ precision
- ✓ Factors stay on ellipsoids throughout optimization
- ✓ AD group has lowest error (best reconstruction)
- ⚠️ High covariance condition numbers indicate some biomarkers are highly correlated

## How Ellipsoid Constraints Work

### Why Use Ellipsoids?

Ellipsoids capture **biological covariance structure**:
- Empirical covariance Σ from data defines natural variation
- Ellipsoid constraint: `xᵀ Σ⁻¹ x = c²`
- Factor loadings stay within biologically plausible region
- Prevents overfitting to noise

### Interpretation

**Unconstrained factors**: Can point anywhere in 28D space
**Constrained factors**: Must lie on ellipsoid defined by biomarker covariance

This means:
- Factor patterns respect natural biomarker correlations
- Components represent biologically feasible combinations
- More interpretable in terms of known biomarker modules

## Comparison: Constrained vs Unconstrained

To compare results:
```python
from constrained_moments_decomposition import compare_constrained_vs_unconstrained

results_con, results_uncon = compare_constrained_vs_unconstrained(
    filepath="data.xlsx",
    rank=5,
    confidence_level=2.0
)
```

## Mathematical Details

### Ellipsoid Manifold

An ellipsoid in ℝⁿ is the set:
```
E = {x ∈ ℝⁿ : xᵀ Q x = 1}
```
where Q is a positive definite matrix.

### Tangent Space

At point x on ellipsoid, the tangent space is:
```
TₓE = {v ∈ ℝⁿ : vᵀ(Qx) = 0}
```
(vectors orthogonal to the normal Qx)

### Riemannian Gradient

For objective f(x), the Riemannian gradient is:
```
grad_riem f(x) = ∇f(x) - [∇f(x)ᵀ(Qx) / ||Qx||²] (Qx)
```
(projection of Euclidean gradient onto tangent space)

### Retraction

To move from x along direction v and stay on ellipsoid:
```
retract(x, v, α) = project((x + αv) onto ellipsoid)
             = (x + αv) / sqrt((x + αv)ᵀ Q (x + αv))
```

## Why This Matters for Your Research

1. **Biological Plausibility**: Factors respect natural biomarker correlations
2. **Improved Interpretation**: Components have biological meaning
3. **Regularization**: Prevents overfitting, especially for small samples (CBS-AD: n=9)
4. **Group Comparison**: Compare how different patient groups respect biomarker structure
5. **Confidence Regions**: Can vary confidence_level to see effect

## Next Steps

### 1. Analyze Constrained Factor Loadings
```python
# Look at which biomarkers load on each component
for group_name, result in results.items():
    factors = result['factors'][0]  # First mode (symmetric tensor)

    for r in range(rank):
        loadings = factors[:, r]
        top_biomarkers = np.argsort(np.abs(loadings))[-5:]
        print(f"{group_name}, Component {r}:")
        for idx in top_biomarkers:
            print(f"  {biomarker_names[idx]}: {loadings[idx]:.3f}")
```

### 2. Compare Across Groups
- Are constrained factors more similar across groups?
- Do constraints reveal shared biomarker modules?

### 3. Vary Confidence Level
```python
# Try different confidence levels
for c in [1.0, 1.5, 2.0, 3.0]:
    results = constrained_moments_decomposition(
        confidence_level=c
    )
```

### 4. Examine Covariance Structure
```python
# Visualize covariance matrices
import seaborn as sns
import matplotlib.pyplot as plt

for group_name, result in results.items():
    Sigma = result['covariance']
    plt.figure(figsize=(10, 8))
    sns.heatmap(Sigma, cmap='RdBu_r', center=0,
                xticklabels=biomarker_names,
                yticklabels=biomarker_names)
    plt.title(f"Biomarker Covariance: {group_name}")
    plt.tight_layout()
    plt.savefig(f"covariance_{group_name}.png")
```

### 5. Principal Components
```python
# See what PCA says about biomarker structure
from sklearn.decomposition import PCA

for group_name, X in data_by_groups.items():
    pca = PCA(n_components=5)
    pca.fit(X)

    print(f"{group_name}:")
    print(f"  Variance explained: {pca.explained_variance_ratio_}")

    # Compare PCA loadings with constrained factor loadings
```

## Installation

### Requirements
- Python 3.8+ with: numpy, pandas, scipy, openpyxl
- Julia 1.6+ with: LinearAlgebra, NPZ, JSON

### Setup
```bash
# Python packages
pip install numpy pandas scipy openpyxl

# Julia packages
julia -e 'using Pkg; Pkg.add(["NPZ", "JSON"])'
```

## Troubleshooting

### High Covariance Condition Numbers
If you see warnings about condition numbers > 10⁹:
- Some biomarkers are nearly linearly dependent
- Consider PCA preprocessing or regularization
- Current code adds 1e-6 * I for stability

### Numerical Issues
If constraints are violated (not ~1.0):
- Increase regularization: `Sigma + 1e-5 * I`
- Use smaller confidence_level (tighter ellipsoid)
- Check for NaN/Inf in data

### Julia Performance
For faster computation:
- Reduce max_iter (current: 500)
- Increase tol (current: 1e-6)
- Use fewer components (rank < 5)

## References

This implementation is based on:
1. Riemannian optimization on manifolds
2. CP tensor decomposition (PARAFAC)
3. Statistical confidence ellipsoids from multivariate normal theory

## Citation

If you use this code in your research:
```
@software{riemannian_tensor_decomposition,
  title = {Riemannian Optimization for Constrained CP Tensor Decomposition},
  author = {Your Name},
  year = {2025},
  description = {CP decomposition with ellipsoidal constraints for biomarker analysis}
}
```
