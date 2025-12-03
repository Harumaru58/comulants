# Cumulant Tensor Model

A Python implementation for tensor decomposition of biomarker data using multiple approaches:
- **Cumulant tensor model**: 3-way tensors (patients × biomarkers × cumulants) with generative latent variable models
- **Moment tensor decomposition**: 3rd and 4th-order empirical moment tensors for group-level analysis
- **Symmetric tensor decomposition**: 28×28×28 symmetric tensors per patient

## Overview

This project provides several tensor decomposition approaches for biomarker data:

### 1. Cumulant Tensor Model
- Each patient has a **latent biomarker state** `z_i` drawn from a distribution
- The distribution is **parameterized by cumulants** `κ = (κ₁, κ₂, κ₃, κ₄)`
- Observed biomarkers are generated from this latent state
- Result: **3-way tensor** `patients × biomarkers × cumulants`

### 2. Moment Tensor Decomposition
- Computes **3rd or 4th-order empirical moments** for each patient group
- Decomposes moment tensors to identify biomarker interaction patterns
- Group-level analysis to compare disease states

### 3. Symmetric Tensor Per Patient
- Creates **28×28×28 symmetric tensors** for each patient
- Captures third-order biomarker interactions at individual level

## Installation

### Prerequisites

- Python 3.8 or higher
- pip

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd comulants
```

2. Create a virtual environment:
```bash
python3 -m venv venv
```

3. Activate the virtual environment:
```bash
# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

4. Install dependencies:
```bash
# Upgrade pip first
pip install --upgrade pip

# Install all requirements from requirements.txt
pip install -r requirements.txt
```

**Required packages:**
- `numpy` >= 1.20.0 - Numerical computations
- `scipy` >= 1.7.0 - Scientific computing
- `pandas` >= 1.3.0 - Data manipulation
- `scikit-learn` >= 1.0.0 - Machine learning utilities
- `tensorly` >= 0.8.0 - Tensor decomposition
- `openpyxl` >= 3.0.0 - Excel file reading
- `matplotlib` >= 3.5.0 - Plotting and visualization
- `seaborn` >= 0.11.0 - Statistical visualizations

**Verify installation:**
```bash
python -c "import numpy, scipy, pandas, sklearn, tensorly, openpyxl, matplotlib, seaborn; print('All packages installed successfully!')"
```

## Usage

### 1. Cumulant Tensor Model

Run the main script to process the CSV data and generate the cumulant tensor:

```bash
python cumulant_tensor_model.py
```

**Programmatic usage:**
```python
from cumulant_tensor_model import parse_csf_data, LatentCumulantTensorModel

# Load data
X, biomarkers, metadata = parse_csf_data("NIpanel_msclin_ATNsharp_20241205(1).csv")

# Initialize model
model = LatentCumulantTensorModel(n_cumulants=4)

# Fit generative model (recommended)
tensor = model.fit_generative(X, latent_dim=3)

# Perform CP decomposition
cp_result = model.decompose_cp(rank=5)

# Create and decompose symmetric biomarker interaction tensor
sym_tensor = model.create_symmetric_biomarker_tensor(X, mode='covariance')
sym_result = model.decompose_symmetric(sym_tensor, rank=5, method='cp')
```

### 2. Moment Tensor Decomposition (Group-Level)

**3rd-order moments:**
```bash
python moments_3rd_order.py
```

**4th-order moments:**
```bash
python moments_4th_order.py
```

Both scripts:
- Read Excel or CSV data grouped by patient groups
- Compute empirical moment tensors for each group
- Perform CP tensor decomposition
- Output: Group-level factor matrices and reconstruction errors

**Example output:**
```
Group 1/4: CU_A-T- - Error: 1.3016
Group 2/4: AD_A+T+ - Error: 1.7188
Group 3/4: CBS_A-T+ - Error: 2.7271
Group 4/4: CBS-AD_A+T+ - Error: 1.8098

Processed 4 groups
Mean error: 1.8893 | Min: 1.3016 | Max: 2.7271
```

### 3. Symmetric Tensor Per Patient

Create 28×28×28 symmetric tensors for each patient:

```bash
python symmetric_tensor_per_patient.py
```

This script:
- Creates a symmetric 28×28×28 tensor for each patient
- Performs tensor decomposition (CP or Tucker)
- Analyzes biomarker interaction patterns at individual level

### 4. Analysis Tools

Analyze and compare decomposition results across groups:

```python
from analyze_decomposition import generate_analysis_report, print_analysis_summary
from moments_3rd_order import main as moments_3rd_main

# Run decomposition
decomps = moments_3rd_main(rank=5)

# Generate comprehensive analysis
report = generate_analysis_report(decomps, biomarker_names=None)

# Print summary
print_analysis_summary(report, biomarker_names)
```

**Analysis capabilities:**
- Compare factor loadings across groups
- Identify shared vs group-specific biomarker patterns
- Cluster biomarkers into modules
- Compute group similarity matrices
- Visualize results (heatmaps, bar plots)

## Data Format

The data file (Excel or CSV) should have:
- **Excel**: Standard format with sheet named "ATN_sharp"
- **CSV**: Semicolon (`;`) delimiter, comma (`,`) decimal separator
- First 5 columns: Metadata (Group, ApoE Pheno, sex, age at LP, Sample)
- Remaining columns: Biomarker measurements (28 biomarkers)

Example CSV structure:
```
Group;ApoE Pheno;sex;age at LP;Sample;APOE_total;APOE2;APOE3;...
CU_A-T-;3/3;w;49;csf109;51,02;0,00;52,55;...
```

## Model Details

### Cumulant Tensor Model

**Generative Model:**
```
x_ij = f(z_i, β_j) + ε_ij
```

where:
- `z_i ~ p(z | κ_i)` is the latent biomarker state for patient i
- `κ_i = (κ₁, κ₂, κ₃, κ₄)` are cumulants parameterizing the distribution
- `β_j` are biomarker loadings
- `x_ij` are observed biomarker values

**Tensor Structure:**
- **Dimension 0 (Patients)**: Each patient's biomarker state distribution
- **Dimension 1 (Biomarkers)**: Different biomarker measurements
- **Dimension 2 (Cumulants)**: Statistical moments (mean, variance, skewness, kurtosis)

**CP Decomposition:**
```
T ≈ Σᵣ aᵣ ⊗ bᵣ ⊗ cᵣ
```

where:
- `aᵣ`: Patient factors (patient phenotypes)
- `bᵣ`: Biomarker factors (biomarker modules)
- `cᵣ`: Cumulant factors (distributional signatures)

### Moment Tensor Decomposition

**3rd-order moments:**
- Computes `M[i,j,k] = E[(x_i - μ_i)(x_j - μ_j)(x_k - μ_k)]`
- Creates symmetric 28×28×28 tensor per group
- Decomposes to identify biomarker interaction patterns

**4th-order moments:**
- Computes `M[i,j,k,l] = E[(x_i - μ_i)(x_j - μ_j)(x_k - μ_k)(x_l - μ_l)]`
- Creates symmetric 28×28×28×28 tensor per group
- Captures higher-order interactions

### Symmetric Tensor Per Patient

- Creates 28×28×28 symmetric tensor for each patient
- Represents third-order biomarker interactions
- Uses Tucker decomposition (better reconstruction than CP for symmetric tensors)

## What Can You Discover?

### Biomarker Modules
- Identify clusters of biomarkers that interact together
- Example: "Complement module" (C1QA, C1QB, C1QC, C3, C4) vs "APOE module"

### Group-Specific Patterns
- Compare AD vs CU vs CBS groups
- Identify disease-specific biomarker interaction patterns
- Discover progression markers

### Shared vs Unique Patterns
- Find biomarkers that are important across all groups (shared)
- Identify biomarkers unique to specific disease states (group-specific)

### Clinical Correlations
- Relate component loadings to:
  - Disease severity
  - Age
  - ApoE genotype
  - Treatment response

## File Structure

```
comulants/
├── README.md
├── requirements.txt
├── LICENSE
├── cumulant_tensor_model.py          # Cumulant tensor model
├── moments_3rd_order.py              # 3rd-order moment decomposition
├── moments_4th_order.py              # 4th-order moment decomposition
├── symmetric_tensor_per_patient.py   # Per-patient symmetric tensors
├── analyze_decomposition.py          # Analysis and comparison tools
├── NIpanel_msclin_ATNsharp_20241205(1).csv  # Data file
├── data.xlsx                         # Excel data file
└── venv/                             # Virtual environment (created after setup)
```

## Available Methods

### LatentCumulantTensorModel
1. **`fit(X)`** - Direct cumulant estimation using method of moments
2. **`fit_generative(X, latent_dim=3)`** - Generative latent variable model (recommended)
3. **`fit_mixture(X, n_components=3)`** - Gaussian mixture model approach
4. **`create_symmetric_biomarker_tensor(X, mode='covariance')`** - Create symmetric biomarker interaction tensor
5. **`decompose_symmetric(symmetric_tensor, rank=5, method='cp')`** - Decompose symmetric tensor
6. **`decompose_cp(rank=5)`** - CP decomposition of cumulant tensor
7. **`generate_samples(n_samples=100)`** - Generate synthetic data

### Analysis Functions
1. **`compare_factors_across_groups(decomps)`** - Compare factor loadings
2. **`analyze_component_structure(decomps)`** - Analyze component structure
3. **`cluster_biomarkers_by_factors(decomps)`** - Cluster biomarkers into modules
4. **`visualize_group_comparison(analysis_results)`** - Create visualizations
5. **`generate_analysis_report(decomps)`** - Comprehensive analysis report

## Notes

- The model standardizes biomarker data within each biomarker for numerical stability
- Missing values are filled with column means
- The generative model uses SVD initialization for latent states
- Tucker decomposition often works better than CP for symmetric 3D tensors
- High reconstruction errors (>100%) are expected for moment tensors with low rank - consider increasing rank or using different decomposition methods

## License

See LICENSE file for details.

## Citation

If you use this code in your research, please cite appropriately.
