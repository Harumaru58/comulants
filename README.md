# Cumulant Tensor Model

A Python implementation for creating 3-way symmetric tensors from biomarker data using a generative latent variable model. This project conducts CP (CANDECOMP/PARAFAC) tensor decomposition on medical datasets with biomarkers, where each patient's biomarker state is parameterized by cumulants.

## Overview

This project implements a generative model where:
- Each patient has a **latent biomarker state** `z_i` drawn from a distribution
- The distribution is **parameterized by cumulants** `κ = (κ₁, κ₂, κ₃, κ₄)`
- Observed biomarkers are generated from this latent state
- The result is a **3-way tensor**: `patients × biomarkers × cumulants`

The tensor `T[i,j,k]` represents the k-th cumulant of biomarker j for patient i, where:
- **κ₁**: Mean (first cumulant)
- **κ₂**: Variance (second cumulant)
- **κ₃**: Skewness (third cumulant)
- **κ₄**: Kurtosis (fourth cumulant)

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

**Verify installation:**
```bash
python -c "import numpy, scipy, pandas, sklearn, tensorly, openpyxl; print('All packages installed successfully!')"
```

## Usage

### Basic Usage

Run the main script to process the CSV data and generate the cumulant tensor:

```bash
python cumulant_tensor_model.py
```

This will:
1. Load the biomarker data from the CSV file
2. Create a 3-way tensor using the generative model
3. Display summary statistics
4. Perform CP tensor decomposition (if tensorly is available)

### Using the Model Programmatically

```python
from cumulant_tensor_model import parse_csf_data, LatentCumulantTensorModel

# Load data
X, biomarkers, metadata = parse_csf_data("NIpanel_msclin_ATNsharp_20241205(1).csv")

# Initialize model
model = LatentCumulantTensorModel(n_cumulants=4)

# Fit generative model (recommended)
tensor = model.fit_generative(X, latent_dim=3)

# Access the tensor
print(f"Tensor shape: {tensor.shape}")  # (n_patients, n_biomarkers, n_cumulants)

# Perform CP decomposition
cp_result = model.decompose_cp(rank=5)
patient_factors = cp_result['patient_factors']
biomarker_factors = cp_result['biomarker_factors']
cumulant_factors = cp_result['cumulant_factors']

# Create and decompose symmetric 28×28 biomarker interaction tensor
sym_tensor = model.create_symmetric_biomarker_tensor(X, mode='covariance')
sym_result = model.decompose_symmetric(sym_tensor, rank=5, method='cp')
biomarker_factors_sym = sym_result['factors']  # (28, 5)

# For 3D symmetric tensor (28×28×4 cumulant interactions)
sym_tensor_3d = model.create_symmetric_biomarker_tensor(X, mode='cumulant_all')
sym_result_3d = model.decompose_symmetric(sym_tensor_3d, rank=5, method='cp')

# Generate synthetic samples
samples = model.generate_samples(n_samples=100)
```

### Available Methods

The `LatentCumulantTensorModel` class provides several fitting methods:

1. **`fit(X)`** - Direct cumulant estimation using method of moments
2. **`fit_generative(X, latent_dim=3)`** - Generative latent variable model (recommended)
3. **`fit_mixture(X, n_components=3)`** - Gaussian mixture model approach
4. **`create_symmetric_biomarker_tensor(X, mode='covariance')`** - Create symmetric 28×28 biomarker interaction tensor
5. **`decompose_symmetric(symmetric_tensor, rank=5, method='cp')`** - Decompose symmetric tensor using tensorly

## Data Format

The CSV file should have:
- **Semicolon (`;`) as delimiter**
- **Comma (`,`) as decimal separator**
- First row(s): Column headers
- Metadata columns: Group, ApoE Pheno, sex, age at LP, Sample
- Remaining columns: Biomarker measurements

Example structure:
```
Group;ApoE Pheno;sex;age at LP;Sample;APOE_total;APOE2;APOE3;...
CU_A-T-;3/3;w;49;csf109;51,02;0,00;52,55;...
```

## Model Details

### Generative Model

The generative model assumes:

```
x_ij = f(z_i, β_j) + ε_ij
```

where:
- `z_i ~ p(z | κ_i)` is the latent biomarker state for patient i
- `κ_i = (κ₁, κ₂, κ₃, κ₄)` are cumulants parameterizing the distribution
- `β_j` are biomarker loadings
- `x_ij` are observed biomarker values

### Tensor Structure

The resulting tensor has three dimensions:
- **Dimension 0 (Patients)**: Each patient's biomarker state distribution
- **Dimension 1 (Biomarkers)**: Different biomarker measurements
- **Dimension 2 (Cumulants)**: Statistical moments (mean, variance, skewness, kurtosis)

### CP Decomposition

The tensor can be decomposed using CP (CANDECOMP/PARAFAC) decomposition:

```
T ≈ Σᵣ aᵣ ⊗ bᵣ ⊗ cᵣ
```

where:
- `aᵣ`: Patient factors (patient phenotypes)
- `bᵣ`: Biomarker factors (biomarker modules)
- `cᵣ`: Cumulant factors (distributional signatures)

### Symmetric Tensor Decomposition

The model also supports symmetric tensor decomposition for biomarker interactions:

1. **2D Symmetric Matrix (28×28)**: Covariance or correlation matrix of biomarkers
   - Decomposition: `T ≈ Σᵣ λᵣ vᵣ ⊗ vᵣ`
   - Where `vᵣ` are biomarker factors

2. **3D Symmetric Tensor (28×28×4)**: Cumulant interaction tensor
   - Decomposition: `T ≈ Σᵣ λᵣ vᵣ ⊗ vᵣ ⊗ cᵣ`
   - Where `vᵣ` are biomarker factors and `cᵣ` are cumulant factors

Available modes for symmetric tensor creation:
- `'covariance'`: Covariance matrix of biomarkers (28×28)
- `'correlation'`: Correlation matrix of biomarkers (28×28)
- `'cumulant_all'`: Cumulant interaction tensor (28×28×4)

Decomposition methods:
- `'cp'`: CANDECOMP/PARAFAC decomposition
- `'tucker'`: Tucker decomposition (for 3D tensors)

## Output

The script produces:
- **Tensor shape**: Dimensions of the 3-way tensor
- **Summary statistics**: Mean and standard deviation of each cumulant
- **CP decomposition factors**: Patient, biomarker, and cumulant factors
- **Symmetric tensor decomposition**: 28×28 biomarker interaction matrices
- **Example cumulants**: Sample values for a specific patient-biomarker pair

Example output:
```
Tensor shape (generative): (111, 28, 4)
  - Patients: 111
  - Biomarkers: 28
  - Cumulants: 4

Mean cumulant values across all patients and biomarkers:
  κ₁ (mean): 0.0000 ± 0.7953
  κ₂ (variance): 0.4675 ± 1.2313
  κ₃ (skewness): -0.0188 ± 0.5084
  κ₄ (kurtosis): -2.6612 ± 0.3054

Symmetric Tensor Decomposition (28×28 biomarker interactions):
  Covariance matrix shape: (28, 28)
  Cumulant tensor shape: (28, 28, 4)
  
  2D symmetric matrix decomposition:
    Biomarker factors shape: (28, 5)
    Reconstruction error: 0.156599
    
  3D symmetric tensor decomposition:
    Biomarker factors shape: (28, 5)
    Cumulant factors shape: (4, 5)
    Reconstruction error: 0.154828
```

## Dependencies

- `numpy` >= 1.20.0
- `scipy` >= 1.7.0
- `pandas` >= 1.3.0
- `scikit-learn` >= 1.0.0
- `tensorly` >= 0.8.0 (for CP decomposition)

## File Structure

```
comulants/
├── README.md
├── requirements.txt
├── cumulant_tensor_model.py
├── NIpanel_msclin_ATNsharp_20241205(1).csv
└── venv/                    # Virtual environment (created after setup)
```

## Notes

- The model standardizes biomarker data within each biomarker for numerical stability
- Missing values are filled with column means
- The generative model uses SVD initialization for latent states
- CP decomposition requires the `tensorly` package

## License

See LICENSE file for details.

## Citation

If you use this code in your research, please cite appropriately.
