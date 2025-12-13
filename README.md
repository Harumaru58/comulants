# Moment Tensor Decomposition

A Python implementation for tensor decomposition of biomarker data using moment tensor decomposition:
- **Moment tensor decomposition**: 3rd and 4th-order empirical moment tensors for group-level analysis

## Overview

This project provides tensor decomposition approaches for biomarker data:

### Moment Tensor Decomposition
- Computes **3rd or 4th-order empirical moments** for each patient group
- Decomposes moment tensors to identify biomarker interaction patterns
- Group-level analysis to compare disease states

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

### Moment Tensor Decomposition (Group-Level)

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

### Analysis Tools

The `analyze_decomposition.py` script provides comprehensive analysis tools to interpret and compare tensor decomposition results across patient groups. It helps identify biomarker interaction patterns, shared vs group-specific features, and biomarker modules.

**Basic usage:**
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

**Main functions:**

1. **`compare_factors_across_groups(decomps, top_n=10)`**
   - Compares factor loadings across groups to identify shared vs group-specific patterns
   - Computes group similarity using cosine similarity on the first component
   - Identifies top N biomarkers per component for each group
   - Finds biomarkers that appear across multiple groups (shared) vs those unique to specific groups

2. **`analyze_component_structure(decomps, biomarker_names=None)`**
   - Analyzes the structure of each component within groups
   - Identifies top positive and negative loadings per component
   - Computes component magnitude and sparsity
   - Records component weights and relative importance

3. **`cluster_biomarkers_by_factors(decomps, n_clusters=5)`**
   - Clusters biomarkers into modules based on their loading patterns across groups
   - Uses K-means clustering to identify biomarkers that behave similarly
   - Helps discover biomarker modules (e.g., "Complement module", "APOE module")

4. **`visualize_group_comparison(analysis_results, biomarker_names=None, save_path=None)`**
   - Creates visualizations comparing groups:
     - **Similarity matrix heatmap**: Shows how similar groups are based on their first component
     - **Biomarker frequency chart**: Shows how often each biomarker appears across groups

5. **`generate_analysis_report(decomps, biomarker_names=None)`**
   - Generates a comprehensive analysis report combining all analyses
   - Compiles summary statistics (number of groups, mean error, etc.)
   - Returns a complete report dictionary with all analysis results

6. **`print_analysis_summary(report, biomarker_names=None)`**
   - Prints a human-readable summary including:
     - Summary statistics
     - Group similarities (pairwise comparisons)
     - Top biomarkers per group
     - Shared biomarkers across groups

**What you can discover:**
- **Shared patterns**: Biomarkers important across multiple groups
- **Group-specific patterns**: Biomarkers unique to certain disease states
- **Biomarker modules**: Clusters of biomarkers that interact together
- **Group similarities**: Which patient groups have similar biomarker patterns
- **Component structure**: Which biomarkers drive each component in each group

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

### Moment Tensor Decomposition

**3rd-order moments:**
- Computes `M[i,j,k] = E[(x_i - μ_i)(x_j - μ_j)(x_k - μ_k)]`
- Creates symmetric 28×28×28 tensor per group
- Decomposes to identify biomarker interaction patterns

**4th-order moments:**
- Computes `M[i,j,k,l] = E[(x_i - μ_i)(x_j - μ_j)(x_k - μ_k)(x_l - μ_l)]`
- Creates symmetric 28×28×28×28 tensor per group
- Captures higher-order interactions

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
├── moments_3rd_order.py              # 3rd-order moment decomposition
├── moments_4th_order.py              # 4th-order moment decomposition
├── analyze_decomposition.py          # Analysis and comparison tools
├── NIpanel_msclin_ATNsharp_20241205(1).csv  # Data file
├── data.xlsx                         # Excel data file
└── venv/                             # Virtual environment (created after setup)
```

## Available Methods

### Analysis Functions (`analyze_decomposition.py`)

1. **`compare_factors_across_groups(decomps, top_n=10)`**
   - Compares factor loadings across groups
   - Computes group similarity matrices
   - Identifies shared vs group-specific biomarkers
   - Returns top biomarkers per component for each group

2. **`analyze_component_structure(decomps, biomarker_names=None)`**
   - Analyzes structure of each component within groups
   - Identifies top positive/negative loadings
   - Computes component magnitude and sparsity
   - Returns detailed component analysis per group

3. **`cluster_biomarkers_by_factors(decomps, n_clusters=5)`**
   - Clusters biomarkers into modules based on loading patterns
   - Uses K-means to identify biomarker modules
   - Returns cluster assignments and characteristics

4. **`visualize_group_comparison(analysis_results, biomarker_names=None, save_path=None)`**
   - Creates similarity matrix heatmap
   - Generates biomarker frequency bar chart
   - Optionally saves figures to file

5. **`generate_analysis_report(decomps, biomarker_names=None)`**
   - Generates comprehensive analysis combining all functions
   - Returns complete report dictionary with summary statistics

6. **`print_analysis_summary(report, biomarker_names=None)`**
   - Prints human-readable summary of analysis results
   - Displays group similarities, top biomarkers, and shared patterns

## Notes

- The model standardizes biomarker data within each biomarker for numerical stability
- Missing values are filled with column means
- High reconstruction errors (>100%) are expected for moment tensors with low rank - consider increasing rank or using different decomposition methods

## License

See LICENSE file for details.

## Citation

If you use this code in your research, please cite appropriately.
