# PMF Analysis Package

This package performs Positive Matrix Factorization (PMF) analysis on environmental datasets. It has been updated to work with the data files stored in the `data/` folder and automatically saves all outputs to a separate `output/` folder.

## Directory Structure

```
pmf/
├── data/                          # Input data files
│   ├── Dataset-BatonRouge-con.csv    # Baton Rouge concentrations
│   ├── Dataset-BatonRouge-unc.csv    # Baton Rouge uncertainties
│   ├── Dataset-StLouis-con.csv       # St. Louis concentrations
│   ├── Dataset-StLouis-unc.csv       # St. Louis uncertainties
│   ├── Dataset-Baltimore_con.txt     # Baltimore concentrations
│   ├── Dataset-Baltimore_unc.txt     # Baltimore uncertainties
│   └── ...
├── output/                        # All analysis outputs (auto-created)
│   ├── [dataset]_factor_contributions.csv
│   ├── [dataset]_factor_profiles.csv
│   ├── [dataset]_*.png              # Various visualizations
│   └── ...
├── src/pmf/                       # PMF algorithm implementation
├── mwe.py                         # Single dataset analysis script
├── analyze_all_datasets.py       # Comprehensive analysis script
└── README.md                      # This file
```

## Available Datasets

The package currently supports three datasets:

1. **BatonRouge**: Air quality data with 41 chemical species and 307 time points
2. **StLouis**: Environmental data with 13 chemical species and 418 time points
3. **Baltimore**: PM2.5 data with 26 chemical species and 657 time points

## Usage

### Option 1: Analyze a Single Dataset

Edit `mwe.py` and change the `dataset_name` variable to analyze a specific dataset:

```python
# Change this line to select your dataset:
dataset_name = "BatonRouge"  # Options: "BatonRouge", "StLouis", "Baltimore"
```

Then run:
```bash
uv run python mwe.py
```

### Option 2: Analyze All Datasets (Recommended)

Run the comprehensive analysis script to analyze all available datasets:

```bash
uv run python analyze_all_datasets.py
```

This will automatically:
- Detect all available datasets
- Run PMF analysis on each dataset
- Generate all visualizations
- Save everything to the `output/` folder with dataset-specific naming

## Output Files

For each dataset, the following files are generated in the `output/` folder:

### Data Files
- `[dataset]_factor_contributions.csv` - Factor contributions over time
- `[dataset]_factor_profiles.csv` - Chemical profiles for each factor

### Visualizations

#### Overview Plots
- `[dataset]_factor_profiles_heatmap.png` - Heatmap of factor profiles
- `[dataset]_factor_contributions_heatmap.png` - Heatmap of factor contributions over time
- `[dataset]_factor_correlation_heatmap.png` - Correlation matrix between factors
- `[dataset]_top_species_by_factor.png` - Bar charts showing top species for each factor

#### Individual Factor Plots (EPA PMF Style)
- `[dataset]_factor_profile_factor_[1-7].png` - Individual factor profile plots
- `[dataset]_factor_contributions_factor_[1-7].png` - Individual factor contribution time series

## PMF Parameters

The analysis uses the following default parameters:
- **Number of factors**: 7
- **Random state**: 42 (for reproducibility)
- **Maximum iterations**: 1000
- **Tolerance**: 1e-4

To modify these parameters, edit the `PMF()` initialization in either script:

```python
pmf = PMF(n_components=5, random_state=42, max_iter=2000, tol=1e-5)
```

## Data Format Requirements

The package expects:
- Concentration and uncertainty files for each dataset
- CSV files (comma-separated) or TXT files (tab-separated)
- First column contains date/time identifiers
- Remaining columns contain chemical species data
- Matching column names between concentration and uncertainty files

## Adding New Datasets

To add a new dataset:

1. Place your concentration and uncertainty files in the `data/` folder
2. Add an entry to the `datasets` dictionary in either script:

```python
datasets = {
    "YourDataset": {
        "concentrations": "data/Your-Dataset-con.csv",
        "uncertainties": "data/Your-Dataset-unc.csv"
    }
}
```

3. Update the `dataset_name` variable in `mwe.py` or run `analyze_all_datasets.py`

## Troubleshooting

### Common Issues

1. **File not found errors**: Ensure your data files are in the `data/` folder with the correct names
2. **Shape mismatch**: Verify that concentration and uncertainty files have the same dimensions
3. **Missing dependencies**: Run `uv sync` to install all required packages

### Data Quality Checks

The scripts automatically:
- Check for matching shapes between concentration and uncertainty files
- Remove columns with all zeros or NaN values
- Display warnings for data inconsistencies
- Show data preprocessing statistics

## Dependencies

The package requires:
- pandas
- numpy
- matplotlib
- seaborn
- pathlib (built-in)

Install with: `uv sync`

## Example Output

After running the analysis, you'll see output like:

```
Successfully analyzed: 3 datasets
  ✓ BatonRouge
  ✓ StLouis
  ✓ Baltimore

All outputs saved to: C:\Projects\pmf\pmf\output
Each dataset has its own set of files with the dataset name as prefix.
```

The `output/` folder will contain 60+ files with comprehensive PMF analysis results and visualizations for all datasets.
