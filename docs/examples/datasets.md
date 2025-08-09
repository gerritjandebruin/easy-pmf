# Example Datasets

Easy PMF comes with three real-world environmental datasets that demonstrate different aspects of PMF analysis. These datasets are perfect for learning, testing, and validation.

## Dataset Overview

| Dataset | Location | Samples | Species | Time Period | Data Type |
|---------|----------|---------|---------|-------------|-----------|
| **Baltimore** | Baltimore, MD | 657 | 26 | PM2.5 composition | Urban air quality |
| **Baton Rouge** | Baton Rouge, LA | 307 | 41 | Air pollutants | Industrial/urban mix |
| **St. Louis** | St. Louis, MO | 418 | 13 | Environmental monitoring | Urban environment |

## Baltimore Dataset

### Description
The Baltimore dataset contains PM2.5 composition data collected in an urban environment. This dataset is excellent for learning source apportionment of fine particulate matter.

### Characteristics
- **Location**: Baltimore, Maryland (urban site)
- **Samples**: 657 daily measurements
- **Species**: 26 chemical components
- **Sources**: Traffic, coal combustion, sea salt, soil dust, secondary sulfate

### Key Species
- **Elements**: Al, Si, K, Ca, Ti, V, Cr, Mn, Fe, Ni, Cu, Zn, As, Se, Br, Pb
- **Ions**: SO4²⁻, NO3⁻, NH4⁺, Na⁺, Cl⁻
- **Carbon**: Organic Carbon (OC), Elemental Carbon (EC)

### Quick Analysis

```python
import pandas as pd
from easy_pmf import PMF

# Load Baltimore data
concentrations = pd.read_csv('data/Dataset-Baltimore_con.txt',
                           sep='\t', index_col=0, parse_dates=True)
uncertainties = pd.read_csv('data/Dataset-Baltimore_unc.txt',
                          sep='\t', index_col=0, parse_dates=True)

print(f"Baltimore dataset shape: {concentrations.shape}")
print(f"Date range: {concentrations.index.min()} to {concentrations.index.max()}")
print(f"Species: {list(concentrations.columns)}")

# Run PMF analysis
pmf = PMF(n_components=7, random_state=42)
pmf.fit(concentrations, uncertainties)

print(f"PMF Results:")
print(f"  Q-value: {pmf.score(concentrations, uncertainties):.2f}")
print(f"  Converged: {pmf.converged_}")
print(f"  Iterations: {pmf.n_iter_}")
```

### Expected Sources
Based on the urban Baltimore location, typical sources include:

1. **Traffic**: High EC, OC, and traffic-related metals (Cu, Zn)
2. **Coal Combustion**: High SO4²⁻, As, Se
3. **Sea Salt**: High Na⁺, Cl⁻ (coastal influence)
4. **Soil Dust**: Crustal elements (Al, Si, Ca, Fe)
5. **Secondary Sulfate**: High SO4²⁻, NH4⁺
6. **Industrial**: Various metals depending on local industry
7. **Oil Combustion**: V, Ni signature

## Baton Rouge Dataset

### Description
Baton Rouge data represents an industrial/urban environment with petrochemical industry influence. This dataset demonstrates complex source signatures in industrial areas.

### Characteristics
- **Location**: Baton Rouge, Louisiana
- **Samples**: 307 measurements
- **Species**: 41 chemical species
- **Sources**: Petrochemical industry, traffic, secondary aerosols, biomass burning

### Key Features
- Large number of species (41) provides rich chemical information
- Industrial setting with unique source signatures
- Mix of organic and inorganic components

### Quick Analysis

```python
# Load Baton Rouge data
concentrations = pd.read_csv('data/Dataset-BatonRouge-con.csv', index_col=0)
uncertainties = pd.read_csv('data/Dataset-BatonRouge-unc.csv', index_col=0)

print(f"Baton Rouge dataset shape: {concentrations.shape}")
print(f"Species: {list(concentrations.columns)}")

# Check data quality
detection_rates = (concentrations > 0).mean()
print(f"Species with >50% detection: {(detection_rates > 0.5).sum()}")

# Run PMF
pmf = PMF(n_components=6, random_state=42)
pmf.fit(concentrations, uncertainties)
```

### Expected Sources
Typical sources in industrial Baton Rouge:

1. **Petrochemical Industry**: Unique organic compound signatures
2. **Traffic**: Vehicle emissions
3. **Secondary Organic Aerosol**: Formed from industrial emissions
4. **Biomass Burning**: Seasonal influence
5. **Marine**: Gulf Coast influence
6. **Industrial Metals**: Heavy industry signatures

## St. Louis Dataset

### Description
St. Louis dataset represents a urban Midwest environment with a moderate number of well-characterized species.

### Characteristics
- **Location**: St. Louis, Missouri
- **Samples**: 418 measurements
- **Species**: 13 chemical components
- **Sources**: Urban mix with midwest characteristics

### Key Features
- Smaller number of species makes it good for learning
- Well-characterized urban environment
- Good data quality

### Quick Analysis

```python
# Load St. Louis data
concentrations = pd.read_csv('data/Dataset-StLouis-con.csv', index_col=0)
uncertainties = pd.read_csv('data/Dataset-StLouis-unc.csv', index_col=0)

print(f"St. Louis dataset shape: {concentrations.shape}")
print(f"Species: {list(concentrations.columns)}")

# Basic statistics
print(f"Data summary:")
print(concentrations.describe())

# Run PMF
pmf = PMF(n_components=5, random_state=42)
pmf.fit(concentrations, uncertainties)
```

### Expected Sources
Urban St. Louis sources:

1. **Traffic**: Vehicle emissions
2. **Coal Combustion**: Midwest power generation
3. **Secondary Sulfate**: Regional transport
4. **Soil Dust**: Local and regional dust
5. **Industrial**: Urban industrial activities

## Using the Command Line Interface

### Interactive Analysis

Run the interactive tool to explore all datasets:

```bash
easy-pmf --interactive
```

This will guide you through:
1. Dataset selection
2. Parameter choice
3. Analysis execution
4. Result visualization

### Batch Analysis

Analyze all datasets automatically:

```bash
easy-pmf --analyze-all
```

This runs PMF on all available datasets with default parameters.

### Custom Analysis

```bash
# Analyze with custom parameters
easy-pmf --factors 6 --data-dir ./data --output-dir ./results

# Use specific dataset
easy-pmf --interactive --data-dir ./baltimore_only
```

## Data Format Requirements

### File Naming Convention
- Concentrations: `*_con.csv`, `*_con.txt`
- Uncertainties: `*_unc.csv`, `*_unc.txt`

### Format Requirements
```python
# Concentrations file format
concentrations = pd.DataFrame({
    'Species1': [12.5, 8.3, ...],
    'Species2': [0.8, 1.2, ...],
    # ... more species
}, index=pd.DatetimeIndex([...]))  # Date/time index

# Uncertainties file (same format)
uncertainties = pd.DataFrame({
    'Species1': [1.2, 0.8, ...],
    'Species2': [0.1, 0.2, ...],
    # ... more species
}, index=concentrations.index)  # Same index
```

## Working with Your Own Data

### Converting Your Data

```python
import pandas as pd

# Example: Convert your data to Easy PMF format
def convert_to_pmf_format(your_data_file, output_prefix):
    """Convert your data to Easy PMF format."""

    # Load your data (adapt as needed)
    data = pd.read_csv(your_data_file)

    # Extract concentrations (adapt column selection)
    concentrations = data[['PM25', 'SO4', 'NO3', 'EC', 'OC', ...]].copy()

    # Set datetime index if needed
    concentrations.index = pd.to_datetime(data['Date'])

    # Estimate uncertainties (adapt as needed)
    uncertainties = concentrations * 0.15  # 15% uncertainty

    # Save in PMF format
    concentrations.to_csv(f'{output_prefix}_con.csv')
    uncertainties.to_csv(f'{output_prefix}_unc.csv')

    return concentrations, uncertainties

# Use the converter
# your_conc, your_unc = convert_to_pmf_format('my_data.csv', 'mysite')
```

### Validation Checklist

Before running PMF on your data:

- [ ] **Non-negative values**: All concentrations ≥ 0
- [ ] **Consistent dimensions**: Same shape for concentrations and uncertainties
- [ ] **Good detection rates**: >50% above detection limit for most species
- [ ] **Datetime index**: Proper time information
- [ ] **Units consistency**: All concentrations in same units (e.g., μg/m³)
- [ ] **No missing values**: Handle NaN appropriately
- [ ] **Reasonable uncertainties**: Typically 10-50% of concentrations

## Comparison Analysis

### Multi-Site Comparison

```python
def compare_datasets():
    """Compare PMF results across all three datasets."""

    datasets = {
        'Baltimore': {
            'conc': 'data/Dataset-Baltimore_con.txt',
            'unc': 'data/Dataset-Baltimore_unc.txt',
            'sep': '\t'
        },
        'BatonRouge': {
            'conc': 'data/Dataset-BatonRouge-con.csv',
            'unc': 'data/Dataset-BatonRouge-unc.csv',
            'sep': ','
        },
        'StLouis': {
            'conc': 'data/Dataset-StLouis-con.csv',
            'unc': 'data/Dataset-StLouis-unc.csv',
            'sep': ','
        }
    }

    results = {}

    for site, files in datasets.items():
        print(f"\nAnalyzing {site}...")

        # Load data
        conc = pd.read_csv(files['conc'], sep=files['sep'], index_col=0)
        unc = pd.read_csv(files['unc'], sep=files['sep'], index_col=0)

        # Determine optimal factor number
        n_factors = {
            'Baltimore': 7,
            'BatonRouge': 6,
            'StLouis': 5
        }[site]

        # Run PMF
        pmf = PMF(n_components=n_factors, random_state=42)
        pmf.fit(conc, unc)

        # Store results
        results[site] = {
            'pmf': pmf,
            'q_value': pmf.score(conc, unc),
            'n_samples': len(conc),
            'n_species': len(conc.columns),
            'n_factors': n_factors
        }

        print(f"  Samples: {results[site]['n_samples']}")
        print(f"  Species: {results[site]['n_species']}")
        print(f"  Factors: {results[site]['n_factors']}")
        print(f"  Q-value: {results[site]['q_value']:.2f}")
        print(f"  Converged: {pmf.converged_}")

    return results

# Run comparison
# comparison_results = compare_datasets()
```

## Next Steps

- Try the [Baltimore Analysis](baltimore.md) detailed walkthrough
- Learn about [Batch Processing](batch-processing.md) multiple datasets
- Explore [Custom Workflows](custom-workflows.md) for your specific needs
- Review the [User Guide](../user-guide/pmf-basics.md) for deeper understanding
