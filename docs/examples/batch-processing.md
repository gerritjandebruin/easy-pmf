# Batch Processing Multiple Datasets

This guide demonstrates how to efficiently process multiple datasets using Easy PMF, including the Baltimore, Baton Rouge, and St. Louis datasets included with the package.

## Overview

When working with multiple datasets, you often want to:

1. Apply consistent analysis parameters across datasets
2. Compare results between different locations or time periods
3. Automate the analysis workflow
4. Generate standardized reports

This example shows how to process all available datasets systematically.

## Available Datasets

Easy PMF includes several sample datasets:

| Dataset | Location | File Format | Description |
|---------|----------|-------------|-------------|
| Baltimore | Baltimore, MD | .txt | Urban atmospheric PM2.5 |
| Baton Rouge | Baton Rouge, LA | .csv | Industrial/urban mix |
| St. Louis | St. Louis, MO | .csv | Urban/industrial PM |

## Batch Processing Script

### 1. Setup and Configuration

```python
from easy_pmf import PMF
import pandas as pd
import os
from pathlib import Path
import matplotlib.pyplot as plt

# Define dataset configurations
datasets = {
    'Baltimore': {
        'concentration_file': 'data/Dataset-Baltimore_con.txt',
        'uncertainty_file': 'data/Dataset-Baltimore_unc.txt',
        'format': 'txt'
    },
    'BatonRouge': {
        'concentration_file': 'data/Dataset-BatonRouge-con.csv',
        'uncertainty_file': 'data/Dataset-BatonRouge-unc.csv',
        'format': 'csv'
    },
    'StLouis': {
        'concentration_file': 'data/Dataset-StLouis-con.csv',
        'uncertainty_file': 'data/Dataset-StLouis-unc.csv',
        'format': 'csv'
    }
}

# Analysis parameters (consistent across all datasets)
analysis_config = {
    'factor_range': range(3, 8),
    'n_runs': 20,
    'random_seed': 42,
    'remove_missing_threshold': 0.5,
    'replace_missing_method': 'median',
    'apply_uncertainty_scaling': True
}
```

### 2. Batch Analysis Function

```python
def analyze_dataset(dataset_name, dataset_config, analysis_config):
    """Analyze a single dataset with standard parameters."""

    print(f"\\n{'='*50}")
    print(f"Analyzing {dataset_name} dataset")
    print(f"{'='*50}")

    # Initialize PMF
    pmf = PMF()

    # Load data
    try:
        pmf.load_data(
            concentration_file=dataset_config['concentration_file'],
            uncertainty_file=dataset_config['uncertainty_file']
        )
        print(f"✓ Data loaded: {pmf.concentration_data.shape}")
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        return None

    # Data preparation
    pmf.prepare_data(
        remove_missing_threshold=analysis_config['remove_missing_threshold'],
        replace_missing_method=analysis_config['replace_missing_method'],
        apply_uncertainty_scaling=analysis_config['apply_uncertainty_scaling']
    )
    print(f"✓ Data prepared: {pmf.prepared_data.shape}")

    # Test multiple factor numbers
    results = {}
    q_qexp_values = {}

    for n_factors in analysis_config['factor_range']:
        print(f"  Running {n_factors} factors...", end='')

        try:
            result = pmf.run_pmf(
                n_factors=n_factors,
                n_runs=analysis_config['n_runs'],
                random_seed=analysis_config['random_seed']
            )

            results[n_factors] = result
            q_qexp_values[n_factors] = result.q_qexp
            print(f" Q/Qexp: {result.q_qexp:.2f}")

        except Exception as e:
            print(f" ✗ Error: {e}")
            continue

    # Select optimal number of factors
    if q_qexp_values:
        # Find factor number where Q/Qexp is closest to 1
        optimal_factors = min(q_qexp_values.keys(),
                            key=lambda x: abs(q_qexp_values[x] - 1.0))

        print(f"✓ Optimal factors: {optimal_factors} (Q/Qexp: {q_qexp_values[optimal_factors]:.2f})")

        return {
            'pmf': pmf,
            'results': results,
            'optimal_result': results[optimal_factors],
            'optimal_factors': optimal_factors,
            'q_qexp_values': q_qexp_values
        }

    return None

def run_batch_analysis(datasets, analysis_config):
    """Run PMF analysis on multiple datasets."""

    batch_results = {}

    for dataset_name, dataset_config in datasets.items():
        result = analyze_dataset(dataset_name, dataset_config, analysis_config)
        if result:
            batch_results[dataset_name] = result

            # Generate plots for this dataset
            result['pmf'].create_all_plots(
                result=result['optimal_result'],
                dataset_name=dataset_name,
                output_dir="output"
            )

    return batch_results
```

### 3. Running the Batch Analysis

```python
# Execute batch analysis
print("Starting batch PMF analysis...")
batch_results = run_batch_analysis(datasets, analysis_config)

print(f"\\n{'='*50}")
print("BATCH ANALYSIS COMPLETE")
print(f"{'='*50}")
print(f"Successfully analyzed {len(batch_results)} datasets")
```

### 4. Comparative Analysis

```python
def compare_results(batch_results):
    """Compare results across datasets."""

    print("\\nCOMPARATIVE ANALYSIS")
    print("-" * 30)

    # Summary table
    summary_data = []
    for dataset_name, result in batch_results.items():
        summary_data.append({
            'Dataset': dataset_name,
            'Optimal Factors': result['optimal_factors'],
            'Q/Qexp': result['optimal_result'].q_qexp,
            'Species Count': result['pmf'].concentration_data.shape[1],
            'Sample Count': result['pmf'].concentration_data.shape[0]
        })

    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))

    # Factor number distribution
    factor_counts = {}
    for result in batch_results.values():
        n_factors = result['optimal_factors']
        factor_counts[n_factors] = factor_counts.get(n_factors, 0) + 1

    print(f"\\nFactor number distribution:")
    for n_factors, count in sorted(factor_counts.items()):
        print(f"  {n_factors} factors: {count} dataset(s)")

    return summary_df

# Run comparative analysis
summary_df = compare_results(batch_results)
```

### 5. Cross-Dataset Visualization

```python
def create_comparison_plots(batch_results):
    """Create plots comparing results across datasets."""

    # Q/Qexp comparison
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Plot 1: Q/Qexp by factor number
    ax1 = axes[0, 0]
    for dataset_name, result in batch_results.items():
        factors = list(result['q_qexp_values'].keys())
        q_values = list(result['q_qexp_values'].values())
        ax1.plot(factors, q_values, 'o-', label=dataset_name)

    ax1.axhline(y=1.0, color='red', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Number of Factors')
    ax1.set_ylabel('Q/Qexp')
    ax1.set_title('Model Fit Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Dataset characteristics
    ax2 = axes[0, 1]
    dataset_names = []
    species_counts = []
    sample_counts = []

    for dataset_name, result in batch_results.items():
        dataset_names.append(dataset_name)
        species_counts.append(result['pmf'].concentration_data.shape[1])
        sample_counts.append(result['pmf'].concentration_data.shape[0])

    x_pos = range(len(dataset_names))
    ax2.bar([x - 0.2 for x in x_pos], species_counts, 0.4, label='Species', alpha=0.7)
    ax2.bar([x + 0.2 for x in x_pos], [s/10 for s in sample_counts], 0.4,
            label='Samples (÷10)', alpha=0.7)

    ax2.set_xlabel('Dataset')
    ax2.set_ylabel('Count')
    ax2.set_title('Dataset Characteristics')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(dataset_names)
    ax2.legend()

    # Plot 3: Optimal factor numbers
    ax3 = axes[1, 0]
    optimal_factors = [result['optimal_factors'] for result in batch_results.values()]
    ax3.bar(dataset_names, optimal_factors, alpha=0.7)
    ax3.set_xlabel('Dataset')
    ax3.set_ylabel('Optimal Number of Factors')
    ax3.set_title('Optimal Factor Numbers')

    # Plot 4: Final Q/Qexp values
    ax4 = axes[1, 1]
    final_q_qexp = [result['optimal_result'].q_qexp for result in batch_results.values()]
    ax4.bar(dataset_names, final_q_qexp, alpha=0.7)
    ax4.axhline(y=1.0, color='red', linestyle='--', alpha=0.5)
    ax4.set_xlabel('Dataset')
    ax4.set_ylabel('Q/Qexp')
    ax4.set_title('Final Model Fit Quality')

    plt.tight_layout()
    plt.savefig('output/batch_analysis_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

# Create comparison visualizations
create_comparison_plots(batch_results)
```

## Automated Report Generation

```python
def generate_batch_report(batch_results, output_file="output/batch_analysis_report.txt"):
    """Generate a comprehensive text report."""

    with open(output_file, 'w') as f:
        f.write("EASY PMF BATCH ANALYSIS REPORT\\n")
        f.write("=" * 50 + "\\n\\n")

        f.write(f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\\n")
        f.write(f"Datasets Analyzed: {len(batch_results)}\\n\\n")

        # Configuration summary
        f.write("ANALYSIS CONFIGURATION\\n")
        f.write("-" * 25 + "\\n")
        for key, value in analysis_config.items():
            f.write(f"{key}: {value}\\n")
        f.write("\\n")

        # Dataset summaries
        for dataset_name, result in batch_results.items():
            f.write(f"DATASET: {dataset_name.upper()}\\n")
            f.write("-" * 30 + "\\n")
            f.write(f"Data shape: {result['pmf'].concentration_data.shape}\\n")
            f.write(f"Optimal factors: {result['optimal_factors']}\\n")
            f.write(f"Final Q/Qexp: {result['optimal_result'].q_qexp:.3f}\\n")

            f.write("\\nQ/Qexp by factor number:\\n")
            for n_factors, q_qexp in result['q_qexp_values'].items():
                f.write(f"  {n_factors} factors: {q_qexp:.3f}\\n")
            f.write("\\n")

    print(f"Report saved to: {output_file}")

# Generate report
generate_batch_report(batch_results)
```

## Best Practices for Batch Processing

### 1. Consistent Parameters
- Use the same analysis parameters across datasets for fair comparison
- Document any dataset-specific modifications
- Consider normalizing data if concentrations differ significantly

### 2. Quality Control
- Check Q/Qexp values for all datasets
- Verify factor interpretability across locations
- Look for consistent source signatures

### 3. Error Handling
- Implement try-catch blocks for robust processing
- Log errors and continue with remaining datasets
- Provide detailed error messages

### 4. Resource Management
- Consider memory usage with large datasets
- Save intermediate results to avoid recomputation
- Use parallel processing for independent analyses

## Output Files

Batch processing generates:

- Individual dataset outputs in `output/` directory
- `batch_analysis_comparison.png`: Cross-dataset comparison plots
- `batch_analysis_report.txt`: Comprehensive text summary
- Dataset-specific plots and CSV files

## Advanced Batch Processing

For more complex scenarios:

1. **Seasonal Analysis**: Process data by seasons or months
2. **Parameter Sensitivity**: Test multiple parameter combinations
3. **Cross-Validation**: Split datasets for validation studies
4. **Automated QA/QC**: Implement automatic quality checks

See [Custom Workflows](custom-workflows.md) for advanced techniques.
