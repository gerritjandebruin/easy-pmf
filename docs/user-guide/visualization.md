# Visualization

Easy PMF provides comprehensive visualization capabilities for exploring and presenting PMF analysis results. This guide covers all available plotting options and customization techniques.

## Built-in Visualization Functions

### Quick Plotting

```python
import matplotlib.pyplot as plt
import seaborn as sns
from easy_pmf import PMF

# After fitting PMF
pmf = PMF(n_components=5, random_state=42)
pmf.fit(concentrations, uncertainties)

# Quick time series plot
pmf.contributions_.plot(figsize=(12, 6))
plt.title('Factor Contributions Over Time')
plt.ylabel('Contribution (μg/m³)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# Quick heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(pmf.profiles_, annot=True, fmt='.3f', cmap='viridis')
plt.title('Factor Profiles Heatmap')
plt.ylabel('Factors')
plt.xlabel('Chemical Species')
plt.tight_layout()
plt.show()
```

## Comprehensive Visualization Suite

### Factor Profiles Visualization

```python
def plot_factor_profiles(pmf, source_names=None, figsize=(15, 10)):
    """Create comprehensive factor profiles visualization."""

    profiles = pmf.profiles_
    n_factors = len(profiles)

    # Create subplots
    cols = 3
    rows = (n_factors + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=figsize)

    if rows == 1:
        axes = [axes]
    if cols == 1:
        axes = [[ax] for ax in axes]

    axes = [ax for row in axes for ax in row]  # Flatten

    colors = plt.cm.Set3(np.linspace(0, 1, n_factors))

    for i, (factor_idx, factor_data) in enumerate(profiles.iterrows()):
        if i >= len(axes):
            break

        ax = axes[i]

        # Bar plot
        bars = ax.bar(range(len(factor_data)), factor_data,
                     color=colors[i], alpha=0.8, edgecolor='black', linewidth=0.5)

        # Formatting
        factor_name = source_names[i] if source_names else f'Factor {i+1}'
        ax.set_title(f'{factor_name}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Chemical Species')
        ax.set_ylabel('Relative Abundance')

        # Rotate x-axis labels
        ax.set_xticks(range(len(factor_data)))
        ax.set_xticklabels(factor_data.index, rotation=45, ha='right')

        # Add grid
        ax.grid(True, alpha=0.3, axis='y')

        # Highlight top contributors
        top_3_indices = factor_data.nlargest(3).index
        for j, species in enumerate(factor_data.index):
            if species in top_3_indices:
                bars[j].set_color('red')
                bars[j].set_alpha(0.9)

    # Remove unused subplots
    for i in range(n_factors, len(axes)):
        fig.delaxes(axes[i])

    plt.suptitle('Factor Profiles (Chemical Signatures)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

# Example usage
source_names = ['Traffic', 'Coal', 'Sea Salt', 'Soil', 'Secondary Sulfate']
plot_factor_profiles(pmf, source_names)
```

### Factor Contributions Visualization

```python
def plot_factor_contributions(pmf, source_names=None, figsize=(15, 12)):
    """Create comprehensive factor contributions visualization."""

    contributions = pmf.contributions_

    fig, axes = plt.subplots(3, 2, figsize=figsize)

    # 1. Time series plot
    ax1 = axes[0, 0]
    contributions.plot(ax=ax1, alpha=0.8)
    ax1.set_title('Factor Contributions Time Series')
    ax1.set_ylabel('Contribution (μg/m³)')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)

    # 2. Stacked area plot
    ax2 = axes[0, 1]
    contributions.plot.area(ax=ax2, alpha=0.7, stacked=True)
    ax2.set_title('Stacked Factor Contributions')
    ax2.set_ylabel('Contribution (μg/m³)')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    # 3. Box plot
    ax3 = axes[1, 0]
    contributions.boxplot(ax=ax3)
    ax3.set_title('Contribution Distributions')
    ax3.set_ylabel('Contribution (μg/m³)')
    ax3.tick_params(axis='x', rotation=45)

    # 4. Pie chart of average contributions
    ax4 = axes[1, 1]
    avg_contributions = contributions.mean()
    colors = plt.cm.Set3(np.linspace(0, 1, len(avg_contributions)))

    wedges, texts, autotexts = ax4.pie(avg_contributions.values,
                                      labels=source_names if source_names else avg_contributions.index,
                                      autopct='%1.1f%%',
                                      colors=colors,
                                      startangle=90)
    ax4.set_title('Average Factor Contributions')

    # 5. Correlation heatmap
    ax5 = axes[2, 0]
    corr_matrix = contributions.corr()
    im = ax5.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
    ax5.set_xticks(range(len(corr_matrix)))
    ax5.set_yticks(range(len(corr_matrix)))
    ax5.set_xticklabels(source_names if source_names else corr_matrix.columns, rotation=45)
    ax5.set_yticklabels(source_names if source_names else corr_matrix.index)
    ax5.set_title('Factor Correlation Matrix')

    # Add correlation values
    for i in range(len(corr_matrix)):
        for j in range(len(corr_matrix)):
            ax5.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                    ha='center', va='center', color='black')

    # 6. Monthly patterns (if data spans multiple months)
    ax6 = axes[2, 1]
    if len(contributions) > 30:  # If we have enough data
        monthly_data = contributions.copy()
        monthly_data['month'] = monthly_data.index.month
        monthly_avg = monthly_data.groupby('month')[contributions.columns].mean()

        monthly_avg.plot(kind='bar', ax=ax6, alpha=0.8)
        ax6.set_title('Monthly Average Contributions')
        ax6.set_xlabel('Month')
        ax6.set_ylabel('Average Contribution')
        ax6.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax6.tick_params(axis='x', rotation=0)
    else:
        ax6.text(0.5, 0.5, 'Insufficient data\nfor monthly analysis',
                ha='center', va='center', transform=ax6.transAxes)
        ax6.set_title('Monthly Analysis')

    plt.tight_layout()
    plt.show()

# Example usage
plot_factor_contributions(pmf, source_names)
```

### EPA PMF-Style Plots

```python
def create_epa_style_plots(pmf, concentrations, uncertainties, source_names=None):
    """Create EPA PMF-style plots for official reporting."""

    contributions = pmf.contributions_
    profiles = pmf.profiles_

    # Calculate additional metrics
    reconstructed = pd.DataFrame(
        contributions.values @ profiles.values,
        index=concentrations.index,
        columns=concentrations.columns
    )

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    # 1. Time series with data points
    ax1 = axes[0, 0]
    total_measured = concentrations.sum(axis=1)
    total_reconstructed = contributions.sum(axis=1)

    ax1.scatter(contributions.index, total_measured, alpha=0.6, s=20, label='Measured', color='blue')
    ax1.plot(contributions.index, total_reconstructed, color='red', linewidth=2, label='PMF Reconstruction')
    ax1.set_title('Total Mass: Measured vs PMF')
    ax1.set_ylabel('Concentration (μg/m³)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Scatter plot: measured vs predicted
    ax2 = axes[0, 1]
    ax2.scatter(total_measured, total_reconstructed, alpha=0.6, s=30)

    # Add 1:1 line
    min_val = min(total_measured.min(), total_reconstructed.min())
    max_val = max(total_measured.max(), total_reconstructed.max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2)

    # Add R²
    r_squared = np.corrcoef(total_measured, total_reconstructed)[0, 1] ** 2
    ax2.text(0.05, 0.95, f'R² = {r_squared:.3f}', transform=ax2.transAxes,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax2.set_xlabel('Measured Total Mass')
    ax2.set_ylabel('PMF Reconstructed Total Mass')
    ax2.set_title('Measured vs PMF Total Mass')
    ax2.grid(True, alpha=0.3)

    # 3. Residuals plot
    ax3 = axes[0, 2]
    residuals = total_measured - total_reconstructed
    ax3.scatter(total_reconstructed, residuals, alpha=0.6, s=30)
    ax3.axhline(y=0, color='red', linestyle='--', linewidth=2)
    ax3.set_xlabel('PMF Reconstructed Total Mass')
    ax3.set_ylabel('Residuals (Measured - PMF)')
    ax3.set_title('Residuals vs PMF Reconstruction')
    ax3.grid(True, alpha=0.3)

    # 4. Factor profiles bar chart (EPA style)
    ax4 = axes[1, 0]
    profiles_pct = profiles.div(profiles.sum(axis=1), axis=0) * 100

    x = np.arange(len(profiles.columns))
    width = 0.8 / len(profiles)
    colors = plt.cm.Set3(np.linspace(0, 1, len(profiles)))

    for i, (factor_idx, factor_data) in enumerate(profiles_pct.iterrows()):
        offset = (i - len(profiles)/2 + 0.5) * width
        factor_name = source_names[i] if source_names else f'Factor {i+1}'
        ax4.bar(x + offset, factor_data, width, label=factor_name,
               color=colors[i], alpha=0.8, edgecolor='black', linewidth=0.5)

    ax4.set_xlabel('Chemical Species')
    ax4.set_ylabel('Percentage of Factor (%)')
    ax4.set_title('Factor Profiles (% by Factor)')
    ax4.set_xticks(x)
    ax4.set_xticklabels(profiles.columns, rotation=45, ha='right')
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')

    # 5. Seasonal analysis
    ax5 = axes[1, 1]
    if len(contributions) > 90:  # Need at least ~3 months of data
        seasonal_data = contributions.copy()
        seasonal_data['season'] = seasonal_data.index.month % 12 // 3
        season_names = ['Winter', 'Spring', 'Summer', 'Fall']
        seasonal_avg = seasonal_data.groupby('season')[contributions.columns].mean()
        seasonal_avg.index = [season_names[i] for i in seasonal_avg.index]

        seasonal_avg.plot(kind='bar', ax=ax5, alpha=0.8, width=0.8)
        ax5.set_title('Seasonal Average Contributions')
        ax5.set_ylabel('Average Contribution (μg/m³)')
        ax5.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax5.tick_params(axis='x', rotation=45)
    else:
        ax5.text(0.5, 0.5, 'Insufficient data\nfor seasonal analysis',
                ha='center', va='center', transform=ax5.transAxes, fontsize=12)
        ax5.set_title('Seasonal Analysis')

    # 6. Q-value information
    ax6 = axes[1, 2]
    ax6.axis('off')

    q_value = pmf.score(concentrations, uncertainties)
    n_samples, n_species = concentrations.shape
    n_factors = pmf.n_components
    q_theoretical = (n_samples * n_species) - (n_factors * (n_samples + n_species))

    info_text = f"""PMF Model Summary

    Number of Factors: {n_factors}
    Samples: {n_samples}
    Species: {n_species}

    Q-value: {q_value:.1f}
    Q-theoretical: {q_theoretical:.0f}
    Q/Q-theo: {q_value/q_theoretical:.2f}

    Converged: {pmf.converged_}
    Iterations: {pmf.n_iter_}

    Status: {'Good' if q_value/q_theoretical < 2 else 'Check model'}
    """

    ax6.text(0.1, 0.9, info_text, transform=ax6.transAxes, fontsize=12,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    plt.suptitle('EPA PMF-Style Analysis Summary', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

# Example usage
create_epa_style_plots(pmf, concentrations, uncertainties, source_names)
```

### Interactive Visualizations

```python
def create_interactive_dashboard(pmf, concentrations):
    """Create interactive dashboard using plotly (if available)."""

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        import plotly.express as px
    except ImportError:
        print("Plotly not available. Install with: pip install plotly")
        return

    contributions = pmf.contributions_
    profiles = pmf.profiles_

    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Factor Contributions', 'Factor Profiles',
                       'Correlation Matrix', 'Monthly Patterns'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )

    # 1. Factor contributions time series
    for col in contributions.columns:
        fig.add_trace(
            go.Scatter(x=contributions.index, y=contributions[col],
                      mode='lines', name=col,
                      line=dict(width=2)),
            row=1, col=1
        )

    # 2. Factor profiles heatmap
    fig.add_trace(
        go.Heatmap(z=profiles.values,
                  x=profiles.columns,
                  y=profiles.index,
                  colorscale='Viridis',
                  showscale=True),
        row=1, col=2
    )

    # 3. Correlation matrix
    corr_matrix = contributions.corr()
    fig.add_trace(
        go.Heatmap(z=corr_matrix.values,
                  x=corr_matrix.columns,
                  y=corr_matrix.index,
                  colorscale='RdBu',
                  zmid=0,
                  showscale=True),
        row=2, col=1
    )

    # 4. Monthly patterns (if applicable)
    if len(contributions) > 30:
        monthly_data = contributions.copy()
        monthly_data['month'] = monthly_data.index.month
        monthly_avg = monthly_data.groupby('month')[contributions.columns].mean()

        for col in contributions.columns:
            fig.add_trace(
                go.Bar(x=monthly_avg.index, y=monthly_avg[col],
                      name=col, showlegend=False),
                row=2, col=2
            )

    # Update layout
    fig.update_layout(
        title="Interactive PMF Analysis Dashboard",
        height=800,
        showlegend=True
    )

    # Show the interactive plot
    fig.show()

# Example usage (requires plotly)
# create_interactive_dashboard(pmf, concentrations)
```

### Publication-Ready Figures

```python
def create_publication_figure(pmf, concentrations, uncertainties,
                             source_names=None, save_path=None):
    """Create publication-ready figure with proper formatting."""

    # Set publication style
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 11,
        'figure.titlesize': 18
    })

    contributions = pmf.contributions_
    profiles = pmf.profiles_

    fig = plt.figure(figsize=(16, 12))

    # Create custom grid layout
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)

    # 1. Factor contributions (spans 2 columns)
    ax1 = fig.add_subplot(gs[0, :2])
    for i, col in enumerate(contributions.columns):
        color = plt.cm.Set2(i / len(contributions.columns))
        factor_name = source_names[i] if source_names else col
        ax1.plot(contributions.index, contributions[col],
                label=factor_name, linewidth=2, color=color)

    ax1.set_xlabel('Date')
    ax1.set_ylabel('Contribution (μg m⁻³)')
    ax1.set_title('(a) Factor Contributions Time Series')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)

    # 2. Average contribution pie chart
    ax2 = fig.add_subplot(gs[0, 2])
    avg_contributions = contributions.mean()
    colors = plt.cm.Set2(np.linspace(0, 1, len(avg_contributions)))

    wedges, texts, autotexts = ax2.pie(
        avg_contributions.values,
        labels=[source_names[i] if source_names else f'F{i+1}'
               for i in range(len(avg_contributions))],
        autopct='%1.1f%%',
        colors=colors,
        startangle=90
    )
    ax2.set_title('(b) Average Contributions')

    # 3. Q-value and model info
    ax3 = fig.add_subplot(gs[0, 3])
    ax3.axis('off')

    q_value = pmf.score(concentrations, uncertainties)
    n_samples, n_species = concentrations.shape
    q_theoretical = (n_samples * n_species) - (pmf.n_components * (n_samples + n_species))

    info_text = f"""Model Performance:

Q-value: {q_value:.1f}
Q/Q_theo: {q_value/q_theoretical:.2f}
R²: {np.corrcoef(concentrations.sum(axis=1), contributions.sum(axis=1))[0,1]**2:.3f}

Converged: {pmf.converged_}
Iterations: {pmf.n_iter_}
    """

    ax3.text(0.1, 0.9, info_text, transform=ax3.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    ax3.set_title('(c) Model Statistics')

    # 4. Factor profiles (spans all columns)
    n_factors = len(profiles)
    factor_axes = []

    for i in range(n_factors):
        if i < 3:  # First row of factor profiles
            ax = fig.add_subplot(gs[1, i])
        else:  # Second row of factor profiles
            ax = fig.add_subplot(gs[2, i-3])

        factor_axes.append(ax)

        factor_data = profiles.iloc[i]
        bars = ax.bar(range(len(factor_data)), factor_data,
                     color=plt.cm.Set2(i / n_factors), alpha=0.8,
                     edgecolor='black', linewidth=0.5)

        # Highlight top 3 species
        top_3_indices = factor_data.nlargest(3).index
        for j, species in enumerate(factor_data.index):
            if species in top_3_indices:
                bars[j].set_color('red')
                bars[j].set_alpha(0.9)

        factor_name = source_names[i] if source_names else f'Factor {i+1}'
        ax.set_title(f'({chr(100+i)}) {factor_name}')
        ax.set_ylabel('Relative Abundance')

        # Format x-axis
        ax.set_xticks(range(len(factor_data)))
        ax.set_xticklabels(factor_data.index, rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')

    # Main title
    plt.suptitle('Positive Matrix Factorization Analysis Results',
                fontsize=18, fontweight='bold', y=0.98)

    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        print(f"Figure saved to: {save_path}")

    plt.show()

# Example usage
create_publication_figure(pmf, concentrations, uncertainties,
                         source_names, save_path='pmf_results_figure.png')
```

### Diagnostic Plots

```python
def create_diagnostic_plots(pmf, concentrations, uncertainties):
    """Create comprehensive diagnostic plots for model validation."""

    # Calculate metrics
    reconstructed = pd.DataFrame(
        pmf.contributions_.values @ pmf.profiles_.values,
        index=concentrations.index,
        columns=concentrations.columns
    )

    residuals = concentrations - reconstructed
    scaled_residuals = residuals / uncertainties

    fig, axes = plt.subplots(3, 3, figsize=(18, 15))

    # 1. Convergence history (if available)
    ax1 = axes[0, 0]
    if hasattr(pmf, '_convergence_history'):
        ax1.semilogy(pmf._convergence_history)
        ax1.axhline(y=pmf.tol, color='red', linestyle='--', label='Tolerance')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Convergence Metric')
        ax1.set_title('Convergence History')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    else:
        ax1.text(0.5, 0.5, 'Convergence history\nnot available',
                ha='center', va='center', transform=ax1.transAxes)
        ax1.set_title('Convergence History')

    # 2. Scaled residuals histogram
    ax2 = axes[0, 1]
    ax2.hist(scaled_residuals.values.flatten(), bins=50, alpha=0.7, density=True)
    x = np.linspace(-4, 4, 100)
    ax2.plot(x, norm.pdf(x, 0, 1), 'r-', linewidth=2, label='Standard Normal')
    ax2.set_xlabel('Scaled Residuals')
    ax2.set_ylabel('Density')
    ax2.set_title('Scaled Residuals Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Q-Q plot
    ax3 = axes[0, 2]
    from scipy import stats
    stats.probplot(scaled_residuals.values.flatten(), dist="norm", plot=ax3)
    ax3.set_title('Q-Q Plot vs Normal')
    ax3.grid(True, alpha=0.3)

    # 4. Species reconstruction quality
    ax4 = axes[1, 0]
    r_squared_values = []
    for col in concentrations.columns:
        obs = concentrations[col]
        pred = reconstructed[col]
        ss_res = np.sum((obs - pred) ** 2)
        ss_tot = np.sum((obs - obs.mean()) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        r_squared_values.append(r2)

    species_r2 = pd.Series(r_squared_values, index=concentrations.columns)
    species_r2.plot(kind='bar', ax=ax4, color='skyblue', alpha=0.8)
    ax4.set_title('Species Reconstruction Quality (R²)')
    ax4.set_ylabel('R²')
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(True, alpha=0.3, axis='y')

    # 5. Residuals vs predicted
    ax5 = axes[1, 1]
    ax5.scatter(reconstructed.values.flatten(), residuals.values.flatten(),
               alpha=0.5, s=10)
    ax5.axhline(y=0, color='red', linestyle='--', linewidth=2)
    ax5.set_xlabel('Predicted Concentration')
    ax5.set_ylabel('Residual')
    ax5.set_title('Residuals vs Predicted')
    ax5.grid(True, alpha=0.3)

    # 6. Factor correlation matrix
    ax6 = axes[1, 2]
    corr_matrix = pmf.contributions_.corr()
    im = ax6.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
    ax6.set_xticks(range(len(corr_matrix)))
    ax6.set_yticks(range(len(corr_matrix)))
    ax6.set_xticklabels(corr_matrix.columns)
    ax6.set_yticklabels(corr_matrix.index)
    ax6.set_title('Factor Correlation Matrix')

    # Add text annotations
    for i in range(len(corr_matrix)):
        for j in range(len(corr_matrix)):
            ax6.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                    ha='center', va='center', color='black', fontsize=10)

    plt.colorbar(im, ax=ax6)

    # 7. Bootstrap stability (placeholder)
    ax7 = axes[2, 0]
    ax7.text(0.5, 0.5, 'Bootstrap stability\nanalysis would go here\n\n' +
            'Run bootstrap validation\nto populate this plot',
            ha='center', va='center', transform=ax7.transAxes)
    ax7.set_title('Bootstrap Stability')

    # 8. Temporal patterns
    ax8 = axes[2, 1]
    if len(pmf.contributions_) > 30:
        monthly_data = pmf.contributions_.copy()
        monthly_data['month'] = monthly_data.index.month
        monthly_cv = monthly_data.groupby('month')[pmf.contributions_.columns].std() / \
                    monthly_data.groupby('month')[pmf.contributions_.columns].mean()

        monthly_cv.mean(axis=1).plot(kind='bar', ax=ax8, color='orange', alpha=0.8)
        ax8.set_title('Monthly Variability (CV)')
        ax8.set_xlabel('Month')
        ax8.set_ylabel('Coefficient of Variation')
        ax8.tick_params(axis='x', rotation=0)
    else:
        ax8.text(0.5, 0.5, 'Insufficient data\nfor temporal analysis',
                ha='center', va='center', transform=ax8.transAxes)
        ax8.set_title('Temporal Patterns')

    # 9. Model summary statistics
    ax9 = axes[2, 2]
    ax9.axis('off')

    q_value = pmf.score(concentrations, uncertainties)
    n_samples, n_species = concentrations.shape
    q_theoretical = (n_samples * n_species) - (pmf.n_components * (n_samples + n_species))

    # Calculate additional metrics
    total_r2 = np.corrcoef(concentrations.sum(axis=1), pmf.contributions_.sum(axis=1))[0,1]**2
    mean_species_r2 = np.mean(r_squared_values)
    outlier_pct = (np.abs(scaled_residuals) > 3).sum().sum() / scaled_residuals.size * 100

    summary_text = f"""Model Diagnostics Summary:

Q-value: {q_value:.1f}
Q/Q_theoretical: {q_value/q_theoretical:.2f}
Total Mass R²: {total_r2:.3f}
Mean Species R²: {mean_species_r2:.3f}

Outliers (|z| > 3): {outlier_pct:.1f}%
Convergence: {pmf.converged_}
Iterations: {pmf.n_iter_}

Status: {'✓ Good' if q_value/q_theoretical < 2 and total_r2 > 0.8 else '⚠ Check'}
    """

    ax9.text(0.1, 0.9, summary_text, transform=ax9.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    ax9.set_title('Diagnostic Summary')

    plt.suptitle('PMF Model Diagnostic Plots', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

from scipy.stats import norm

# Example usage
create_diagnostic_plots(pmf, concentrations, uncertainties)
```

## Customization Options

### Color Schemes and Styling

```python
# Custom color palettes
def get_custom_colors(n_colors, palette='environmental'):
    """Get custom color palettes for PMF plots."""

    palettes = {
        'environmental': ['#2E8B57', '#4682B4', '#CD853F', '#8B4513', '#9370DB', '#20B2AA'],
        'pollution': ['#FF4500', '#DC143C', '#8B0000', '#FF6347', '#FF1493', '#B22222'],
        'sources': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'],
        'scientific': ['#0072BD', '#D95319', '#EDB120', '#7E2F8E', '#77AC30', '#4DBEEE']
    }

    if palette in palettes:
        colors = palettes[palette]
        # Extend if needed
        while len(colors) < n_colors:
            colors.extend(colors)
        return colors[:n_colors]
    else:
        return plt.cm.Set3(np.linspace(0, 1, n_colors))

# Apply custom styling
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.facecolor': 'white'
})
```

### Export Options

```python
def save_all_plots(pmf, concentrations, uncertainties, output_dir='plots'):
    """Save all PMF plots in multiple formats."""

    import os
    os.makedirs(output_dir, exist_ok=True)

    # Set high DPI for publication quality
    plt.rcParams['savefig.dpi'] = 300

    # Factor profiles
    plot_factor_profiles(pmf)
    plt.savefig(f'{output_dir}/factor_profiles.png', bbox_inches='tight')
    plt.savefig(f'{output_dir}/factor_profiles.pdf', bbox_inches='tight')
    plt.close()

    # Factor contributions
    plot_factor_contributions(pmf)
    plt.savefig(f'{output_dir}/factor_contributions.png', bbox_inches='tight')
    plt.savefig(f'{output_dir}/factor_contributions.pdf', bbox_inches='tight')
    plt.close()

    # Diagnostic plots
    create_diagnostic_plots(pmf, concentrations, uncertainties)
    plt.savefig(f'{output_dir}/diagnostics.png', bbox_inches='tight')
    plt.close()

    print(f"All plots saved to {output_dir}/")

# Example usage
# save_all_plots(pmf, concentrations, uncertainties)
```

## Best Practices for Visualization

### 1. Choose Appropriate Plot Types
- **Time series**: For temporal patterns
- **Bar charts**: For comparing factor profiles
- **Heatmaps**: For correlation matrices
- **Scatter plots**: For model validation
- **Pie charts**: For contribution percentages

### 2. Use Clear Labels and Titles
- Include units in axis labels
- Use descriptive factor names
- Add informative titles
- Include legends when needed

### 3. Color Considerations
- Use colorblind-friendly palettes
- Maintain consistency across plots
- Use red for highlighting important features
- Consider printing in grayscale

### 4. Publication Guidelines
- Use vector formats (PDF, SVG) for scalability
- Ensure text is readable at target size
- Follow journal-specific formatting requirements
- Include all necessary information in captions

## Next Steps

- Apply these visualizations to your own data
- Explore the [Examples](../examples/baltimore.md) for real-world applications
- Learn about [Advanced Analysis](../user-guide/interpreting-results.md) techniques
- Contribute new visualization features via [GitHub](https://github.com/gerritjandebruin/easy-pmf)
