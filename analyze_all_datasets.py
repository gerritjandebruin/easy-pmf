"""PMF Analysis for All Datasets.

This script runs PMF analysis on all available datasets in the data folder.
It creates organized outputs in the output folder with dataset-specific naming.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from pmf import PMF


def analyze_dataset(dataset_name, dataset_info, output_dir, n_components=7):
    """Analyze a single dataset with PMF and generate all visualizations.

    Parameters:
    -----------
    dataset_name : str
        Name of the dataset
    dataset_info : dict
        Dictionary containing paths to concentration and uncertainty files
    output_dir : Path
        Directory to save outputs
    n_components : int
        Number of PMF factors to extract
    """
    print(f"\n{'=' * 50}")
    print(f"ANALYZING {dataset_name.upper()} DATASET")
    print(f"{'=' * 50}")

    # Load the concentration and uncertainty data
    print(f"Loading {dataset_name} dataset...")
    try:
        if dataset_info["concentrations"].endswith(".csv"):
            concentrations = pd.read_csv(dataset_info["concentrations"])
            uncertainties = pd.read_csv(dataset_info["uncertainties"])
        else:
            # For .txt files, assume tab-separated
            concentrations = pd.read_csv(dataset_info["concentrations"], sep="\t")
            uncertainties = pd.read_csv(dataset_info["uncertainties"], sep="\t")

        print(f"Successfully loaded {dataset_name} dataset")
        print(f"Concentrations shape: {concentrations.shape}")
        print(f"Uncertainties shape: {uncertainties.shape}")

    except FileNotFoundError:
        print(f"Error: Could not find data files for {dataset_name}")
        print("Make sure the following files exist:")
        print(f"- {dataset_info['concentrations']}")
        print(f"- {dataset_info['uncertainties']}")
        return False
    except Exception as e:
        print(f"Error loading {dataset_name} dataset: {e}")
        return False

    # The first column is often a date or identifier, so we'll set it as the index
    concentrations = concentrations.set_index(concentrations.columns[0])
    uncertainties = uncertainties.set_index(uncertainties.columns[0])

    # Ensure both datasets have the same shape and columns
    if concentrations.shape != uncertainties.shape:
        print(
            f"Warning: Concentrations shape {concentrations.shape} != "
            f"Uncertainties shape {uncertainties.shape}"
        )

    if not concentrations.columns.equals(uncertainties.columns):
        print("Warning: Column names in concentrations and uncertainties don't match")
        print("Concentrations columns:", list(concentrations.columns[:5]), "...")
        print("Uncertainties columns:", list(uncertainties.columns[:5]), "...")

    # Remove any columns with all NaN or zero values
    initial_cols = len(concentrations.columns)
    concentrations = concentrations.loc[:, (concentrations != 0).any(axis=0)]
    concentrations = concentrations.dropna(axis=1, how="all")
    uncertainties = uncertainties[
        concentrations.columns
    ]  # Keep same columns in uncertainties

    print(
        f"Data preprocessing: Removed "
        f"{initial_cols - len(concentrations.columns)} columns with all zeros or NaN"
    )
    print(f"Final data shape: {concentrations.shape}")
    print(
        f"Chemical species included: {list(concentrations.columns[:5])}..."
        if len(concentrations.columns) > 5
        else f"Chemical species: {list(concentrations.columns)}"
    )

    # Initialize the PMF model
    print(f"Running PMF analysis with {n_components} factors...")
    pmf = PMF(n_components=n_components, random_state=42)

    # Fit the model to the data
    pmf.fit(concentrations, uncertainties)

    # Save the results
    print("Saving PMF results...")
    # The contributions tell you how much each factor contributes
    # to each sample (time point)
    contributions = pd.DataFrame(
        pmf.contributions_.values,
        index=concentrations.index,
        columns=[f"Factor_{i + 1}" for i in range(pmf.n_components)],
    )
    contributions.to_csv(output_dir / f"{dataset_name}_factor_contributions.csv")

    # The profiles tell you the chemical profile of each factor
    profiles = pd.DataFrame(
        pmf.profiles_.values,
        columns=concentrations.columns,
        index=[f"Factor_{i + 1}" for i in range(pmf.n_components)],
    )
    profiles.to_csv(output_dir / f"{dataset_name}_factor_profiles.csv")

    print(
        f"PMF analysis complete. Results saved to {output_dir}/"
        f"{dataset_name}_factor_contributions.csv and "
        f"{output_dir}/{dataset_name}_factor_profiles.csv"
    )

    # Generate all visualizations
    generate_visualizations(
        dataset_name, concentrations, contributions, profiles, output_dir
    )

    return True


def generate_visualizations(
    dataset_name, concentrations, contributions, profiles, output_dir
):
    """Generate all PMF visualizations for a dataset."""
    print("Creating heatmap visualizations...")

    # Set up the plotting style
    plt.style.use("default")
    sns.set_palette("viridis")

    # 1. Factor Profiles Heatmap
    _fig, ax = plt.subplots(figsize=(16, 8))
    sns.heatmap(
        profiles,
        annot=False,
        cmap="viridis",
        cbar_kws={"label": "Concentration"},
        ax=ax,
    )
    plt.title(
        f"{dataset_name} - PMF Factor Profiles - Chemical Species Composition",
        fontsize=14,
        fontweight="bold",
    )
    plt.xlabel("Chemical Species", fontsize=12)
    plt.ylabel("PMF Factors", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(
        output_dir / f"{dataset_name}_factor_profiles_heatmap.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # 2. Factor Contributions Heatmap (time series)
    _fig, ax = plt.subplots(figsize=(12, 10))
    # For better visualization, we'll show every 5th time point to avoid overcrowding
    step = max(1, len(contributions) // 30)  # Show ~30 time points maximum
    contributions_subset = contributions.iloc[::step, :]

    sns.heatmap(
        contributions_subset.T,
        annot=False,
        cmap="plasma",
        cbar_kws={"label": "Factor Contribution"},
        ax=ax,
    )
    plt.title(
        f"{dataset_name} - PMF Factor Contributions Over Time",
        fontsize=14,
        fontweight="bold",
    )
    plt.xlabel("Time Points", fontsize=12)
    plt.ylabel("PMF Factors", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(
        output_dir / f"{dataset_name}_factor_contributions_heatmap.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # 3. Correlation matrix of factors
    _fig, ax = plt.subplots(figsize=(8, 6))
    correlation_matrix = contributions.corr()
    sns.heatmap(
        correlation_matrix,
        annot=True,
        cmap="coolwarm",
        center=0,
        square=True,
        cbar_kws={"label": "Correlation Coefficient"},
        ax=ax,
    )
    plt.title(
        f"{dataset_name} - Factor Correlation Matrix", fontsize=14, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(
        output_dir / f"{dataset_name}_factor_correlation_heatmap.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # 4. Top species contributions for each factor
    _fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    for i, factor in enumerate(profiles.index):
        if i < len(axes):
            # Get top 10 species for this factor
            top_species = profiles.loc[factor].nlargest(10)

            ax = axes[i]
            bars = ax.bar(
                range(len(top_species)),
                top_species.values,
                color=plt.cm.viridis(i / len(profiles.index)),
            )
            ax.set_title(f"{factor} - Top Species", fontweight="bold")
            ax.set_xticks(range(len(top_species)))
            ax.set_xticklabels(top_species.index, rotation=45, ha="right")
            ax.set_ylabel("Concentration")

            # Add value labels on bars
            for _j, bar in enumerate(bars):
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{height:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    # Hide any unused subplots
    for i in range(len(profiles.index), len(axes)):
        axes[i].set_visible(False)

    plt.suptitle(
        f"{dataset_name} - Top Chemical Species by PMF Factor",
        fontsize=16,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(
        output_dir / f"{dataset_name}_top_species_by_factor.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    print("Heatmap visualizations saved:")
    print(f"- {dataset_name}_factor_profiles_heatmap.png")
    print(f"- {dataset_name}_factor_contributions_heatmap.png")
    print(f"- {dataset_name}_factor_correlation_heatmap.png")
    print(f"- {dataset_name}_top_species_by_factor.png")

    # Create EPA PMF-style visualizations
    print("Creating EPA PMF-style visualizations...")

    # 5. EPA-style Factor Profile plots (one plot per factor)
    for _factor_idx, factor_name in enumerate(profiles.index):
        _fig, ax = plt.subplots(figsize=(14, 8))

        # Get the factor profile data
        factor_data = profiles.loc[factor_name]
        species_names = factor_data.index
        concentrations_data = factor_data.values

        # Calculate percentage of each species explained by this factor
        total_species_across_factors = profiles.sum(
            axis=0
        )  # Sum each species across all factors
        percentages = (concentrations_data / total_species_across_factors) * 100
        # Handle division by zero
        percentages = (
            percentages.fillna(0)
            if hasattr(percentages, "fillna")
            else np.where(total_species_across_factors == 0, 0, percentages)
        )

        # Create bar positions
        x_pos = np.arange(len(species_names))

        # Create the main bars (concentrations)
        bars = ax.bar(
            x_pos,
            concentrations_data,
            color="lightblue",
            edgecolor="navy",
            linewidth=0.5,
        )

        # Create secondary y-axis for percentages
        ax2 = ax.twinx()

        # Add percentage markers (red squares)
        ax2.scatter(x_pos, percentages, color="red", marker="s", s=30, zorder=5)

        # Customize the plot
        ax.set_xlabel("Chemical Species", fontsize=12, fontweight="bold")
        ax.set_ylabel("Concentration", fontsize=12, fontweight="bold", color="blue")
        ax2.set_ylabel(
            "% of Species Total", fontsize=12, fontweight="bold", color="red"
        )

        # Set logarithmic scale for concentrations (like EPA PMF)
        ax.set_yscale("log")
        ax.set_ylim(
            bottom=max(0.001, concentrations_data[concentrations_data > 0].min() * 0.1)
            if any(concentrations_data > 0)
            else 0.001
        )

        # Set percentage scale
        ax2.set_ylim(0, 100)

        # Customize x-axis
        ax.set_xticks(x_pos)
        ax.set_xticklabels(species_names, rotation=45, ha="right")

        # Add grid
        ax.grid(True, alpha=0.3, axis="y")

        # Color the y-axis labels
        ax.tick_params(axis="y", colors="blue")
        ax2.tick_params(axis="y", colors="red")

        # Add title
        plt.title(
            f"{dataset_name} - Factor Profile - {factor_name}",
            fontsize=14,
            fontweight="bold",
            pad=20,
        )

        # Add legend
        from matplotlib.patches import Rectangle

        legend_elements = [
            Rectangle(
                (0, 0),
                1,
                1,
                facecolor="lightblue",
                edgecolor="navy",
                label="Concentration",
            ),
            plt.Line2D(
                [0],
                [0],
                marker="s",
                color="w",
                markerfacecolor="red",
                markersize=8,
                label="% of Species Total",
            ),
        ]
        ax.legend(handles=legend_elements, loc="upper right")

        plt.tight_layout()
        plt.savefig(
            output_dir
            / f"{dataset_name}_factor_profile_{factor_name.lower().replace(' ', '_')}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    # 6. EPA-style Factor Contributions time series plots (one plot per factor)
    for _factor_idx, factor_name in enumerate(contributions.columns):
        _fig, ax = plt.subplots(figsize=(14, 6))

        # Get the factor contributions over time
        factor_contributions = contributions[factor_name]
        dates = contributions.index

        # Convert dates to datetime if they're strings
        if isinstance(dates[0], str):
            try:
                dates = pd.to_datetime(dates)
            except Exception:
                # If conversion fails, use numeric index
                dates = range(len(dates))

        # Create the line plot
        ax.plot(
            dates,
            factor_contributions,
            color="blue",
            linewidth=1.5,
            marker="o",
            markersize=3,
        )

        # Customize the plot
        ax.set_xlabel("Date", fontsize=12, fontweight="bold")
        ax.set_ylabel("Normalized Contributions", fontsize=12, fontweight="bold")
        ax.set_title(
            f"{dataset_name} - Factor Contributions - {factor_name}",
            fontsize=14,
            fontweight="bold",
            pad=20,
        )

        # Add grid
        ax.grid(True, alpha=0.3)

        # Format x-axis dates
        if hasattr(dates, "dtype") and "datetime" in str(dates.dtype):
            ax.tick_params(axis="x", rotation=45)

        # Set y-axis to start from 0
        ax.set_ylim(bottom=0)

        # Add some styling to match EPA PMF
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        plt.tight_layout()
        plt.savefig(
            output_dir
            / f"{dataset_name}_factor_contributions_{factor_name.lower().replace(' ', '_')}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    print("EPA PMF-style visualizations saved:")
    print(
        f"- Individual factor profile plots: {dataset_name}_factor_profile_factor_X.png"
    )
    print(
        f"- Individual factor contribution plots: "
        f"{dataset_name}_factor_contributions_factor_X.png"
    )


def main():
    """Main function to analyze all datasets."""
    # Create output directory if it doesn't exist
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    # Available datasets
    datasets = {
        "BatonRouge": {
            "concentrations": "data/Dataset-BatonRouge-con.csv",
            "uncertainties": "data/Dataset-BatonRouge-unc.csv",
        },
        "StLouis": {
            "concentrations": "data/Dataset-StLouis-con.csv",
            "uncertainties": "data/Dataset-StLouis-unc.csv",
        },
        "Baltimore": {
            "concentrations": "data/Dataset-Baltimore_con.txt",
            "uncertainties": "data/Dataset-Baltimore_unc.txt",
        },
    }

    print("PMF ANALYSIS FOR ALL DATASETS")
    print("=" * 60)
    print("This script will analyze all available datasets using PMF.")
    print(f"Outputs will be saved to: {output_dir.absolute()}")
    print("=" * 60)

    successful_analyses = []
    failed_analyses = []

    # Analyze each dataset
    for dataset_name, dataset_info in datasets.items():
        try:
            success = analyze_dataset(
                dataset_name, dataset_info, output_dir, n_components=7
            )
            if success:
                successful_analyses.append(dataset_name)
            else:
                failed_analyses.append(dataset_name)
        except Exception as e:
            print(f"Error analyzing {dataset_name}: {e}")
            failed_analyses.append(dataset_name)

    # Summary
    print("\n" + "=" * 60)
    print("ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"Successfully analyzed: {len(successful_analyses)} datasets")
    for name in successful_analyses:
        print(f"  ✓ {name}")

    if failed_analyses:
        print(f"\nFailed to analyze: {len(failed_analyses)} datasets")
        for name in failed_analyses:
            print(f"  ✗ {name}")

    print(f"\nAll outputs saved to: {output_dir.absolute()}")
    print("Each dataset has its own set of files with the dataset name as prefix.")


if __name__ == "__main__":
    main()
