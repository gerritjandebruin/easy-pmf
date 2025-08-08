"""Quick PMF Analysis Script

This script provides an easy interface to run PMF analysis on a specific dataset.
Simply run this script and follow the prompts to select which dataset to analyze.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pmf import PMF


def get_available_datasets():
    """Get list of available datasets from the data folder."""
    data_dir = Path("data")
    if not data_dir.exists():
        return {}

    datasets = {}

    # Look for concentration files and try to find matching uncertainty files
    for con_file in data_dir.glob("*con*"):
        # Try to find corresponding uncertainty file
        base_name = con_file.stem

        # Different naming patterns
        possible_unc_names = [
            base_name.replace("con", "unc"),
            base_name.replace("-con", "-unc"),
            base_name.replace("_con", "_unc"),
        ]

        for unc_name in possible_unc_names:
            unc_file = data_dir / (unc_name + con_file.suffix)
            if unc_file.exists():
                # Extract dataset name
                dataset_name = (
                    base_name.replace("Dataset-", "")
                    .replace("_con", "")
                    .replace("-con", "")
                )
                dataset_name = dataset_name.replace("_", " ").replace("-", " ").title()

                datasets[dataset_name] = {
                    "concentrations": str(con_file),
                    "uncertainties": str(unc_file),
                }
                break

    return datasets


def select_dataset(datasets):
    """Interactive dataset selection."""
    if not datasets:
        print("No datasets found in the data folder!")
        print("Make sure you have matching concentration and uncertainty files.")
        return None

    print("\nAvailable datasets:")
    print("-" * 30)

    dataset_list = list(datasets.keys())
    for i, name in enumerate(dataset_list, 1):
        con_file = Path(datasets[name]["concentrations"])
        unc_file = Path(datasets[name]["uncertainties"])
        print(f"{i}. {name}")
        print(f"   Concentrations: {con_file.name}")
        print(f"   Uncertainties:  {unc_file.name}")
        print()

    while True:
        try:
            choice = (
                input(
                    f"Select dataset (1-{len(dataset_list)}) or 'all' for all datasets: "
                )
                .strip()
                .lower()
            )

            if choice == "all":
                return "all"

            choice_num = int(choice)
            if 1 <= choice_num <= len(dataset_list):
                return dataset_list[choice_num - 1]
            else:
                print(f"Please enter a number between 1 and {len(dataset_list)}")

        except ValueError:
            print("Please enter a valid number or 'all'")


def analyze_single_dataset(dataset_name, dataset_info, output_dir):
    """Analyze a single dataset - simplified version of the main analysis function."""
    print(f"\nAnalyzing {dataset_name}...")
    print("-" * 40)

    try:
        # Load data
        if dataset_info["concentrations"].endswith(".csv"):
            concentrations = pd.read_csv(dataset_info["concentrations"])
            uncertainties = pd.read_csv(dataset_info["uncertainties"])
        else:
            concentrations = pd.read_csv(dataset_info["concentrations"], sep="\t")
            uncertainties = pd.read_csv(dataset_info["uncertainties"], sep="\t")

        # Set index
        concentrations = concentrations.set_index(concentrations.columns[0])
        uncertainties = uncertainties.set_index(uncertainties.columns[0])

        # Clean data
        initial_cols = len(concentrations.columns)
        concentrations = concentrations.loc[:, (concentrations != 0).any(axis=0)]
        concentrations = concentrations.dropna(axis=1, how="all")
        uncertainties = uncertainties[concentrations.columns]

        print(f"Dataset shape: {concentrations.shape}")
        print(f"Removed {initial_cols - len(concentrations.columns)} empty columns")

        # Run PMF
        pmf = PMF(n_components=7, random_state=42)
        pmf.fit(concentrations, uncertainties)

        # Save results
        clean_name = dataset_name.replace(" ", "")

        contributions = pd.DataFrame(
            pmf.contributions_.values,
            index=concentrations.index,
            columns=[f"Factor_{i + 1}" for i in range(pmf.n_components)],
        )
        contributions.to_csv(output_dir / f"{clean_name}_factor_contributions.csv")

        profiles = pd.DataFrame(
            pmf.profiles_.values,
            columns=concentrations.columns,
            index=[f"Factor_{i + 1}" for i in range(pmf.n_components)],
        )
        profiles.to_csv(output_dir / f"{clean_name}_factor_profiles.csv")

        # Create summary plot
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # Factor profiles heatmap
        sns.heatmap(profiles, annot=False, cmap="viridis", ax=ax1)
        ax1.set_title("Factor Profiles")
        ax1.set_xlabel("Chemical Species")
        ax1.set_ylabel("Factors")

        # Contributions over time
        step = max(1, len(contributions) // 20)
        contributions_subset = contributions.iloc[::step, :]
        sns.heatmap(contributions_subset.T, annot=False, cmap="plasma", ax=ax2)
        ax2.set_title("Factor Contributions Over Time")

        # Factor correlation
        correlation_matrix = contributions.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", center=0, ax=ax3)
        ax3.set_title("Factor Correlations")

        # Top species for first factor
        top_species = profiles.iloc[0].nlargest(10)
        top_species.plot(kind="bar", ax=ax4)
        ax4.set_title("Top Species - Factor 1")
        ax4.set_xlabel("Chemical Species")
        ax4.tick_params(axis="x", rotation=45)

        plt.suptitle(
            f"{dataset_name} - PMF Analysis Summary", fontsize=16, fontweight="bold"
        )
        plt.tight_layout()
        plt.savefig(
            output_dir / f"{clean_name}_summary.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

        print("✓ Analysis complete! Files saved:")
        print(f"  - {clean_name}_factor_contributions.csv")
        print(f"  - {clean_name}_factor_profiles.csv")
        print(f"  - {clean_name}_summary.png")

        return True

    except Exception as e:
        print(f"✗ Error analyzing {dataset_name}: {e}")
        return False


def main():
    """Main function for interactive PMF analysis."""
    print("=== Quick PMF Analysis ===")
    print("This tool will help you analyze environmental datasets using PMF.")

    # Create output directory
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    # Get available datasets
    datasets = get_available_datasets()

    if not datasets:
        print("\nNo datasets found!")
        print(
            "Please ensure you have matching concentration and uncertainty files in the 'data' folder."
        )
        return

    # Select dataset
    selection = select_dataset(datasets)

    if not selection:
        return

    print(f"\nOutputs will be saved to: {output_dir.absolute()}")

    if selection == "all":
        print("\nRunning analysis on all datasets...")
        successful = 0
        for name, info in datasets.items():
            if analyze_single_dataset(name, info, output_dir):
                successful += 1

        print(f"\n{'=' * 50}")
        print(f"Completed analysis on {successful}/{len(datasets)} datasets")

    else:
        analyze_single_dataset(selection, datasets[selection], output_dir)

    print(f"\nAll outputs saved to: {output_dir.absolute()}")


if __name__ == "__main__":
    main()
