from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from pmf import PMF

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

# Select dataset to analyze (you can change this)
dataset_name = (
    "BatonRouge"  # Change to "StLouis" or "Baltimore" to analyze other datasets
)
dataset = datasets[dataset_name]

print(f"Analyzing {dataset_name} dataset...")

# Load the concentration and uncertainty data
print(f"Loading {dataset_name} dataset...")
try:
    if dataset["concentrations"].endswith(".csv"):
        concentrations = pd.read_csv(dataset["concentrations"])
        uncertainties = pd.read_csv(dataset["uncertainties"])
    else:
        # For .txt files, assume tab-separated
        concentrations = pd.read_csv(dataset["concentrations"], sep="\t")
        uncertainties = pd.read_csv(dataset["uncertainties"], sep="\t")

    print(f"Successfully loaded {dataset_name} dataset")
    print(f"Concentrations shape: {concentrations.shape}")
    print(f"Uncertainties shape: {uncertainties.shape}")

except FileNotFoundError:
    print(f"Error: Could not find data files for {dataset_name}")
    print("Make sure the following files exist:")
    print(f"- {dataset['concentrations']}")
    print(f"- {dataset['uncertainties']}")
    exit(1)
except Exception as e:
    print(f"Error loading {dataset_name} dataset: {e}")
    exit(1)

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
# We'll start with 7 factors, a common starting point
pmf = PMF(n_components=7, random_state=42)

# Fit the model to the data
pmf.fit(concentrations, uncertainties)

# Save the results
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

# Create heatmap visualizations
print("Creating heatmap visualizations...")

# Set up the plotting style
plt.style.use("default")
sns.set_palette("viridis")

# 1. Factor Profiles Heatmap
fig, ax = plt.subplots(figsize=(16, 8))
sns.heatmap(
    profiles, annot=False, cmap="viridis", cbar_kws={"label": "Concentration"}, ax=ax
)
plt.title(
    "PMF Factor Profiles - Chemical Species Composition", fontsize=14, fontweight="bold"
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
fig, ax = plt.subplots(figsize=(12, 10))
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
plt.title("PMF Factor Contributions Over Time", fontsize=14, fontweight="bold")
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
fig, ax = plt.subplots(figsize=(8, 6))
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
plt.title("Factor Correlation Matrix", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(
    output_dir / f"{dataset_name}_factor_correlation_heatmap.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close()

# 4. Top species contributions for each factor
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
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

plt.suptitle("Top Chemical Species by PMF Factor", fontsize=16, fontweight="bold")
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
    fig, ax = plt.subplots(figsize=(14, 8))

    # Get the factor profile data
    factor_data = profiles.loc[factor_name]
    species_names = factor_data.index
    concentrations = factor_data.values

    # Calculate percentage of each species explained by this factor
    # For each species, calculate what % this factor contributes
    # to the total across all factors
    total_species_across_factors = profiles.sum(
        axis=0
    )  # Sum each species across all factors
    percentages = (concentrations / total_species_across_factors) * 100
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
        x_pos, concentrations, color="lightblue", edgecolor="navy", linewidth=0.5
    )

    # Create secondary y-axis for percentages
    ax2 = ax.twinx()

    # Add percentage markers (red squares)
    ax2.scatter(x_pos, percentages, color="red", marker="s", s=30, zorder=5)

    # Customize the plot
    ax.set_xlabel("Chemical Species", fontsize=12, fontweight="bold")
    ax.set_ylabel("Concentration", fontsize=12, fontweight="bold", color="blue")
    ax2.set_ylabel("% of Species Total", fontsize=12, fontweight="bold", color="red")

    # Set logarithmic scale for concentrations (like EPA PMF)
    ax.set_yscale("log")
    ax.set_ylim(
        bottom=max(0.001, concentrations[concentrations > 0].min() * 0.1)
        if any(concentrations > 0)
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
    plt.title(f"Factor Profile - {factor_name}", fontsize=14, fontweight="bold", pad=20)

    # Add legend
    from matplotlib.lines import Line2D

    legend_elements = [
        plt.Rectangle(
            (0, 0), 1, 1, facecolor="lightblue", edgecolor="navy", label="Concentration"
        ),
        Line2D(
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
    fig, ax = plt.subplots(figsize=(14, 6))

    # Get the factor contributions over time
    factor_contributions = contributions[factor_name]
    dates = contributions.index

    # Convert dates to datetime if they're strings
    if isinstance(dates[0], str):
        dates = pd.to_datetime(dates)

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
        f"Factor Contributions - {factor_name}", fontsize=14, fontweight="bold", pad=20
    )

    # Add grid
    ax.grid(True, alpha=0.3)

    # Format x-axis dates
    ax.tick_params(axis="x", rotation=45)

    # Set y-axis to start from 0
    ax.set_ylim(bottom=0)

    # Add some styling to match EPA PMF
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    factor_filename = f"{factor_name.lower().replace(' ', '_')}"
    plt.savefig(
        output_dir / f"{dataset_name}_factor_contributions_{factor_filename}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

# 7. Combined Factor Profiles Overview (all factors in one plot)
fig, ax = plt.subplots(figsize=(16, 10))

# Create a grouped bar chart
n_species = len(profiles.columns)
n_factors = len(profiles.index)
x_pos = np.arange(n_species)
width = 0.8 / n_factors  # Width of each bar

colors = plt.cm.Set3(np.linspace(0, 1, n_factors))

for i, (factor_name, factor_data) in enumerate(profiles.iterrows()):
    offset = (i - n_factors / 2 + 0.5) * width
    bars = ax.bar(
        x_pos + offset,
        factor_data.values,
        width,
        label=factor_name,
        color=colors[i],
        alpha=0.8,
    )

ax.set_xlabel("Chemical Species", fontsize=12, fontweight="bold")
ax.set_ylabel("Concentration", fontsize=12, fontweight="bold")
ax.set_title(
    "All Factor Profiles - Chemical Species Composition", fontsize=14, fontweight="bold"
)
ax.set_xticks(x_pos)
ax.set_xticklabels(profiles.columns, rotation=45, ha="right")
ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
ax.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig(
    output_dir / f"{dataset_name}_all_factor_profiles_overview.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close()

# 8. Combined Factor Contributions Overview (all factors in one plot)
fig, ax = plt.subplots(figsize=(14, 8))

dates = contributions.index
if isinstance(dates[0], str):
    dates = pd.to_datetime(dates)

colors = plt.cm.Set3(np.linspace(0, 1, len(contributions.columns)))

for i, factor_name in enumerate(contributions.columns):
    ax.plot(
        dates,
        contributions[factor_name],
        label=factor_name,
        color=colors[i],
        linewidth=1.5,
        marker="o",
        markersize=2,
    )

ax.set_xlabel("Date", fontsize=12, fontweight="bold")
ax.set_ylabel("Normalized Contributions", fontsize=12, fontweight="bold")
ax.set_title("All Factor Contributions Over Time", fontsize=14, fontweight="bold")
ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
ax.grid(True, alpha=0.3)
ax.tick_params(axis="x", rotation=45)
ax.set_ylim(bottom=0)

plt.tight_layout()
plt.savefig(
    output_dir / f"{dataset_name}_all_factor_contributions_overview.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close()

print("EPA PMF-style visualizations saved:")
print(f"- Individual factor profile plots: {dataset_name}_factor_profile_factor_X.png")
print(
    f"- Individual factor contribution plots: "
    f"{dataset_name}_factor_contributions_factor_X.png"
)
print(f"- {dataset_name}_all_factor_profiles_overview.png")
print(f"- {dataset_name}_all_factor_contributions_overview.png")

print(f"\nAnalysis complete for {dataset_name} dataset!")
print(f"All outputs saved to: {output_dir.absolute()}")
print(
    "\nTo analyze a different dataset, change the 'dataset_name' variable "
    "at the top of the script to:"
)
print("- 'BatonRouge'")
print("- 'StLouis'")
print("- 'Baltimore'")

# Optional: Analyze all datasets automatically
# Uncomment the code below to run PMF analysis on all available datasets

# print("\n" + "="*50)
# print("ANALYZING ALL DATASETS")
# print("="*50)
#
# for name in datasets.keys():
#     if name != dataset_name:  # Skip the one we already analyzed
#         print(f"\nAnalyzing {name} dataset...")
#         # You would need to wrap the analysis code in a function to reuse it here
