import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os

# Attempt to import scipy, which is needed for the 3D surface plot.
try:
    from scipy.interpolate import griddata

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

RESULTS_CSV = "slurm_scaling_results.csv"
PLOTS_DIR = "slurm_plots"


def analyze_and_plot(results_file):
    """
    Reads the scaling results, performs a comprehensive analysis,
    and generates a variety of insightful plots.
    """
    if not os.path.exists(results_file):
        print(
            f"Error: Results file '{results_file}' not found. Please run run_scaling_tests.py first."
        )
        return

    df = pd.read_csv(results_file)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    print("--- Comprehensive Performance Analysis ---")
    print("\n1. Raw Data Head:")
    print(df.head())
    print("\n2. Descriptive Statistics:")
    print(df.describe())

    # --- Core Analysis: Calculate Speedup and Efficiency ---
    df = df.sort_values(by=["dataset_id", "num_cores"])
    # Calculate time for 1 core for each dataset to use as a baseline for speedup
    df["time_1_core"] = df.groupby("dataset_id")["time_seconds"].transform(
        lambda x: x.iloc[0]
    )
    df["speedup"] = df["time_1_core"] / df["time_seconds"]
    df["efficiency"] = df["speedup"] / df["num_cores"]

    print("\n3. Data with Speedup and Efficiency Metrics:")
    print(
        df[["dataset_id", "num_cores", "time_seconds", "speedup", "efficiency"]].head()
    )

    # --- Map/Reduce Time Analysis ---
    if "total_map_time_ms" in df.columns and "total_reduce_time_ms" in df.columns:
        df["map_time_s"] = df["total_map_time_ms"] / 1000
        df["reduce_time_s"] = df["total_reduce_time_ms"] / 1000
        df["other_time_s"] = df["time_seconds"] - df["map_time_s"] - df["reduce_time_s"]

        # Calculate proportions
        df["map_prop"] = df["map_time_s"] / df["time_seconds"]
        df["reduce_prop"] = df["reduce_time_s"] / df["time_seconds"]
        df["other_prop"] = df["other_time_s"] / df["time_seconds"]

        print("\n4. Map/Reduce/Other Time Breakdown (in seconds):")
        print(
            df[
                [
                    "dataset_id",
                    "num_cores",
                    "map_time_s",
                    "reduce_time_s",
                    "other_time_s",
                ]
            ].head()
        )

    # --- Plotting ---
    print(f"\n--- Generating Plots in '{PLOTS_DIR}' Directory ---")

    plot_3d_surface(df)
    plot_combined_time_vs_cores(df)
    plot_combined_speedup_vs_cores(df)
    plot_combined_efficiency_vs_cores(df)

    if "map_time_s" in df.columns:
        plot_time_breakdown_stacked_bar(df)

    print(f"\nAnalysis complete. All plots saved to the '{PLOTS_DIR}' directory.")


def plot_3d_surface(df):
    """Plots a 3D surface of time vs. points and cores."""
    if not SCIPY_AVAILABLE:
        print(
            "\nSkipping 3D surface plot: `scipy` library not found. Please install it (`pip install scipy`)."
        )
        # Fallback to scatter plot
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(
            df["num_points"],
            df["num_cores"],
            df["time_seconds"],
            c=df["time_seconds"],
            cmap="viridis",
            s=50,
        )
        ax.set_xlabel("Number of Points")
        ax.set_ylabel("Number of Cores")
        ax.set_zlabel("Time (seconds)")
        ax.set_title("K-Means Scaling: Time vs. Points and Cores (Scatter)")
        plt.savefig(os.path.join(PLOTS_DIR, "3d_time_points_cores_scatter.png"))
        plt.close()
        print("Generated 3D scatter plot as a fallback.")
        return

    fig = plt.figure(figsize=(14, 12))
    ax = fig.add_subplot(111, projection="3d")

    # Create a grid for interpolation
    x = df["num_points"]
    y = df["num_cores"]
    z = df["time_seconds"]
    xi = np.linspace(x.min(), x.max(), 100)
    yi = np.linspace(y.min(), y.max(), 100)
    X, Y = np.meshgrid(xi, yi)

    # Interpolate the scattered data onto the grid
    Z = griddata((x, y), z, (X, Y), method="cubic")

    surf = ax.plot_surface(X, Y, Z, cmap="viridis", edgecolor="none")

    ax.set_xlabel("Number of Points", labelpad=15)
    ax.set_ylabel("Number of Cores", labelpad=15)
    ax.set_zlabel("Time (seconds)", labelpad=15)
    ax.set_title("K-Means Performance Surface: Time vs. Points and Cores", pad=20)
    fig.colorbar(surf, shrink=0.5, aspect=5, label="Time (s)")

    plt.savefig(os.path.join(PLOTS_DIR, "3d_performance_surface.png"))
    plt.close()
    print("Generated 3D performance surface plot.")


def plot_combined_time_vs_cores(df):
    """Plots execution time vs. number of cores for all datasets on one graph."""
    plt.figure(figsize=(12, 8))
    for dataset_id in sorted(df["dataset_id"].unique()):
        subset = df[df["dataset_id"] == dataset_id]
        label = f"Dataset {dataset_id} ({subset['num_points'].iloc[0]} points)"
        plt.plot(
            subset["num_cores"],
            subset["time_seconds"],
            marker="o",
            linestyle="-",
            label=label,
        )

    plt.xlabel("Number of Cores")
    plt.ylabel("Execution Time (seconds)")
    plt.title("Execution Time vs. Number of Cores (Strong Scaling)")
    plt.legend(title="Dataset Size")
    plt.grid(True, which="both", linestyle="--")
    plt.xscale("log", base=2)
    plt.yscale("log")
    plt.xticks(df["num_cores"].unique(), df["num_cores"].unique())
    plt.savefig(os.path.join(PLOTS_DIR, "combined_time_vs_cores.png"))
    plt.close()
    print("Generated combined 'Time vs. Cores' plot.")


def plot_combined_speedup_vs_cores(df):
    """Plots speedup vs. number of cores for all datasets on one graph."""
    plt.figure(figsize=(12, 8))
    # Ideal speedup line
    cores = sorted(df["num_cores"].unique())
    plt.plot(cores, cores, "k--", label="Ideal Speedup")

    for dataset_id in sorted(df["dataset_id"].unique()):
        subset = df[df["dataset_id"] == dataset_id]
        label = f"Dataset {dataset_id} ({subset['num_points'].iloc[0]} points)"
        plt.plot(
            subset["num_cores"],
            subset["speedup"],
            marker="o",
            linestyle="-",
            label=label,
        )

    plt.xlabel("Number of Cores")
    plt.ylabel("Speedup (T_1 / T_p)")
    plt.title("Speedup vs. Number of Cores")
    plt.legend(title="Dataset Size")
    plt.grid(True, which="both", linestyle="--")
    plt.xticks(cores, cores)
    plt.savefig(os.path.join(PLOTS_DIR, "combined_speedup_vs_cores.png"))
    plt.close()
    print("Generated combined 'Speedup vs. Cores' plot.")


def plot_combined_efficiency_vs_cores(df):
    """Plots efficiency vs. number of cores for all datasets on one graph."""
    plt.figure(figsize=(12, 8))
    cores = sorted(df["num_cores"].unique())
    plt.axhline(y=1.0, color="k", linestyle="--", label="Ideal Efficiency (100%)")

    for dataset_id in sorted(df["dataset_id"].unique()):
        subset = df[df["dataset_id"] == dataset_id]
        label = f"Dataset {dataset_id} ({subset['num_points'].iloc[0]} points)"
        plt.plot(
            subset["num_cores"],
            subset["efficiency"],
            marker="o",
            linestyle="-",
            label=label,
        )

    plt.xlabel("Number of Cores")
    plt.ylabel("Efficiency (Speedup / Cores)")
    plt.title("Parallel Efficiency vs. Number of Cores")
    plt.legend(title="Dataset Size")
    plt.grid(True, which="both", linestyle="--")
    plt.xticks(cores, cores)
    plt.ylim(0, 1.1)
    plt.savefig(os.path.join(PLOTS_DIR, "combined_efficiency_vs_cores.png"))
    plt.close()
    print("Generated combined 'Efficiency vs. Cores' plot.")


def plot_time_breakdown_stacked_bar(df):
    """Generates a stacked bar chart showing the proportion of time for each phase."""
    unique_datasets = df["dataset_id"].unique()
    datasets_to_plot = [unique_datasets[0], unique_datasets[-1]]  # Smallest and largest

    for dataset_id in datasets_to_plot:
        subset = df[df["dataset_id"] == dataset_id].set_index("num_cores")
        num_points = subset["num_points"].iloc[0]

        proportions = subset[["map_prop", "reduce_prop", "other_prop"]]
        proportions.columns = ["Map Phase", "Reduce Phase", "Other (I/O, etc.)"]

        ax = proportions.plot(
            kind="bar", stacked=True, figsize=(12, 8), colormap="viridis", alpha=0.8
        )

        plt.xlabel("Number of Cores")
        plt.ylabel("Proportion of Total Execution Time")
        plt.title(
            f"Proportional Time Breakdown by Phase (Dataset {dataset_id}: {num_points} points)"
        )
        plt.xticks(rotation=0)
        plt.legend(title="Phase")
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter("{:.0%}".format)
        )  # Format y-axis as percentage

        plt.tight_layout()
        plt.savefig(
            os.path.join(PLOTS_DIR, f"proportional_breakdown_dataset_{dataset_id}.png")
        )
        plt.close()
        print(f"Generated proportional time breakdown plot for dataset {dataset_id}.")


if __name__ == "__main__":
    analyze_and_plot(RESULTS_CSV)
