import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- Configuration ---
RESULTS_FILE = "slurm_scaling_results.csv"
PLOTS_DIR = "plots"


# --- Main Script Logic ---
def main():
    """
    Reads the benchmark results and generates scaling analysis plots.
    """
    # --- Data Loading and Preparation ---
    if not os.path.exists(RESULTS_FILE):
        print(f"Error: Results file '{RESULTS_FILE}' not found.")
        print("Please run the 'run_benchmark.py' script first.")
        return

    print(f"Reading results from '{RESULTS_FILE}'...")
    df = pd.read_csv(RESULTS_FILE)

    # Calculate the mean execution time for each configuration
    avg_df = (
        df.groupby(["matrix_size", "processor_count"])[
            ["execution_time_seconds", "total_time_seconds"]
        ]
        .mean()
        .reset_index()
    )

    # Get the baseline time (time with 1 processor) for each matrix size to calculate speedup
    # We base speedup on the time reported by the application itself
    time_p1 = avg_df[avg_df["processor_count"] == 1].set_index("matrix_size")[
        "execution_time_seconds"
    ]
    avg_df["baseline_time"] = avg_df["matrix_size"].map(time_p1)

    # Calculate Speedup and Efficiency
    avg_df["speedup"] = avg_df["baseline_time"] / avg_df["execution_time_seconds"]
    avg_df["efficiency"] = avg_df["speedup"] / avg_df["processor_count"]

    print("Data processing complete. Generating plots...")
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # Use a nice style for the plots
    sns.set_theme(style="whitegrid")

    # --- Plot 1: Execution Time vs. Processors (Strong Scaling) ---
    # Melt the dataframe to plot both execution time and total time
    time_df = avg_df.melt(
        id_vars=["matrix_size", "processor_count"],
        value_vars=["execution_time_seconds", "total_time_seconds"],
        var_name="time_type",
        value_name="seconds",
    )
    time_df["time_type"] = time_df["time_type"].map(
        {
            "execution_time_seconds": "App-Reported Time",
            "total_time_seconds": "Total Wall-Clock Time",
        }
    )

    plt.figure(figsize=(12, 7))
    sns.lineplot(
        data=time_df,
        x="processor_count",
        y="seconds",
        hue="matrix_size",
        style="time_type",
        marker="o",
        palette="viridis",
    )
    plt.title("Execution Time vs. Number of Processes (Strong Scaling)")
    plt.xlabel("Number of Processes")
    plt.ylabel("Average Time (seconds)")
    plt.xscale("log", base=2)
    plt.yscale("log")
    plt.xticks(avg_df["processor_count"].unique())
    plt.legend(title="Matrix Size (N x N) / Time Type")
    plot1_path = os.path.join(PLOTS_DIR, "1_execution_time_comparison.png")
    plt.savefig(plot1_path)
    print(f"Saved execution time comparison plot to '{plot1_path}'")
    plt.close()

    # --- Plot 2: Speedup vs. Processors ---
    plt.figure(figsize=(10, 6))
    # Plot ideal speedup line
    proc_counts = sorted(avg_df["processor_count"].unique())
    plt.plot(proc_counts, proc_counts, "k--", label="Ideal Speedup")

    sns.lineplot(
        data=avg_df,
        x="processor_count",
        y="speedup",
        hue="matrix_size",
        marker="o",
        palette="viridis",
    )
    plt.title("Speedup vs. Number of Processes")
    plt.xlabel("Number of Processes")
    plt.ylabel("Speedup (T_1 / T_p)")
    plt.xscale("log", base=2)
    plt.xticks(proc_counts)
    plt.legend(title="Matrix Size (N x N)")
    plot2_path = os.path.join(PLOTS_DIR, "2_speedup.png")
    plt.savefig(plot2_path)
    print(f"Saved speedup plot to '{plot2_path}'")
    plt.close()

    # --- Plot 3: Efficiency vs. Processors ---
    plt.figure(figsize=(10, 6))
    # Plot ideal efficiency line
    plt.axhline(1.0, color="k", linestyle="--", label="Ideal Efficiency (100%)")

    sns.lineplot(
        data=avg_df,
        x="processor_count",
        y="efficiency",
        hue="matrix_size",
        marker="o",
        palette="viridis",
    )
    plt.title("Efficiency vs. Number of Processes")
    plt.xlabel("Number of Processes")
    plt.ylabel("Efficiency (Speedup / P)")
    plt.xscale("log", base=2)
    plt.xticks(proc_counts)
    plt.ylim(0, 1.2)  # Efficiency can sometimes be > 1 due to cache effects
    plt.legend(title="Matrix Size (N x N)")
    plot3_path = os.path.join(PLOTS_DIR, "3_efficiency.png")
    plt.savefig(plot3_path)
    print(f"Saved efficiency plot to '{plot3_path}'")
    plt.close()

    # --- Plot 4: Overhead Analysis ---
    avg_df["overhead_seconds"] = (
        avg_df["total_time_seconds"] - avg_df["execution_time_seconds"]
    )
    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=avg_df,
        x="processor_count",
        y="overhead_seconds",
        hue="matrix_size",
        marker="o",
        palette="viridis",
    )
    plt.title("Process & Communication Overhead vs. Number of Processes")
    plt.xlabel("Number of Processes")
    plt.ylabel("Overhead (Total Time - App Time) (seconds)")
    plt.xscale("log", base=2)
    plt.xticks(proc_counts)
    plt.legend(title="Matrix Size (N x N)")
    plot4_path = os.path.join(PLOTS_DIR, "4_overhead.png")
    plt.savefig(plot4_path)
    print(f"Saved overhead analysis plot to '{plot4_path}'")
    plt.close()

    try:
        efficiency_pivot = avg_df.pivot(
            index="matrix_size", columns="processor_count", values="efficiency"
        )
        plt.figure(figsize=(12, 8))
        sns.heatmap(
            efficiency_pivot, annot=True, fmt=".2f", cmap="viridis", linewidths=0.5
        )
        plt.title("Parallel Efficiency Heatmap")
        plt.xlabel("Number of Processes")
        plt.ylabel("Matrix Size (N x N)")
        plot5_path = os.path.join(PLOTS_DIR, "5_efficiency_heatmap.png")
        plt.savefig(plot5_path)
        print(f"Saved efficiency heatmap to '{plot5_path}'")
        plt.close()
    except Exception as e:
        print(
            f"Could not generate heatmap. It might require a complete grid of data points. Error: {e}"
        )

    print("\nAll plots generated and saved in the 'plots' directory.")


if __name__ == "__main__":
    main()
