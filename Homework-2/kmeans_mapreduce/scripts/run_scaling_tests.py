import subprocess
import time
import csv
import os
import re
from dataset_configs import dataset_configs

KMEANS_BINARY = "./target/release/kmeans_mapreduce"
OUTPUT_DIR_BASE = "./output/generated"
TEST_DATA_DIR = "./input/generated"
RESULTS_CSV = "scaling_results.csv"
MAX_ITERATIONS = 10000

core_counts = [1, 2, 4, 8]


def run_test(num_cores, points_file, centers_file, output_run_dir, num_centers_arg):
    command = [
        "mpirun",
        "-np",
        str(num_cores),
        KMEANS_BINARY,
        str(num_centers_arg),
        points_file,
        centers_file,
        str(MAX_ITERATIONS),
        output_run_dir,
    ]

    print(f"Running command: {' '.join(command)}")
    start_time = time.time()
    total_map_time = 0
    total_reduce_time = 0
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        end_time = time.time()
        duration = end_time - start_time
        print(f"Command finished in {duration:.4f} seconds")

        # Parse metrics from stdout
        for line in result.stdout.splitlines():
            if "[METRICS]" in line:
                match = re.search(r"map_time_ms=(\d+)", line)
                if match:
                    total_map_time += int(match.group(1))
                match = re.search(r"reduce_time_ms=(\d+)", line)
                if match:
                    total_reduce_time += int(match.group(1))

        return duration, total_map_time, total_reduce_time
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        return None, None, None
    except FileNotFoundError:
        print(
            f"Error: {KMEANS_BINARY} or mpirun not found. Make sure the binary is built and mpirun is in your PATH."
        )
        return None, None, None


if __name__ == "__main__":
    print("Building kmeans_mapreduce in release mode...")
    build_command = ["cargo", "build", "--release"]
    try:
        subprocess.run(build_command, check=True, capture_output=True, text=True)
        print("Build successful.")
    except subprocess.CalledProcessError as e:
        print(f"Build failed: {e}")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        exit(1)
    except FileNotFoundError:
        print(
            "Error: cargo command not found. Make sure Rust is installed and in your PATH."
        )
        exit(1)

    with open(RESULTS_CSV, "w", newline="") as csvfile:
        fieldnames = [
            "dataset_id",
            "num_points",
            "num_centers",
            "num_cores",
            "time_seconds",
            "total_map_time_ms",
            "total_reduce_time_ms",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for i, (num_points, num_centers, _) in enumerate(dataset_configs):
            dataset_id = i + 1
            dataset_path = os.path.join(TEST_DATA_DIR, f"dataset_{dataset_id}")
            points_file = os.path.join(dataset_path, "points.txt")
            centers_file = os.path.join(dataset_path, "centers.txt")

            if not os.path.exists(points_file) or not os.path.exists(centers_file):
                print(
                    f"Warning: Dataset {dataset_id} not found at {dataset_path}. Skipping."
                )
                continue

            for num_cores in core_counts:
                output_run_dir = os.path.join(
                    OUTPUT_DIR_BASE, f"dataset_{dataset_id}_cores_{num_cores}"
                )
                os.makedirs(output_run_dir, exist_ok=True)

                print(
                    f"\n--- Running Dataset {dataset_id} ({num_points} points, {num_centers} centers) with {num_cores} cores ---"
                )
                duration, map_time, reduce_time = run_test(
                    num_cores, points_file, centers_file, output_run_dir, num_centers
                )

                if duration is not None:
                    writer.writerow(
                        {
                            "dataset_id": dataset_id,
                            "num_points": num_points,
                            "num_centers": num_centers,
                            "num_cores": num_cores,
                            "time_seconds": duration,
                            "total_map_time_ms": map_time,
                            "total_reduce_time_ms": reduce_time,
                        }
                    )
                    csvfile.flush()

    print(f"\nScaling tests complete. Results saved to {RESULTS_CSV}")
