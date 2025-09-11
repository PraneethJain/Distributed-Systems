import os
import glob
import subprocess
import re
import csv
import sys
import time

# --- Configuration ---
INPUT_DIR = "input/generated"
RESULTS_FILE = "benchmark_results.csv"
PROCESSOR_COUNTS = [1, 2, 4, 8]
NUM_REPETITIONS = 1
RUST_EXECUTABLE_PATH = "./a.out"
MPI_RUN_COMMAND = "mpirun"

# --- Main Script Logic ---
def main():
    """
    Runs the MPI benchmark, collects timing data, and saves it to a CSV file.
    """
    # Find input files
    input_files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.txt")))
    if not input_files:
        print(f"Error: No input files found in '{INPUT_DIR}'.")
        print("Please run the 'generate_inputs.py' script first.")
        sys.exit(1)

    print(f"Found {len(input_files)} input files to benchmark.")

    # Prepare CSV file
    with open(RESULTS_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["matrix_size", "processor_count", "run_number", "execution_time_seconds", "total_time_seconds"])

    # Regex to parse timing info from stderr
    time_regex = re.compile(r"ExecutionTime: (\d+\.\d+)")

    # Run benchmarks
    for input_file in input_files:
        # Extract matrix size from filename, e.g., "matrix_1024x1024_..." -> 1024
        try:
            matrix_size = int(os.path.basename(input_file).split("_")[1].split("x")[0])
        except (IndexError, ValueError):
            print(f"Warning: Could not parse matrix size from filename '{input_file}'. Skipping.")
            continue

        for p_count in PROCESSOR_COUNTS:
            print(f"\nBenchmarking {input_file} with {p_count} processes...")
            
            for run_num in range(1, NUM_REPETITIONS + 1):
                print(f"  -> Run {run_num}/{NUM_REPETITIONS}", end="", flush=True)
                
                command = [
                    MPI_RUN_COMMAND,
                    "-n",
                    str(p_count),
                    "--oversubscribe",
                    RUST_EXECUTABLE_PATH,
                    input_file,
                ]

                try:
                    start_time = time.time()
                    proc = subprocess.run(
                        command, capture_output=True, text=True, check=True
                    )
                    end_time = time.time()
                    total_time = end_time - start_time
                    
                    # Timing is printed to stderr
                    match = time_regex.search(proc.stderr)
                    if not match:
                        print(f"\nError: Could not parse execution time for P={p_count} on {input_file}.")
                        print("Stderr:", proc.stderr)
                        continue

                    exec_time = float(match.group(1))
                    print(f" - Time: {exec_time:.4f}s - Total: {total_time:.4f}s")

                    # Append result to CSV
                    with open(RESULTS_FILE, "a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow([matrix_size, p_count, run_num, exec_time, total_time])

                except FileNotFoundError:
                    print(f"\nError: Command not found. Ensure '{MPI_RUN_COMMAND}' and '{RUST_EXECUTABLE_PATH}' are correct.")
                    sys.exit(1)
                except subprocess.CalledProcessError as e:
                    print("\n--- MPI Program failed with an error ---")
                    print(f"Return Code: {e.returncode}")
                    print("--- Stderr ---")
                    print(e.stderr)
                    # Continue to next benchmark
                    break 

    print(f"\nBenchmark complete. Results saved to '{RESULTS_FILE}'.")

if __name__ == "__main__":
    main()
