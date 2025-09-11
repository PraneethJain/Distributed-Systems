import subprocess
import numpy as np
from scipy.sparse import random, csr_matrix
import io
import sys
import tempfile
import os

# RUST_EXECUTABLE_PATH = "./a.out"
RUST_EXECUTABLE_PATH = "./target/release/sparse_matmul"
MPI_RUN_COMMAND = "mpirun"
NUM_MPI_PROCESSES = 8

N, M, P = 10001, 10001, 10001
DENSITY = 0.01
TOLERANCE = 1e-9


def generate_sparse_matrix(rows, cols, density):
    return random(rows, cols, density=density, format="csr", dtype=np.float64)


def format_matrix_for_rust(matrix: csr_matrix) -> str:
    output = io.StringIO()
    for i in range(matrix.shape[0]):
        start, end = matrix.indptr[i], matrix.indptr[i + 1]
        row_indices = matrix.indices[start:end]
        row_data = matrix.data[start:end]

        output.write(str(len(row_indices)))
        for col, val in zip(row_indices, row_data):
            output.write(f" {col} {val}")
        output.write("\n")
    return output.getvalue()


def parse_rust_output(output_str: str, n_rows: int, n_cols: int) -> csr_matrix:
    data, indices, indptr = [], [], [0]
    lines = output_str.strip().split("\n")

    if not output_str.strip():
        return csr_matrix((n_rows, n_cols), dtype=np.float64)

    if len(lines) != n_rows:
        raise ValueError(
            f"Output format error: Expected {n_rows} rows, but got {len(lines)}"
        )

    for line in lines:
        parts = line.split()
        k = int(parts[0])
        for i in range(k):
            indices.append(int(parts[2 * i + 1]))
            data.append(float(parts[2 * i + 2]))
        indptr.append(len(data))

    return csr_matrix((data, indices, indptr), shape=(n_rows, n_cols), dtype=np.float64)


def compare_matrices(
    matrix1: csr_matrix, matrix2: csr_matrix, tolerance: float
) -> bool:
    if matrix1.shape != matrix2.shape:
        print(f"Error: Shape mismatch! Expected {matrix1.shape}, Got {matrix2.shape}")
        return False
    diff = abs(matrix1 - matrix2)
    if diff.nnz > 0:
        max_diff = diff.max()
        if max_diff > tolerance:
            print(
                f"Verification FAILED: Maximum difference is {max_diff}, which is > tolerance {tolerance}"
            )
            return False
    return True


if __name__ == "__main__":
    print(
        f"Generating sparse matrices with dimensions A({N}x{M}) and B({M}x{P}) and density {DENSITY}..."
    )
    matrix_a = generate_sparse_matrix(N, M, DENSITY)
    matrix_b = generate_sparse_matrix(M, P, DENSITY)
    print("Matrix generation complete.\n")

    print("Calculating expected result using Scipy...")
    expected_c = matrix_a.dot(matrix_b)
    print("Scipy calculation complete.\n")

    with tempfile.NamedTemporaryFile(
        mode="w", delete=False, suffix=".txt"
    ) as tmp_input_file:
        print(f"Writing input to temporary file: {tmp_input_file.name}...")
        tmp_input_file.write(f"{N} {M} {P}\n")
        tmp_input_file.write(format_matrix_for_rust(matrix_a))
        tmp_input_file.write(format_matrix_for_rust(matrix_b))
        temp_filename = tmp_input_file.name
    print("Input writing complete.\n")

    command = [
        MPI_RUN_COMMAND,
        "-n",
        str(NUM_MPI_PROCESSES),
        RUST_EXECUTABLE_PATH,
        temp_filename,
    ]

    print(f"--- Running MPI Program with command: {' '.join(command)} ---")

    try:
        proc = subprocess.run(command, capture_output=True, text=True, check=True)

        if proc.stderr:
            print("\n--- Warnings or messages on Stderr ---")
            print(proc.stderr)

    except FileNotFoundError:
        print(
            f"Error: Command not found. Ensure '{MPI_RUN_COMMAND}' and '{RUST_EXECUTABLE_PATH}' are correct."
        )
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print("\n--- Rust Program failed with an error ---", file=sys.stderr)
        print(f"Return Code: {e.returncode}", file=sys.stderr)
        print("--- Stdout ---", file=sys.stderr)
        print(e.stdout, file=sys.stderr)
        print("--- Stderr ---", file=sys.stderr)
        print(e.stderr, file=sys.stderr)
        sys.exit(1)
    finally:
        os.remove(temp_filename)

    print("--- MPI Program Finished ---\n")

    print("Parsing Rust program output...")
    actual_c = parse_rust_output(proc.stdout, N, P)
    print("Parsing complete.\n")

    print("Verifying the result...")
    if compare_matrices(expected_c, actual_c, TOLERANCE):
        print("✅ SUCCESS: The output from the Rust program is correct!")
    else:
        print("❌ FAILURE: The output from the Rust program is incorrect.")
