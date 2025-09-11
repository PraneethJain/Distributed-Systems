import os
import numpy as np
from scipy.sparse import random, csr_matrix
import io

# --- Configuration ---
OUTPUT_DIR = "input/generated"
DIMENSIONS = [1024, 2048, 4096, 8192, 16384]
SPARSITY = 0.01


# --- Helper Function from run_random_test.py ---
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


# --- Main Script Logic ---
def main():
    """
    Generates sparse matrix input files for the MPI benchmark.
    """
    print(f"Creating output directory: {OUTPUT_DIR}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for n in DIMENSIONS:
        print(f"Generating {n}x{n} matrix with {SPARSITY*100:.1f}% sparsity...")

        # For sparse matmul A(NxM) * B(MxP), we use square matrices, so N=M=P
        m, p = n, n

        # Generate two random sparse matrices
        matrix_a = random(n, m, density=SPARSITY, format="csr", dtype=np.float64)
        matrix_b = random(m, p, density=SPARSITY, format="csr", dtype=np.float64)

        filename = os.path.join(OUTPUT_DIR, f"matrix_{n}x{n}_sparsity{SPARSITY}.txt")

        print(f"Writing to file: {filename}")
        with open(filename, "w") as f:
            # Write dimensions header
            f.write(f"{n} {m} {p}\n")

            # Write matrix A data
            f.write(format_matrix_for_rust(matrix_a))

            # Write matrix B data
            f.write(format_matrix_for_rust(matrix_b))

    print("\nInput generation complete.")
    print(f"Files are located in the '{OUTPUT_DIR}' directory.")


if __name__ == "__main__":
    main()
