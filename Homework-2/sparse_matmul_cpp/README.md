# C++/MPI Sparse Matrix Multiplication

This project implements a parallel sparse matrix-matrix multiplication ($C = A \times B$) in C++ using the Message Passing Interface (MPI).

It uses a primary-worker pattern where the primary process distributes rows of matrix A to workers, which then compute the corresponding rows of the output matrix C.

## How to Run

1.  **Compile:**
    ```bash
    mpic++ -O3 -march=native -std=c++17 main.cpp -o sparse_matmul_cpp
    ```

2.  **Execute:**
    ```bash
    mpirun -n <num_procs> --oversubscribe ./sparse_matmul_cpp <input_file>
    ```
    -   `<num_procs>`: The number of MPI processes to use (e.g., 8).
    -   `<input_file>`: Path to a file containing the sparse matrices.

    **Example using sample data:**
    ```bash
    mpirun -n 8 --oversubscribe ./sparse_matmul_cpp sample.txt
    ```

