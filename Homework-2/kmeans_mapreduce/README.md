# K-Means via MapReduce (Rust/MPI)

This project implements the K-Means clustering algorithm in Rust, using MPI to simulate a MapReduce parallelism model.

The primary process acts as the coordinator, scattering data points to workers. Each iteration involves a parallel "map" phase where points are assigned to clusters, and a "reduce" phase where new cluster centroids are calculated.

## How to Run

1.  **Build:**
    Build the project in release mode using Cargo.
    ```bash
    cargo build --release
    ```

2.  **Execute:**
    Run the compiled binary using `mpirun`, providing the necessary arguments.
    ```bash
    mpirun -np <num_procs> ./target/release/kmeans_mapreduce <num_centers> <points_file> <centers_file> <max_iter> <out_dir>
    ```
    -   `<num_procs>`: The number of MPI processes (e.g., 4).
    -   `<num_centers>`: The number of clusters to find (e.g., 3).
    -   `<points_file>`: Path to the data points file.
    -   `<centers_file>`: Path to the initial centers file.
    -   `<max_iter>`: Maximum number of iterations (e.g., 100).
    -   `<out_dir>`: Directory to save the output files.

    **Example using sample data:**
    ```bash
    mpirun -np 4 ./target/release/kmeans_mapreduce 3 input/sample/points.txt input/sample/centers.txt 100 output/sample
    ```

    or 
    
    ```bash
    ./run_km_mapreduce.sh 3 input/sample/points.txt input/sample/centers.txt 100 output/sample
    ```
