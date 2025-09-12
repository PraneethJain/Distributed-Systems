# gRPC Matrix Calculation Server (Rust/Tonic)

This project is a gRPC server written in Rust that provides a service for matrix calculations. A client can stream rows to build a matrix, and the server will compute and cache its rank and determinant for fast, repeated querying.

## How to Run

The server and client are separate binaries.

1.  **Run the Server:**
    The server will start and listen on `[::1]:50051`.
    ```bash
    cargo run --bin server
    ```

2.  **Run the Client:**
    In a separate terminal, run the client, providing a path to a text file containing its part of the matrix.

    ```bash
    cargo run --bin client -- <path_to_matrix_file> [client_id]
    ```
    -   `<path_to_matrix_file>`: A text file where each line contains a space-separated row of numbers. For example, you can use the `input/part1.txt` and `input/part2.txt` files.
    -   `[client_id]`: An optional identifier for the client.

    **Example:**
    ```bash
    cargo run --bin client -- input/part1.txt client_A
    ```
