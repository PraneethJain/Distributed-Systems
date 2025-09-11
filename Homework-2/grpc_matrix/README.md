# Matrix gRPC Server

A high-performance gRPC server in Rust that builds matrices from client rows and efficiently answers queries using cached results.

## Features

- **Bidirectional streaming** for matrix construction and queries
- **Automatic dimension detection** from first row
- **Pre-calculation and caching** of rank and determinant upon matrix completion
- **Thread-safe state management** using Arc<Mutex>
- **Custom matrix operations** without external math libraries
- **Reset functionality** to clear state and start new matrix

## Building and Running

### Prerequisites

- Rust 1.70+ (stable)
- Protocol Buffers compiler (protoc)

### Build

```bash
cargo build --release
```

### Run Server

```bash
cargo run --bin server
```

### Run Client Example

```bash
cargo run --bin client
```

## Architecture

- **server.rs**: Main gRPC server implementation with streaming handlers
- **state.rs**: Thread-safe matrix state management
- **matrix_ops.rs**: Custom implementations of rank (Gaussian elimination) and determinant (LU decomposition)
- **client.rs**: Example client demonstrating all server features

## Protocol

The server exposes two RPC methods:

1. **Interact** (Bidirectional Streaming): Handles row submissions and queries
2. **Reset** (Unary): Clears the matrix state

## Performance Optimizations

- Rank and determinant are calculated **once** when the matrix is complete
- All subsequent queries are answered from cached values
- No unnecessary recalculations
- Efficient matrix operations using in-place algorithms

## Testing

Run the test suite:

```bash
cargo test
```

## Example Usage

The included client demonstrates:

1. Building a 3x3 matrix row by row
2. Querying rank and determinant
3. Resetting the server state
4. Building a new 2x2 matrix
5. Batch operations with streaming
