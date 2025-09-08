#!/bin/bash

cargo build --release
mpirun -np 5 target/release/kmeans_mapreduce $@