#!/usr/bin/env python3
"""
K-Means MapReduce Validation and Benchmark Script

This script validates the output of the MPI K-Means implementation by:
1. Checking that final centers match sklearn's K-Means results
2. Verifying that all points are assigned to their nearest centers
3. Computing quality metrics (inertia, silhouette score)
4. Measuring execution time for benchmarking

Usage: python validate_kmeans.py <output_folder> [--input_points <path>] [--tolerance <float>]
"""

import argparse
import csv
import os
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import pandas as pd


def load_points_from_csv(filepath: str) -> np.ndarray:
    """Load points from CSV file."""
    try:
        return np.loadtxt(filepath, delimiter=',')
    except Exception as e:
        print(f"Error loading points from {filepath}: {e}")
        sys.exit(1)


def load_centers_from_file(filepath: str) -> np.ndarray:
    """Load final centers from output file."""
    try:
        return np.loadtxt(filepath, delimiter=',')
    except Exception as e:
        print(f"Error loading centers from {filepath}: {e}")
        sys.exit(1)


def load_assignments_from_file(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load point assignments from output file. Returns (point_ids, cluster_ids)."""
    try:
        data = np.loadtxt(filepath, delimiter=',', dtype=int)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        return data[:, 0], data[:, 1]
    except Exception as e:
        print(f"Error loading assignments from {filepath}: {e}")
        sys.exit(1)


def compute_distance_matrix(points: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """Compute distance matrix between all points and centers."""
    # Using broadcasting for efficient computation
    # points: (n_points, n_dims), centers: (n_centers, n_dims)
    # result: (n_points, n_centers)
    return np.sqrt(np.sum((points[:, np.newaxis, :] - centers[np.newaxis, :, :]) ** 2, axis=2))


def validate_assignments(points: np.ndarray, centers: np.ndarray, assignments: np.ndarray) -> bool:
    """Check if all points are assigned to their nearest center."""
    distances = compute_distance_matrix(points, centers)
    nearest_centers = np.argmin(distances, axis=1)
    
    misassigned = np.sum(nearest_centers != assignments)
    total_points = len(points)
    
    print(f"Assignment validation:")
    print(f"  Total points: {total_points}")
    print(f"  Correctly assigned: {total_points - misassigned}")
    print(f"  Misassigned: {misassigned}")
    print(f"  Accuracy: {((total_points - misassigned) / total_points) * 100:.2f}%")
    
    return misassigned == 0


def compute_inertia(points: np.ndarray, centers: np.ndarray, assignments: np.ndarray) -> float:
    """Compute the within-cluster sum of squares (inertia)."""
    inertia = 0.0
    for i, point in enumerate(points):
        cluster = assignments[i]
        inertia += np.sum((point - centers[cluster]) ** 2)
    return inertia


def compare_with_sklearn(points: np.ndarray, k: int, max_iter: int = 300, 
                        tolerance: float = 1e-4, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, float]:
    """Run sklearn K-Means for comparison."""
    print(f"\nRunning sklearn K-Means for comparison...")
    start_time = time.time()
    
    sklearn_kmeans = KMeans(
        n_clusters=k, 
        max_iter=max_iter, 
        tol=tolerance,
        random_state=random_state,
        n_init=1,
        init='k-means++'
    )
    
    sklearn_labels = sklearn_kmeans.fit_predict(points)
    sklearn_centers = sklearn_kmeans.cluster_centers_
    sklearn_inertia = sklearn_kmeans.inertia_
    
    sklearn_time = time.time() - start_time
    print(f"Sklearn K-Means completed in {sklearn_time:.4f} seconds")
    
    return sklearn_centers, sklearn_labels, sklearn_inertia


def align_clusters(our_centers: np.ndarray, our_labels: np.ndarray, 
                  ref_centers: np.ndarray, ref_labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Align cluster labels to match reference clustering (sklearn)."""
    k = len(our_centers)
    
    # Find best matching between our clusters and reference clusters
    distances = compute_distance_matrix(our_centers, ref_centers)
    
    # Use Hungarian algorithm or simple greedy matching
    used_ref = set()
    mapping = {}
    
    for our_idx in range(k):
        # Find closest unused reference cluster
        best_ref_idx = None
        best_distance = float('inf')
        
        for ref_idx in range(k):
            if ref_idx not in used_ref and distances[our_idx, ref_idx] < best_distance:
                best_distance = distances[our_idx, ref_idx]
                best_ref_idx = ref_idx
        
        if best_ref_idx is not None:
            mapping[our_idx] = best_ref_idx
            used_ref.add(best_ref_idx)
    
    # Remap our labels and centers
    aligned_labels = np.array([mapping.get(label, label) for label in our_labels])
    aligned_centers = np.zeros_like(ref_centers)
    
    for our_idx, ref_idx in mapping.items():
        aligned_centers[ref_idx] = our_centers[our_idx]
    
    return aligned_centers, aligned_labels


def main():
    parser = argparse.ArgumentParser(description='Validate K-Means MapReduce output')
    parser.add_argument('output_folder', help='Path to output folder containing results')
    parser.add_argument('--input_points', default='input/points.csv', 
                       help='Path to input points CSV file')
    parser.add_argument('--tolerance', type=float, default=1e-4,
                       help='Tolerance for center comparison')
    parser.add_argument('--max_iter', type=int, default=100,
                       help='Maximum iterations for sklearn comparison')
    
    args = parser.parse_args()
    
    # Validate paths
    output_folder = Path(args.output_folder)
    if not output_folder.exists():
        print(f"Error: Output folder {output_folder} does not exist")
        sys.exit(1)
    
    points_file = Path(args.input_points)
    if not points_file.exists():
        print(f"Error: Input points file {points_file} does not exist")
        sys.exit(1)
    
    centers_file = output_folder / "final_centers.txt"
    assignments_file = output_folder / "final_assignments.txt"
    
    if not centers_file.exists():
        print(f"Error: Final centers file {centers_file} does not exist")
        sys.exit(1)
        
    if not assignments_file.exists():
        print(f"Error: Final assignments file {assignments_file} does not exist")
        sys.exit(1)
    
    print("=" * 60)
    print("K-MEANS MAPREDUCE VALIDATION")
    print("=" * 60)
    
    # Load data
    print(f"Loading data...")
    points = load_points_from_csv(str(points_file))
    our_centers = load_centers_from_file(str(centers_file))
    point_ids, our_assignments = load_assignments_from_file(str(assignments_file))
    
    # Sort assignments by point_id to ensure correct order
    sort_indices = np.argsort(point_ids)
    our_assignments = our_assignments[sort_indices]
    
    k = len(our_centers)
    n_points, n_dims = points.shape
    
    print(f"Dataset info:")
    print(f"  Points: {n_points}")
    print(f"  Dimensions: {n_dims}")
    print(f"  Clusters: {k}")
    
    # Validate assignments
    print(f"\n" + "=" * 40)
    print("ASSIGNMENT VALIDATION")
    print("=" * 40)
    
    assignments_valid = validate_assignments(points, our_centers, our_assignments)
    
    # Compute our metrics
    our_inertia = compute_inertia(points, our_centers, our_assignments)
    
    try:
        our_silhouette = silhouette_score(points, our_assignments)
    except:
        our_silhouette = None
        print("Could not compute silhouette score (might need more clusters)")
    
    print(f"\nOur implementation metrics:")
    print(f"  Inertia (WCSS): {our_inertia:.6f}")
    if our_silhouette is not None:
        print(f"  Silhouette score: {our_silhouette:.6f}")
    
    # Compare with sklearn
    print(f"\n" + "=" * 40)
    print("SKLEARN COMPARISON")
    print("=" * 40)
    
    sklearn_centers, sklearn_labels, sklearn_inertia = compare_with_sklearn(
        points, k, args.max_iter, args.tolerance
    )
    
    try:
        sklearn_silhouette = silhouette_score(points, sklearn_labels)
    except:
        sklearn_silhouette = None
    
    print(f"\nSklearn metrics:")
    print(f"  Inertia (WCSS): {sklearn_inertia:.6f}")
    if sklearn_silhouette is not None:
        print(f"  Silhouette score: {sklearn_silhouette:.6f}")
    
    # Align clusters for fair comparison
    aligned_centers, aligned_labels = align_clusters(
        our_centers, our_assignments, sklearn_centers, sklearn_labels
    )
    
    # Compare centers
    center_distances = np.sqrt(np.sum((aligned_centers - sklearn_centers) ** 2, axis=1))
    max_center_diff = np.max(center_distances)
    avg_center_diff = np.mean(center_distances)
    
    print(f"\nCenter comparison (after alignment):")
    print(f"  Max center difference: {max_center_diff:.6f}")
    print(f"  Average center difference: {avg_center_diff:.6f}")
    print(f"  Centers match (tolerance={args.tolerance}): {max_center_diff <= args.tolerance}")
    
    # Compare inertias
    inertia_diff = abs(our_inertia - sklearn_inertia)
    inertia_rel_diff = inertia_diff / sklearn_inertia if sklearn_inertia > 0 else float('inf')
    
    print(f"\nInertia comparison:")
    print(f"  Absolute difference: {inertia_diff:.6f}")
    print(f"  Relative difference: {inertia_rel_diff:.6f} ({inertia_rel_diff*100:.2f}%)")
    
    # Final summary
    print(f"\n" + "=" * 40)
    print("VALIDATION SUMMARY")
    print("=" * 40)
    
    all_valid = True
    
    if assignments_valid:
        print("✓ All points correctly assigned to nearest centers")
    else:
        print("✗ Some points not assigned to nearest centers")
        all_valid = False
    
    if max_center_diff <= args.tolerance:
        print("✓ Centers match sklearn within tolerance")
    else:
        print("✗ Centers differ significantly from sklearn")
        all_valid = False
    
    if inertia_rel_diff <= 0.05:  # 5% tolerance (more realistic for different algorithms)
        print("✓ Inertia matches sklearn closely")
    else:
        print("✗ Inertia differs significantly from sklearn")
        if our_inertia < sklearn_inertia:
            print(f"  Note: Your implementation found a better solution!")
        all_valid = False
    
    if all_valid:
        print(f"\n🎉 VALIDATION PASSED: Implementation appears correct!")
    else:
        print(f"\n⚠️  VALIDATION FAILED: Implementation may have issues")
    
    return 0 if all_valid else 1


if __name__ == "__main__":
    sys.exit(main())