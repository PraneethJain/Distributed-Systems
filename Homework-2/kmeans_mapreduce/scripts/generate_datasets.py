import numpy as np
import os
from dataset_configs import dataset_configs


def generate_dataset(
    num_points, num_centers, dimensions, dataset_id, output_dir="test_data"
):
    points = np.random.rand(num_points, dimensions) * 100
    centers = np.random.rand(num_centers, dimensions) * 100

    dataset_path = os.path.join(output_dir, f"dataset_{dataset_id}")
    os.makedirs(dataset_path, exist_ok=True)

    points_file = os.path.join(dataset_path, "points.txt")
    centers_file = os.path.join(dataset_path, "centers.txt")

    np.savetxt(points_file, points, fmt="%.6f")
    np.savetxt(centers_file, centers, fmt="%.6f")

    print(
        f"Generated dataset {dataset_id}: {num_points} points, {num_centers} centers in {dataset_path}"
    )
    return points_file, centers_file


if __name__ == "__main__":
    output_base_dir = "./input/generated"
    os.makedirs(output_base_dir, exist_ok=True)

    # Define parameters for 8 datasets

    for i, (num_points, num_centers, dimensions) in enumerate(dataset_configs):
        generate_dataset(num_points, num_centers, dimensions, i + 1, output_base_dir)
