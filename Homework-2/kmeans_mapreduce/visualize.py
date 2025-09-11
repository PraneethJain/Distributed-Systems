import matplotlib.pyplot as plt
import numpy as np

# dataset = "sample"
dataset = "generated/dataset_10"
points = np.loadtxt(f"input/{dataset}/points.txt", delimiter=" ")
centers = np.loadtxt(f"output/{dataset}_cores_8/centers.txt", delimiter=" ")
assignments = np.loadtxt(f"output/{dataset}_cores_8/mapping.txt", delimiter=" ", dtype=int)[:, ::-1]

# Sort assignments by point ID
sorted_assignments = assignments[np.argsort(assignments[:, 0]), 1]

# Plot
plt.figure(figsize=(10, 8))
colors = ["red", "blue", "green", "purple", "orange", "brown"]

# Plot points colored by cluster
for cluster in range(len(centers)):
    cluster_points = points[sorted_assignments == cluster]
    if len(cluster_points) > 0:
        plt.scatter(
            cluster_points[:, 0],
            cluster_points[:, 1],
            c=colors[cluster % len(colors)],
            alpha=0.6,
            s=20,
            label=f"Cluster {cluster}",
        )

# Plot centers
plt.scatter(
    centers[:, 0],
    centers[:, 1],
    c="black",
    marker="x",
    s=200,
    linewidth=3,
    label="Centers",
)

plt.legend()
plt.title("K-Means Clustering Results")
plt.savefig("kmeans_result.png")
