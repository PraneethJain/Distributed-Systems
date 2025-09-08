use csv;
use kmeans_mapreduce::{
    combine_assignment_files, convert_flat_to_points, find_nearest_center,
    squared_euclidean_distance, write_assignments, write_centers, Point,
};
use mpi::traits::*;
use std::error::Error;

fn read_points_from_csv(path: &str) -> Vec<Point> {
    csv::ReaderBuilder::new()
        .has_headers(false)
        .from_path(path)
        .expect("Failed to open CSV file")
        .records()
        .filter_map(|result| {
            result.ok().map(|record| {
                record
                    .iter()
                    .map(|s| s.parse::<f64>().expect("Failed to parse float"))
                    .collect::<Point>()
            })
        })
        .collect()
}

fn main() -> Result<(), Box<dyn Error>> {
    let universe = mpi::initialize().unwrap();
    let world = universe.world();
    let rank = world.rank() as usize;
    let size = world.size() as usize;

    let num_points = 1000;
    let num_centers = 5;
    let dimensions = 2;
    let max_iterations = 100;
    let tolerance = 1e-4;

    let local_points = scatter_points_to_workers(&world, rank, size, num_points, dimensions)?;
    let mut centers = broadcast_centers(&world, rank, num_centers, dimensions)?;

    let mut iteration = 0;
    let final_assignments = loop {
        println!("Process {}: Starting iteration {}", rank, iteration);
        // MAP PHASE
        let (local_assignments, local_sums, local_counts) =
            map_and_combine(&local_points, &centers, num_centers, dimensions);

        // REDUCE PHASE (only master gets global results)
        let global_results = reduce_phase(
            &world,
            &local_sums,
            &local_counts,
            rank,
            num_centers,
            dimensions,
        )?;

        // MASTER: Compute new centers and check convergence
        let (new_centers, converged) = if rank == 0 {
            let (global_sums, global_counts) = global_results.unwrap();
            let (new_centers, converged) = compute_new_centers_and_check_convergence(
                &global_sums,
                &global_counts,
                &centers,
                num_centers,
                dimensions,
                tolerance,
            );
            (Some(new_centers), Some(converged))
        } else {
            (None, None)
        };

        // Broadcast new centers and convergence status
        let (updated_centers, is_converged) = broadcast_centers_and_convergence(
            &world,
            rank,
            new_centers,
            converged,
            num_centers,
            dimensions,
        )?;

        centers = updated_centers;

        if is_converged {
            break Some(local_assignments);
        }

        iteration += 1;
        if iteration >= max_iterations {
            break None;
        }
    };

    println!("Process {}: K-means complete", rank);

    if let Some(final_assignments) = final_assignments {
        println!(
            "Process {}: Converged after {} iterations",
            rank,
            iteration + 1
        );
        write_assignments(&local_points, &final_assignments, rank, "output")?;
    } else {
        println!(
            "Process {}: Reached max iterations ({})",
            rank, max_iterations
        );
    }

    if rank == 0 {
        write_centers(&centers, "output")?;
        combine_assignment_files(size, "output")?;
    }

    Ok(())
}

fn scatter_points_to_workers(
    world: &mpi::topology::SimpleCommunicator,
    rank: usize,
    size: usize,
    total_points: usize,
    dimensions: usize,
) -> Result<Vec<Point>, Box<dyn Error>> {
    let points_per_worker = total_points.div_ceil(size);
    let total_elements_per_worker = points_per_worker * dimensions;
    let total_elements = size * total_elements_per_worker;

    let mut local_buffer = vec![0.0; total_elements_per_worker];

    if rank == 0 {
        let points_path = "input/points.csv";
        let all_points = read_points_from_csv(points_path);

        assert_eq!(all_points.len(), total_points, "Total points mismatch");
        assert_eq!(all_points[0].len(), dimensions, "Point dimension mismatch");

        let mut flat_points: Vec<f64> = all_points
            .iter()
            .flat_map(|point| point.iter().cloned())
            .collect();
        flat_points.resize(total_elements, 0.0);
        world
            .process_at_rank(0)
            .scatter_into_root(&flat_points, &mut local_buffer);
    } else {
        world.process_at_rank(0).scatter_into(&mut local_buffer);
    }

    let start_idx = rank * points_per_worker;
    let actual_count = std::cmp::min(points_per_worker, total_points.saturating_sub(start_idx));

    let local_points = convert_flat_to_points(&local_buffer, dimensions, actual_count);
    Ok(local_points)
}

fn broadcast_centers(
    world: &mpi::topology::SimpleCommunicator,
    rank: usize,
    num_centers: usize,
    dimensions: usize,
) -> Result<Vec<Point>, Box<dyn Error>> {
    let total_elements = num_centers * dimensions;
    let mut buffer = vec![0.0; total_elements];

    if rank == 0 {
        let centers_path = "input/centers.txt";
        let centers = read_points_from_csv(centers_path);

        assert_eq!(centers.len(), num_centers, "Number of centers mismatch");
        assert_eq!(centers[0].len(), dimensions, "Center dimension mismatch");

        let flat_centers: Vec<f64> = centers
            .iter()
            .flat_map(|point| point.iter().cloned())
            .collect();

        buffer.copy_from_slice(&flat_centers);
    }

    world.process_at_rank(0).broadcast_into(&mut buffer);

    let centers = convert_flat_to_points(&buffer, dimensions, num_centers);
    Ok(centers)
}

fn map_and_combine(
    local_points: &[Point],
    centers: &[Point],
    num_centers: usize,
    dimensions: usize,
) -> (Vec<usize>, Vec<Vec<f64>>, Vec<i32>) {
    let mut local_assignments = Vec::with_capacity(local_points.len());

    let mut local_sums = vec![vec![0.0; dimensions]; num_centers];
    let mut local_counts = vec![0; num_centers];

    // Assign each local point to nearest center
    for point in local_points {
        let nearest_cluster = find_nearest_center(point, centers);
        local_assignments.push(nearest_cluster);

        for d in 0..dimensions {
            local_sums[nearest_cluster][d] += point[d];
        }
        local_counts[nearest_cluster] += 1;
    }

    (local_assignments, local_sums, local_counts)
}

fn reduce_phase(
    world: &mpi::topology::SimpleCommunicator,
    local_sums: &[Vec<f64>],
    local_counts: &[i32],
    rank: usize,
    num_centers: usize,
    dimensions: usize,
) -> Result<Option<(Vec<Vec<f64>>, Vec<i32>)>, Box<dyn Error>> {
    if rank == 0 {
        // Master: receive global results
        let mut global_counts = vec![0i32; num_centers];
        let mut global_sums = vec![vec![0.0f64; dimensions]; num_centers];

        // Reduce cluster counts to master
        world.process_at_rank(0).reduce_into_root(
            local_counts,
            &mut global_counts,
            mpi::collective::SystemOperation::sum(),
        );

        // Reduce cluster sums to master (cluster by cluster)
        for cluster in 0..num_centers {
            world.process_at_rank(0).reduce_into_root(
                &local_sums[cluster],
                &mut global_sums[cluster],
                mpi::collective::SystemOperation::sum(),
            );
        }

        // Verification prints (only master)
        let total_points: i32 = global_counts.iter().sum();
        println!(
            "Master: Global reduce complete - total points: {}",
            total_points
        );

        for cluster in 0..num_centers {
            println!(
                "Master: Cluster {} has {} total points",
                cluster, global_counts[cluster]
            );
            if global_counts[cluster] > 0 {
                println!(
                    "Master: Cluster {} sum: {:?}",
                    cluster,
                    &global_sums[cluster][..2.min(dimensions)]
                );
            }
        }

        Ok(Some((global_sums, global_counts)))
    } else {
        // Workers: send their local data to master
        world
            .process_at_rank(0)
            .reduce_into(local_counts, mpi::collective::SystemOperation::sum());

        // Send local sums to master (cluster by cluster)
        for cluster in 0..num_centers {
            world.process_at_rank(0).reduce_into(
                &local_sums[cluster],
                mpi::collective::SystemOperation::sum(),
            );
        }

        println!("Process {}: Sent local data to master for reduction", rank);
        Ok(None)
    }
}

fn compute_new_centers_and_check_convergence(
    global_sums: &[Vec<f64>],
    global_counts: &[i32],
    old_centers: &[Point],
    num_centers: usize,
    dimensions: usize,
    tolerance: f64,
) -> (Vec<Point>, bool) {
    let mut new_centers = vec![vec![0.0; dimensions]; num_centers];

    // Compute new centers
    for cluster in 0..num_centers {
        if global_counts[cluster] > 0 {
            for d in 0..dimensions {
                new_centers[cluster][d] = global_sums[cluster][d] / global_counts[cluster] as f64;
            }
        } else {
            // Keep old center if no points assigned
            new_centers[cluster] = old_centers[cluster].clone();
            println!(
                "Warning: Cluster {} has no assigned points, keeping old center",
                cluster
            );
        }
    }

    // Check convergence
    let mut converged = true;
    for (cluster, (old, new)) in old_centers.iter().zip(new_centers.iter()).enumerate() {
        let distance = squared_euclidean_distance(old, new);
        println!(
            "Master: Cluster {} center moved by distance: {:.6}",
            cluster, distance
        );
        if distance > tolerance {
            converged = false;
        }
    }

    println!("Master: Convergence check - converged: {}", converged);
    (new_centers, converged)
}

fn broadcast_centers_and_convergence(
    world: &mpi::topology::SimpleCommunicator,
    rank: usize,
    new_centers: Option<Vec<Point>>,
    converged: Option<bool>,
    num_centers: usize,
    dimensions: usize,
) -> Result<(Vec<Point>, bool), Box<dyn Error>> {
    // Broadcast new centers
    let total_elements = num_centers * dimensions;
    let mut buffer = vec![0.0; total_elements];

    if rank == 0 {
        let centers = new_centers.unwrap();
        let flat_centers: Vec<f64> = centers
            .iter()
            .flat_map(|point| point.iter().cloned())
            .collect();
        buffer.copy_from_slice(&flat_centers);
        println!("Master: Broadcasting new centers");
    }

    world.process_at_rank(0).broadcast_into(&mut buffer);
    let centers = convert_flat_to_points(&buffer, dimensions, num_centers);

    // Broadcast convergence status
    let mut converged_buf = if rank == 0 { converged.unwrap() } else { false };
    world.process_at_rank(0).broadcast_into(&mut converged_buf);

    if rank != 0 {
        println!(
            "Process {}: Received new centers and convergence status: {}",
            rank, converged_buf
        );
    }

    Ok((centers, converged_buf))
}
